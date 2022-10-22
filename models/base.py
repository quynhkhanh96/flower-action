import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.checkpoint import load_checkpoint

class Recognizer3D(nn.Module):
    def __init__(self, backbone, neck=None, cls_head=None):
        super().__init__()  
        self.backbone_from = 'mmaction2'      
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
        self.cls_head = cls_head if cls_head else None
        # max_testing_views should be int
        self.max_testing_views = None
        self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        self.blending = None
        self.init_weights()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None
    
    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_neck:
            self.neck.init_weights()
    
    # @auto_fp16()
    def extract_feat(self, imgs):
        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            x = self.backbone.features(imgs)
        elif self.backbone_from == 'timm':
            x = self.backbone.forward_features(imgs)
        elif self.backbone_from == 'mmcls':
            x = self.backbone(imgs)
            if isinstance(x, tuple):
                assert len(x) == 1
                x = x[0]
        else:
            x = self.backbone(imgs)
        return x
    
    def average_clip(self, cls_score, num_segs=1):
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)

        return cls_score

    def forward(self, imgs):
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)

        return cls_score

def build_mmaction_model(cfgs, mode='train'):
    if cfgs.arch == 'slowonly_r50':
        from .backbones.resnet3d_slowonly import ResNet3dSlowOnly
        from .heads.i3d_head import I3DHead
        backbone = ResNet3dSlowOnly(depth=50, pretrained='torchvision://resnet50',
            lateral=False, conv1_kernel=(1, 7, 7), conv1_stride_t=1,
            pool1_stride_t=1, inflate=(0, 0, 1, 1), norm_eval=False
        )
        cls_head = I3DHead(in_channels=2048, num_classes=51,
            spatial_type='avg', dropout_ratio=0.5)
        model = Recognizer3D(backbone=backbone, cls_head=cls_head)
        if mode == 'train':
            load_checkpoint(model, cfgs.pretrained_model)
    else:
        raise ValueError(f'No implementation for {cfgs.arch}.')

    return model