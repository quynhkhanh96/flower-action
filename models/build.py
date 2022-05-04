from __future__ import absolute_import
import os 
import torch 
import torch.nn as nn
import torch.optim as optim

from . import resnet3d
from . import c3d
from movinets import MoViNet
from movinets.config import _C
from .efficientnet_pytorch_3d import EfficientNet3D
from . import mobilenet3d_v2
from .slow_fast_r2plus1d import slow_fast_r3d_18
# import init_model

from .ResNet import *

__factory = {
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TA,
    'resnet50rnn': ResNet50RNN,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)

def build_model(cfgs, mode='train'):
    if cfgs.arch == 'resnet503d':
        model = resnet3d.resnet50(num_classes=cfgs.num_classes, 
                                sample_width=cfgs.width, sample_height=cfgs.height, 
                                sample_duration=cfgs.seq_len)
        if mode == 'train':
            if not os.path.exists(cfgs.pretrained_model):
                raise IOError("Can't find pretrained model: {}".format(cfgs.pretrained_model))
            print("Loading checkpoint from '{}'".format(cfgs.pretrained_model))
            checkpoint = torch.load(cfgs.pretrained_model)
            state_dict = {}
            for key in checkpoint['state_dict']:
                if 'fc' in key: continue
                state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict, strict=False)
    elif cfgs.arch =='resnet183d':
        model = resnet3d.resnet18(num_classes=cfgs.num_classes, 
                                sample_width=cfgs.width, sample_height=cfgs.height, 
                                sample_duration=cfgs.seq_len)		
        if mode == 'train':
            if not os.path.exists(cfgs.pretrained_model):
                raise IOError("Can't find pretrained model: {}".format(cfgs.pretrained_model))
            print("Loading checkpoint from '{}'".format(cfgs.pretrained_model))
            checkpoint = torch.load(cfgs.pretrained_model)
            state_dict = {}
            for key in checkpoint['state_dict']:
                if 'fc' in key: continue
                state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict, strict=False)	

    elif cfgs.arch == 'c3d_bn':
        model = c3d.c3d_bn(num_classes=cfgs.num_classes, dropout=0.2)
        if mode == 'train':
            if not os.path.exists(cfgs.pretrained_model):
                raise IOError("Can't find pretrained model: {}".format(cfgs.pretrained_model))
            print("Loading checkpoint from '{}'".format(cfgs.pretrained_model))
            checkpoint = torch.load(cfgs.pretrained_model)
            model.load_state_dict(checkpoint)	

    elif cfgs.arch == 'movinet_a0':
        model = MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=True)
        model.classifier[3] = torch.nn.Conv3d(2048, cfgs.num_classes, (1, 1, 1))
    
    elif cfgs.arch == 'movinet_a2':
        model = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained = True)
        model.classifier[3] = torch.nn.Conv3d(2048, cfgs.num_classes, (1, 1, 1))
    
    elif cfgs.arch == 'movinet_a5':
        model = MoViNet(_C.MODEL.MoViNetA5, causal=False, pretrained=True)
        model.classifier[3] = torch.nn.Conv3d(2048, cfgs.num_classes, (1, 1, 1))
    
    elif cfgs.arch == 'efficientnet3d':
        model = EfficientNet3D.from_name("efficientnet-b0", 
                                override_params={'num_classes': cfgs.num_classes}, 
                                in_channels=3)
    
    elif cfgs.arch[:14] == 'mobilenet3d_v2':
        model = mobilenet3d_v2.mobilenet3d_v2(num_classes=cfgs.num_classes,width_mult=cfgs.width_mult)
        if not os.path.exists(cfgs.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(cfgs.pretrained_model))
        print("Loading checkpoint from '{}'".format(cfgs.pretrained_model))
        checkpoint = torch.load(cfgs.pretrained_model)
        model.load_state_dict(checkpoint,strict=False)
    
    elif cfgs.arch == 'slow_fast_r3d_18':
        model = slow_fast_r3d_18(num_classes=cfgs.num_classes,
                                pretrained=False, progress=True,
                                alpha=4, beta=8) # 64//beta -->8	
    else:
        model = init_model(name=cfgs.arch, num_classes=cfgs.num_classes, loss={'xent', 'htri'})
    
    return model 

def build_loss(cfgs):
    if cfgs.loss == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'No implementation for {cfgs.loss} loss.')

    return loss 

def build_optimizer(cfgs, model):
    if cfgs.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfgs.lr, momentum=0.9)
    else:
        raise ValueError(f'No implementation for {cfgs.optimizer} optimizer.')

    return optimizer

