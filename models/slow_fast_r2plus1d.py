from typing import Tuple, Optional, Callable, List, Type, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .fusion.slow_fast_lateral_connection import SlowFastLateralConnection
from .fusion import ConcatenateFusionBlock
from .parallel_module_list import ParallelModuleList

__all__ = ['SlowFastVideoResNet',
           'slow_fast_r3d_18',
           'slow_fast_mc3_18',
           'slow_fast_r2plus1d_18',
           'slow_fast_r1plus2d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class NonDegenerateTemporalConv3DSimple(nn.Conv3d):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            midplanes: Optional[int] = None,
            stride: int = 1,
            padding: int = 1,
            no_temporal: bool = False
    ) -> None:
        super(NonDegenerateTemporalConv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1 if no_temporal else 3, 3, 3),
            stride=(1, stride, stride),
            padding=(0 if no_temporal else 1, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class NonDegenerateTemporalConv2Plus1D(nn.Sequential):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            midplanes: int,
            stride: int = 1,
            padding: int = 1,
            no_temporal: bool = False
    ) -> None:
        super(NonDegenerateTemporalConv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(1 if no_temporal else 3, 1, 1),
                      stride=(1, 1, 1), padding=(0 if no_temporal else 1, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class NonDegenerateTemporalConv1Plus2D(nn.Sequential):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            midplanes: int,
            stride: int = 1,
            padding: int = 1,
            no_temporal: bool = False
    ) -> None:
        super(NonDegenerateTemporalConv1Plus2D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1 if no_temporal else 3, 1, 1),
                      stride=(1, 1, 1), padding=(0 if no_temporal else 1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            mid_planes: Optional[int] = None,
            stride: int = 1,
            padding: int = 1,
            no_temporal: bool = False
    ) -> None:
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            conv_builder: Callable[..., nn.Module],
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            no_temporal: bool = False
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride, no_temporal=no_temporal),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, no_temporal=no_temporal),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            conv_builder: Callable[..., nn.Module],
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            no_temporal: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride, no_temporal=no_temporal),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self,
                 planes: int = 64,
                 no_temporal: bool = False) -> None:
        super(BasicStem, self).__init__(
            nn.Conv3d(3, planes, kernel_size=(1 if no_temporal else 3, 7, 7),
                      stride=(1, 2, 2), padding=(0 if no_temporal else 1, 3, 3),
                      bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self,
                 planes: int = 64,
                 no_temporal: bool = False) -> None:
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, planes, kernel_size=(1 if no_temporal else 3, 1, 1),
                      stride=(1, 1, 1), padding=(0 if no_temporal else 1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))


class R1Plus2dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self,
                 planes: int = 64,
                 no_temporal: bool = False) -> None:
        super(R1Plus2dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1 if no_temporal else 3, 1, 1),
                      stride=(1, 1, 1), padding=(0 if no_temporal else 1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, planes, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))


class SlowFastVideoResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            conv_makers: List[
                Type[Union[
                    NonDegenerateTemporalConv3DSimple,
                    Conv3DNoTemporal,
                    NonDegenerateTemporalConv2Plus1D,
                    NonDegenerateTemporalConv1Plus2D,
                ]]],
            layers: List[int],
            stem: Callable[..., nn.Module],
            num_classes: int = 400,
            zero_init_residual: bool = False,
            alpha: int = 4,
            beta: int = 8,
            order: str = 'slow_fast'
    ) -> None:
        """SlowFast resnet video generator.
        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[NonDegenerateTemporalConv3DSimple, Conv3DNoTemporal, NonDegenerateTemporalConv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(SlowFastVideoResNet, self).__init__()
        self._inplanes = [64, 64 // beta]
        self.alpha = alpha
        self.beta = beta

        self.stem = ParallelModuleList([
            stem(planes=self._inplanes[0], no_temporal=True),
            stem(planes=self._inplanes[1]),
        ])
        self.lc_stem = SlowFastLateralConnection(slow_dim=64,
                                                 fast_dim=64 // beta,
                                                 fusion_kernel=5,
                                                 alpha=alpha,
                                                 order=order)

        self.layer1 = ParallelModuleList([
            self._make_layer(0, block, conv_makers[0], 64, layers[0], stride=1, no_temporal=True),
            self._make_layer(1, block, conv_makers[0], 64 // beta, layers[0], stride=1),
        ])
        self.lc1 = SlowFastLateralConnection(slow_dim=64,
                                             fast_dim=64 // beta,
                                             fusion_kernel=5,
                                             alpha=alpha,
                                             order=order)

        self.layer2 = ParallelModuleList([
            self._make_layer(0, block, conv_makers[1], 128, layers[1], stride=2, no_temporal=True),
            self._make_layer(1, block, conv_makers[1], 128 // beta, layers[1], stride=2),
        ])
        self.lc2 = SlowFastLateralConnection(slow_dim=128,
                                             fast_dim=128 // beta,
                                             fusion_kernel=5,
                                             alpha=alpha,
                                             order=order)

        self.layer3 = ParallelModuleList([
            self._make_layer(0, block, conv_makers[2], 256, layers[2], stride=2),
            self._make_layer(1, block, conv_makers[2], 256 // beta, layers[2], stride=2),
        ])
        self.lc3 = SlowFastLateralConnection(slow_dim=256,
                                             fast_dim=256 // beta,
                                             fusion_kernel=5,
                                             alpha=alpha,
                                             order=order)

        self.layer4 = ParallelModuleList([
            self._make_layer(0, block, conv_makers[3], 512, layers[3], stride=2),
            self._make_layer(1, block, conv_makers[3], 512 // beta, layers[3], stride=2),
        ])
        self.lc4 = SlowFastLateralConnection(slow_dim=512,
                                             fast_dim=512 // beta,
                                             fusion_kernel=5,
                                             alpha=alpha,
                                             order=order)

        self.avgpool = ParallelModuleList([
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        ])
        self.fuse = ConcatenateFusionBlock(num_streams=2, dim=1)
        self.fc = nn.Linear((512 + 512 // beta) * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x_fast = x
        x_slow = x.index_select(2,
                                torch.linspace(0, x.size(2) - 1, self.alpha,
                                               dtype=torch.long, device=x.device))
        x = [x_slow, x_fast]

        x = self.stem(x)
        x = self.lc_stem(x)
        # print('stem', [_.shape for _ in x])

        x = self.layer1(x)
        x = self.lc1(x)
        # print('layer1', [_.shape for _ in x])
        x = self.layer2(x)
        x = self.lc2(x)
        # print('layer2', [_.shape for _ in x])
        x = self.layer3(x)
        x = self.lc3(x)
        # print('layer3', [_.shape for _ in x])
        x = self.layer4(x)
        x = self.lc4(x)
        # print('layer4', [_.shape for _ in x])

        x = self.avgpool(x)
        # print('avgpool', [_.shape for _ in x])
        x = self.fuse(x).flatten(1)
        # print('fuse', x.shape)
        x = self.fc(x)
        # print('fc', x.shape)

        return x

    def _make_layer(
            self,
            pathway_id: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            conv_builder: Type[
                Union[NonDegenerateTemporalConv3DSimple, Conv3DNoTemporal, NonDegenerateTemporalConv2Plus1D]],
            planes: int,
            blocks: int,
            stride: int = 1,
            no_temporal: bool = False
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self._inplanes[pathway_id] != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self._inplanes[pathway_id], planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = [block(self._inplanes[pathway_id], planes, conv_builder, stride, downsample, no_temporal=no_temporal)]

        self._inplanes[pathway_id] = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self._inplanes[pathway_id], planes, conv_builder, no_temporal=no_temporal))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _slow_fast_video_resnet(arch: str, pretrained: bool = False, progress: bool = True,
                            **kwargs: Any) -> SlowFastVideoResNet:
    model = SlowFastVideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 400:
            if 'fc.weight' in state_dict:
                state_dict.pop('fc.weight')
            if 'fc.bias' in state_dict:
                state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


def slow_fast_r3d_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SlowFastVideoResNet:
    """Construct 18 layer SlowFast Resnet3D model as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: R3D-18 network
    """

    return _slow_fast_video_resnet('r3d_18',
                                   pretrained, progress,
                                   block=BasicBlock,
                                   conv_makers=[NonDegenerateTemporalConv3DSimple] * 4,
                                   layers=[2, 2, 2, 2],
                                   stem=BasicStem, **kwargs)


def slow_fast_mc3_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SlowFastVideoResNet:
    """Constructor for 18 layer SlowFast Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: MC3 Network definition
    """
    return _slow_fast_video_resnet('mc3_18',
                                   pretrained, progress,
                                   block=BasicBlock,
                                   conv_makers=[NonDegenerateTemporalConv3DSimple] + [Conv3DNoTemporal] * 3,
                                   # type: ignore[list-item]
                                   layers=[2, 2, 2, 2],
                                   stem=BasicStem, **kwargs)


def slow_fast_r2plus1d_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SlowFastVideoResNet:
    """Constructor for the 18 layer SlowFast R(2+1)D network as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _slow_fast_video_resnet('r2plus1d_18',
                                   pretrained, progress,
                                   block=BasicBlock,
                                   conv_makers=[NonDegenerateTemporalConv2Plus1D] * 4,
                                   layers=[2, 2, 2, 2],
                                   stem=R2Plus1dStem, **kwargs)


def slow_fast_r1plus2d_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SlowFastVideoResNet:
    """Constructor for the 18 layer SlowFast R(2+1)D network as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _slow_fast_video_resnet('r2plus1d_18',
                                   pretrained, progress,
                                   block=BasicBlock,
                                   conv_makers=[NonDegenerateTemporalConv1Plus2D] * 4,
                                   layers=[2, 2, 2, 2],
                                   stem=R1Plus2dStem, **kwargs)
