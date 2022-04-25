"""
This is the c3d implementation.

References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
Proceedings of the IEEE international conference on computer vision. 2015.
"""

import torch
import torch.nn as nn


__all__ = [
    'c3d',
    'c3d_bn',
]


class C3D(nn.Module):

    def __init__(self,
                 num_classes=487,
                 dropout=0.25):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        # init weights
        self._initialize_weights()

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3, 4)  # (b, c, s, w, h)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.flatten(1)
        x_feat = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x_feat))
        x = self.fc8(x)
        return x, x_feat

    def _initialize_weights(self):
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

    def load_state_dict(self, state_dict, strict=False):
        current_state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key in current_state_dict.keys() and state_dict[key].shape != current_state_dict[key].shape:
                print(f"[Warning] Key {key} has incompatible shape of {state_dict[key].shape}, "
                      f"expecting {current_state_dict[key].shape}.")
                state_dict.pop(key)
        super().load_state_dict(state_dict, strict)


class C3DBatchNorm(nn.Module):

    def __init__(self,
                 num_classes=400,
                 dropout=0.25):
        super(C3DBatchNorm, self).__init__()

        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1a_bn = nn.BatchNorm3d(64, eps=1e-3, momentum=0.9)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2a_bn = nn.BatchNorm3d(128, eps=1e-3, momentum=0.9)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3a_bn = nn.BatchNorm3d(256, eps=1e-3, momentum=0.9)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b_bn = nn.BatchNorm3d(256, eps=1e-3, momentum=0.9)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4a_bn = nn.BatchNorm3d(512, eps=1e-3, momentum=0.9)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b_bn = nn.BatchNorm3d(512, eps=1e-3, momentum=0.9)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5a_bn = nn.BatchNorm3d(512, eps=1e-3, momentum=0.9)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b_bn = nn.BatchNorm3d(512, eps=1e-3, momentum=0.9)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(4608, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        # init weights
        self._initialize_weights()

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3, 4)  # (b, c, s, w, h)
        x = self.relu(self.conv1a_bn(self.conv1a(x)))
        x = self.pool1(x)

        x = self.relu(self.conv2a_bn(self.conv2a(x)))
        x = self.pool2(x)

        x = self.relu(self.conv3a_bn(self.conv3a(x)))
        x = self.relu(self.conv3b_bn(self.conv3b(x)))
        x = self.pool3(x)

        x = self.relu(self.conv4a_bn(self.conv4a(x)))
        x = self.relu(self.conv4b_bn(self.conv4b(x)))
        x = self.pool4(x)

        x = self.relu(self.conv5a_bn(self.conv5a(x)))
        x = self.relu(self.conv5b_bn(self.conv5b(x)))
        x = self.pool5(x)

        x = x.flatten(1)
        x_feat = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x_feat))
        x = self.fc8(x)
        return x

    def _initialize_weights(self):
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

    def load_state_dict(self, state_dict, strict=False):
        current_state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key in current_state_dict.keys() and state_dict[key].shape != current_state_dict[key].shape:
                print(f"[Warning] Key {key} has incompatible shape of {state_dict[key].shape}, "
                      f"expecting {current_state_dict[key].shape}.")
                state_dict.pop(key)
        super().load_state_dict(state_dict, strict)


def c3d(**kwargs):
    """Construct original C3D network as described in [1].
    """
    return C3D(**kwargs)


def c3d_bn(**kwargs):
    """Construct the modified C3D network with batch normalization hosted in github Video Model Zoo.
    """
    return C3DBatchNorm(**kwargs)


if __name__ == '__main__':
    model = c3d(num_classes=12)

    inputs = torch.randn(5, 3, 16, 112, 112)
    output = model(inputs)
    print(output.shape)
