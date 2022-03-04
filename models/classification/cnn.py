import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import flwr 

# Moderate size of CNN for CIFAR-10 dataset
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.fp_con1 = nn.Sequential(OrderedDict([
            ('con0', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)),
            ('relu0', nn.ReLU(inplace=True)),
            ]))

        self.ternary_con2 = nn.Sequential(OrderedDict([
            # Conv Layer block 1
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Layer block 2
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))


        self.fp_fc = nn.Linear(4096, num_classes, bias = False)

    def get_weights(self) -> flwr.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.fp_con1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        output = F.log_softmax(x, dim=1)
        return output


def Quantized_CNN(pre_model, args):
    """
    quantize the moderated cnn model
    :param pre_model:
    :param args:
    :return:
    """

    #full-precision first and last layer
    weights = [p for n, p in pre_model.named_parameters() if 'fp' in n and 'weight' in n]
    biases = [p for n, p in pre_model.named_parameters() if 'fp' in n and 'bias' in n]

    #layers that need to be quantized
    ternary_weights = [p for n, p in pre_model.named_parameters() if 'ternary' in n and 'conv' in n]

    #weights and biases of batch normlization layer
    bn_weights = [p for n, p in pre_model.named_parameters() if 'norm' in n and 'weight' in n]
    bn_biases = [p for n, p in pre_model.named_parameters() if 'norm' in n and 'bias' in n]

    params = [
        {'params': weights},
        {'params': ternary_weights},
        {'params': biases},

        {'params': bn_weights},
        {'params': bn_biases}
    ]

    optimizer = optim.Adam(params, lr=args.lr)
    loss_fun = nn.CrossEntropyLoss()

    return pre_model, loss_fun, optimizer

