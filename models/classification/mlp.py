import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
# from ..config import Args
import flwr 

class MLP(nn.Module):
    """
    define MLP model
    """
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fp_layer1 = nn.Linear(784, 30, bias=False)

        self.ternary_layer1 = nn.Linear(30, 20, bias=False)

        self.fp_layer2 = nn.Linear(20, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # x = x.cpu().view(-1, 784).to(Args.device)
        x = x.cpu().view(-1, 784).to(self.device)

        x = self.fp_layer1(x)
        x = self.ternary_layer1(x)
        x = self.fp_layer2(x)

        output = F.log_softmax(x, dim=1)
        return output

def Quantized_MLP(pre_model, args):
    """
    quantize the MLP model
    :param pre_model:
    :param args:
    :return:
    """

    #full-precision first and last layer
    weights = [p for n, p in pre_model.named_parameters() if 'fp_layer' in n and 'weight' in n]
    biases = [pre_model.fp_layer2.bias]

    #layers that need to be quantized
    ternary_weights = [p for n, p in pre_model.named_parameters() if 'ternary' in n]

    params = [
        {'params': weights},
        {'params': ternary_weights},
        {'params': biases}
    ]

    optimizer = optim.SGD(params, lr=args.lr)
    loss_fun = nn.CrossEntropyLoss()

    return pre_model, loss_fun, optimizer
