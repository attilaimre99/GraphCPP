import torch
import torch.nn as nn

class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

act_dict = {
    'relu': nn.ReLU(inplace=False),
    'selu': nn.SELU(inplace=False),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(inplace=False),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1, inplace=False),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25, inplace=False),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5, inplace=False),
    'swish': SWISH(inplace=False),
}

