import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        W = torch.randn(self.out_features, self.in_features) * (2.0/in_features)**0.5
        b = torch.zeros(out_features)

        self.weight = nn.Parameter(W)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        return torch.matmul(x, self.weight.T) + self.bias