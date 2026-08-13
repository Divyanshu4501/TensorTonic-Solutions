import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features

        self.linear1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        h = self.linear1(x)
        h = self.relu(h)
        y = self.linear2(h)
        return y
        
