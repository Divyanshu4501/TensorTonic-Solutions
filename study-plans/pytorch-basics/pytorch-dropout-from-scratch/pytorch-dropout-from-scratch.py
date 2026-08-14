import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        if self.p == 1:
            return torch.zeros_like(x)
        if not self.training:
            return x
        m = (torch.rand_like(x) >= self.p).float()
        return m * x / (1-self.p)
        
