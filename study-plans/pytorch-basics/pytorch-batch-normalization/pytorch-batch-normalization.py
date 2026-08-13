import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    data = torch.tensor(X, dtype=torch.float32)
    mean = data.mean(dim = 0)
    var = data.var(dim = 0, unbiased = False)
    X_norm = (data - mean)/(var+eps)**0.5
    return torch.tensor(gamma, dtype=torch.float32)*X_norm + torch.tensor(beta, dtype=torch.float32)