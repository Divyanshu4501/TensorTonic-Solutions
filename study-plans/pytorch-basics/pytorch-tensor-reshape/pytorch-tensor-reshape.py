import torch

def reshape_tensor(x, op):
    A = torch.tensor(x, dtype = torch.float32)
    if op == 'flatten':
        return torch.flatten(A).tolist()

    if op == 'squeeze':
        return torch.squeeze(A).tolist()

    if op == 'transpose':
        return torch.transpose(A, 0, 1).tolist()