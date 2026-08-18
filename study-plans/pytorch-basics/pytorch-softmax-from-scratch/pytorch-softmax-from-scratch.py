import torch

def softmax(logits):
    logits = torch.tensor(logits, dtype = torch.float32)
    max_values, _ = torch.max(logits, dim=1, keepdim = True)
    shifted = torch.tensor(logits - max_values, dtype=torch.float32)
    exps = torch.exp(shifted)
    return exps/exps.sum(dim = 1, keepdim = True)
    
