import torch

def tensor_op(x, y, op):
    A, B = torch.tensor(x), torch.tensor(y)
    if op == "add":
        return torch.add(A,B).tolist()
        
    if op == "multiply":
        return torch.mul(A,B).tolist()
        
    if op == "matmul":
        return torch.matmul(A, B).tolist()

    if op == "power":
        return torch.pow(A,B).tolist()
    
    if op == "max":
        return torch.max(A,B).tolist()

