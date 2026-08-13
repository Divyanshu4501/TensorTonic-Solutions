import torch

def activate(x, method="relu"):
    nums = torch.tensor(x, dtype = torch.float32)
    if method == "relu":
        return torch.where(nums>0, nums, torch.zeros_like(nums)).tolist()

    if method == "sigmoid":
        return (1/(1+torch.exp(-nums))).tolist()

    if method == "tanh":
        return ((torch.exp(nums) - torch.exp(-nums))/(torch.exp(nums) + torch.exp(-nums))).tolist()

    if method == "leaky_relu":
        return torch.where(nums>0, nums, 0.01*nums).tolist()