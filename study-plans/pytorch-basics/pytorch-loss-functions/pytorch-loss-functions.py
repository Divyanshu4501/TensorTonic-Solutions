import torch

def compute_loss(pred, target, method, delta=1.0):
    if method == "mse":
        y_pred = torch.tensor(pred, dtype=torch.float32)
        y_true = torch.tensor(target, dtype=torch.float32)
        return torch.nn.functional.mse_loss(y_pred, y_true)

    if method == "cross_entropy":
        y_pred = torch.tensor(pred, dtype=torch.float32)   # logits
        y_true = torch.tensor(target, dtype=torch.long)    # class indices
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_pred, y_true)

    if method == "huber":
        y_pred = torch.tensor(pred, dtype=torch.float32)
        y_true = torch.tensor(target, dtype=torch.float32)
        loss_fn = torch.nn.HuberLoss(delta=delta)
        return loss_fn(y_pred, y_true)