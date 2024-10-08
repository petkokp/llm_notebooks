import torch
from .get_batch import get_batch

@torch.no_grad()
def estimate_loss(model, iterations, batch_size, context_length, train_data, val_data, device):
    estimation = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(iterations)
        for k in range(iterations):
            X, Y = get_batch(split, batch_size, context_length, train_data, val_data, device)
            X, Y = X.to(device), Y.to(device)
            
            _, loss = model(X, Y)
            losses[k] = loss.item()
        estimation[split] = losses.mean()
    model.train()
    return estimation