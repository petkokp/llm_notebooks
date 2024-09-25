import torch
from .estimate_loss import estimate_loss
from .get_batch import get_batch

def train_model(model, max_iterations, iterations_interval, eval_iterations, batch_size, learning_rate, context_length, train_data, val_data, device):
    train_losses = []
    val_losses = []
    steps = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for iteration in range(max_iterations):
        if iteration % iterations_interval == 0:
            losses = estimate_loss(model, eval_iterations, batch_size, context_length, train_data, val_data, device)
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            steps.append(iteration)
            
        xb, yb = get_batch("train", batch_size, context_length, train_data, val_data, device)
        
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    return train_losses, val_losses, steps