import torch
from .estimate_loss import estimate_loss
from .get_batch import get_batch

n_embeddings = 384
n_heads = 6
n_layers = 6
max_iterations = 5000
iterations_interval = 100
eval_iterations = 200
batch_size = 64
learning_rate = 1e-4
context_length = 256
dropout = 0.2

def train_model(model, max_iterations=max_iterations, iterations_interval=iterations_interval, eval_iterations=eval_iterations, batch_size=batch_size, learning_rate=learning_rate, context_length=context_length):
    train_losses = []
    val_losses = []
    steps = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for iteration in range(max_iterations):
        if iteration % iterations_interval == 0:
            losses = estimate_loss(model, eval_iterations, batch_size, context_length)
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            steps.append(iteration)
            
        xb, yb = get_batch("train", batch_size, context_length)
        
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    return train_losses, val_losses, steps