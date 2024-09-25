import torch

def get_batch(split: str, batch_size: int, context_length: int, train_data, val_data, device):
    batch_data = train_data if split == "train" else val_data
    ix = torch.randint(len(batch_data) - context_length, (batch_size,))

    x = torch.stack([batch_data[i:i+context_length] for i in ix])
        
    y = torch.stack([batch_data[i+1:i+context_length+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    
    return x, y