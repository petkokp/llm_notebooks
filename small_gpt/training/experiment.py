import matplotlib.pyplot as plt
from .train_model import train_model

def experiment():
    model = None # TODO
    
    train_losses, val_losses, steps = train_model(model, learning_rate=1e-3, batch_size=32, max_iterations=10000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(steps, train_losses, label="Training loss")
    ax1.plot(steps, val_losses, label="Validation loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.set_title("Model training and validation loss (lr = 1e-3)")
    ax1.legend()
    
    model = None # TODO

    train_losses, val_losses, steps = train_model(model, learning_rate=1e-2, batch_size=32, max_iterations=10000)

    ax2.plot(steps, train_losses, label="Training loss")
    ax2.plot(steps, val_losses, label="Validation loss")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    ax2.set_title("Model training and validation loss (lr = 1e-2)")
    ax2.legend()
    plt.tight_layout()
    plt.show()