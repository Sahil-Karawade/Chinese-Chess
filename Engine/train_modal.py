# train_modal.py
import glob
import modal
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
from .model import XiangqiNet
from .device_config import device
from .modal_config import app
# Modal setup
torch_image = modal.Image.debian_slim().pip_install("torch", "numpy", "matplotlib")
vol = modal.Volume.from_name("xiangqi-vol", create_if_missing=True)

@app.function(image=torch_image, volumes={"/vol": vol}, gpu="enter_name", timeout=3*60*60)
def train_modal():
    print("Running training on Modal Cloud...")

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    DATA_DIR = "/vol/self_play_data" # Path inside the volume where training data is saved
    MODEL_PATH = "/vol/checkpoint.pt"
    PLOT_PATH = "/vol/training_loss.png"
    LR = 1e-3
    ALPHA = 0.5  # Weight for policy loss

    def load_data(data_dir):
        files = sorted(glob.glob(f"{data_dir}/game_*.pkl"))
        all_data = []
        for file in files:
            with open(file, "rb") as f:
                game_data = pickle.load(f)
                all_data.extend(game_data)
        # Check data validity
        assert all(isinstance(x[0], np.ndarray) for x in all_data), "States must be NumPy arrays"
        assert all(x[1].shape == (8100,) for x in all_data), "Policy targets must be 8100-dim"
        return all_data


    #def load_data(path):
    #    with open(path, "rb") as f:
    #        data = pickle.load(f)
    #    assert all(isinstance(x[0], np.ndarray) for x in data), "States must be NumPy arrays"
    #    assert all(x[1].shape == (8100,) for x in data), "Policy targets must be 8100-dim"
    #   return data

    # Model & optimizer
    model = XiangqiNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Load data
    data = load_data(DATA_DIR)
    print(f"Loaded {len(data)} samples from {DATA_DIR}")

    # Tracking
    epoch_losses = []
    batch_loss_history = []
    policy_loss_history = []
    value_loss_history = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        np.random.shuffle(data)

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            states = torch.stack([torch.from_numpy(s) for s, _, _ in batch]).to(device)
            pis = torch.from_numpy(np.array([p for _, p, _ in batch], dtype=torch.float32)).to(device)
            vs = torch.tensor([v for _, _, v in batch], dtype=torch.float32).unsqueeze(1).to(device)

            pred_pi, pred_v = model(states)
            loss_policy = nn.MSELoss()(pred_pi, pis)
            loss_value = nn.MSELoss()(pred_v, vs)
            loss = ALPHA * loss_policy + (1 - ALPHA) * loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            batch_loss_history.append(loss.item())
            policy_loss_history.append(loss_policy.item())
            value_loss_history.append(loss_value.item())

            if i % (10 * BATCH_SIZE) == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(data)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / (len(data) // BATCH_SIZE)
        epoch_losses.append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Total Loss: {avg_loss:.4f} | "
              f"Policy Loss: {total_policy_loss:.4f} | "
              f"Value Loss: {total_value_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'loss': avg_loss
    }, MODEL_PATH)

    # Plot losses
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epoch_losses, 'b-o')
    plt.title('Epoch Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(batch_loss_history, 'r-', alpha=0.5)
    plt.title('Batch Loss History')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(policy_loss_history, 'g-', alpha=0.5, label='Policy')
    plt.plot(value_loss_history, 'm-', alpha=0.5, label='Value')
    plt.title('Component Losses')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    vol.commit()

    print(f"Model saved to {MODEL_PATH}")
    print(f"Loss plots saved to {PLOT_PATH}")

