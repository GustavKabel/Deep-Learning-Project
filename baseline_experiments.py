import time
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassJaccardIndex

from utility import Sentinel2CloudMaskDataset, set_seed
from model import get_unet_resnet50
import config

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_CLASSES = 6
ROOT_DIR = "/content/train"  # Update for Colab local storage

# Load full dataset, but we will subset it for fast experiments
full_train_dataset = Sentinel2CloudMaskDataset(
    root_dir=ROOT_DIR, global_mean=config.TRAIN_GLOBAL_MEAN, global_std=config.TRAIN_GLOBAL_STD
)

# Use only the first 500 images for fast screening runs
subset_indices = list(range(500))
train_dataset = Subset(full_train_dataset, subset_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

loss_fn = nn.CrossEntropyLoss()
EXPERIMENT_LOG = []


def run_experiment(name, lr=1e-4, epochs=3, seed=0, log=True):
    """Runs a fast screening experiment and logs results."""
    set_seed(seed)
    model = get_unet_resnet50(in_channels=13, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    diverged = False
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()

            if not torch.isfinite(loss):
                diverged = True
                break

            running_loss += loss.item() * len(images)
            seen += len(images)

        if diverged:
            print(f"  [{name} | seed {seed}] diverged (loss = nan) in epoch {epoch + 1}")
            break

    runtime = time.time() - start_time
    final_loss = running_loss / seen if seen > 0 else float('nan')

    result = {'name': name, 'lr': lr, 'epochs': epochs, 'seed': seed,
              'diverged': diverged, 'train_loss': final_loss, 'runtime_s': round(runtime, 1)}

    if log:
        EXPERIMENT_LOG.append(result)
        pd.DataFrame(EXPERIMENT_LOG).to_csv('experiment_log.csv', index=False)

    if not diverged:
        print(f"  [{name} | seed {seed}] train loss {final_loss:.4f} | {runtime:.0f}s")

    return model


if __name__ == "__main__":
    print("--- 1. Learning Rate Sweep ---")
    SWEEP_LRS = [1e-5, 1e-4, 1e-3, 1e-2]
    for lr in SWEEP_LRS:
        run_experiment(f'sweep_lr={lr:g}', lr=lr, epochs=3, seed=0)

    print("\n--- 2. Noise Floor Establishment (Chosen LR) ---")
    # Update this value based on which LR performed best in the sweep above!
    BEST_LR = 1e-4
    for seed in [0, 1, 2]:
        run_experiment('E0_baseline', lr=BEST_LR, epochs=5, seed=seed)