import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # pip install tqdm for progress bars

# Import from your existing files
from utility import Sentinel2CloudMaskDataset
import config
from model import get_unet_resnet50

# --- Hyperparameters ---
BATCH_SIZE = 8  # Reduce to 4 if your GPU runs out of memory (OOM)
EPOCHS = 10
LEARNING_RATE = 1e-4

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Dataset
train_dataset = Sentinel2CloudMaskDataset(
    root_dir="/home/gustav/Desktop/sentinel2_cloudmask_kz/train",  # Adjust path to your train split
    global_mean=config.GLOBAL_MEAN,
    global_std=config.GLOBAL_STD
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 2. Initialize Model, Loss, and Optimizer
model = get_unet_resnet50(in_channels=13, num_classes=6).to(device)

# E0 Baseline: Standard Cross-Entropy (No class weights yet)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # tqdm creates a nice progress bar in your terminal
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for images, masks, _ in loop:
        images = images.to(device)
        masks = masks.to(device)  # Shape: (Batch, 512, 512)

        # 1. Forward pass
        logits = model(images)  # Shape: (Batch, 6, 512, 512)

        # 2. Calculate loss
        loss = criterion(logits, masks)

        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"End of Epoch {epoch + 1} | Average Training Loss: {epoch_loss:.4f}\n")

    # Save a checkpoint after each epoch
    torch.save(model.state_dict(), f"unet_resnet50_e0_epoch{epoch + 1}.pth")