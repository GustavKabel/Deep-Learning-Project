import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex

from utility import Sentinel2CloudMaskDataset, save_checkpoint, load_checkpoint
import config
from model import get_unet_resnet50

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3  # Optimal LR from the sweep

# --- Checkpoint Resuming ---
# Example: "/content/drive/MyDrive/unet_resnet50_E1_epoch6.pth"
RESUME_CHECKPOINT = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load FULL Dataset (No Subsets)
train_dataset = Sentinel2CloudMaskDataset(
    root_dir="/content/train",
    global_mean=config.TRAIN_GLOBAL_MEAN,
    global_std=config.TRAIN_GLOBAL_STD
)

# 2. A100 Optimized DataLoader (Batch 32, 8 workers)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

# 3. Initialize Model, Loss (with weights), and Optimizer
model = get_unet_resnet50(in_channels=13, num_classes=6).to(device)

# --- E1 MODIFICATION: Inject inverse frequency class weights ---
class_weights = torch.tensor(config.LOSS_WEIGHTS, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Resume from Checkpoint (if provided)
start_epoch = 0
if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    print(f"Resuming from {RESUME_CHECKPOINT}...")
    start_epoch, _ = load_checkpoint(RESUME_CHECKPOINT, model, optimizer, device)

# 5. Initialize Metric
iou_metric = MulticlassJaccardIndex(num_classes=6, average="none").to(device)

# --- Training Loop ---
for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    iou_metric.reset()

    # Updated progress bar to reflect E1
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (E1 Weighted)")

    for images, masks, _ in loop:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        iou_metric.update(logits, masks)
        loop.set_postfix(loss=loss.item())

    # Calculate Epoch Metrics
    epoch_loss = running_loss / len(train_loader)
    per_class_iou = iou_metric.compute()

    # We only care about the 4 valid classes (1: Clear, 2: Shadow, 3: Semi-trans, 4: Cloud)
    valid_ious = per_class_iou[1:5]
    macro_miou = valid_ious.mean().item()

    print(f"\n--- End of Epoch {epoch + 1} (E1) ---")
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Macro mIoU (Valid Classes): {macro_miou:.4f}")
    print(f"Class 2 (Cloud Shadow) IoU: {per_class_iou[2].item():.4f}")
    print("-" * 30 + "\n")

    # --- E1 MODIFICATION: Save checkpoint to a unique E1 path ---
    save_path = f"/content/drive/MyDrive/unet_resnet50_E1_epoch{epoch + 1}.pth"
    save_checkpoint(save_path, model, optimizer, epoch + 1, {'lr': LEARNING_RATE})
    print(f"Checkpoint saved to {save_path}\n")