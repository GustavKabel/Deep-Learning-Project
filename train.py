import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex

# Import from your existing files
from utility import Sentinel2CloudMaskDataset
import config
from model import get_unet_resnet50

# --- Hyperparameters ---
BATCH_SIZE = 8  # A cloud GPU with 16GB VRAM (like a T4) can handle batch size 8 or 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Dataset (Update root_dir for your cloud environment paths later!)
train_dataset = Sentinel2CloudMaskDataset(
    root_dir="/path/to/your/cloud/sentinel2_splits/train",
    global_mean=config.TRAIN_GLOBAL_MEAN,
    global_std=config.TRAIN_GLOBAL_STD
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 2. Initialize Model, Loss, and Optimizer
model = get_unet_resnet50(in_channels=13, num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Initialize Metric
# We use average="none" so we can see the exact score for each specific class
iou_metric = MulticlassJaccardIndex(num_classes=6, average="none").to(device)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    iou_metric.reset()  # Reset metric at the start of each epoch

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for images, masks, _ in loop:
        images = images.to(device)
        masks = masks.to(device)

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
    # Classes 0 (Missing) and 5 (Undefined) are ignored in our macro average
    valid_ious = per_class_iou[1:5]
    macro_miou = valid_ious.mean().item()

    print(f"\n--- End of Epoch {epoch + 1} ---")
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Macro mIoU (Valid Classes): {macro_miou:.4f}")
    print(f"Class 2 (Cloud Shadow) IoU: {per_class_iou[2].item():.4f}")
    print("-" * 30 + "\n")

    torch.save(model.state_dict(), f"unet_resnet50_e0_epoch{epoch + 1}.pth")