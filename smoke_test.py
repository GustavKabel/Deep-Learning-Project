import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import from your existing files
from utility import Sentinel2CloudMaskDataset
import config
from model import get_unet_resnet50

# --- Hyperparameters for Smoke Test ---
BATCH_SIZE = 4  # Small batch to easily fit in your 16GB of RAM
EPOCHS = 1  # Only run one pass to verify backpropagation
LEARNING_RATE = 1e-4

# --- Setup ---
# This will default to CPU on your Dell XPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Dataset
full_dataset = Sentinel2CloudMaskDataset(
    root_dir="/home/gustav/Desktop/sentinel2_cloudmask_kz/train",
    global_mean=config.TRAIN_GLOBAL_MEAN,
    global_std=config.TRAIN_GLOBAL_STD
)

# 2. Isolate a tiny subset for local CPU testing
# We grab just the first 16 images so the test finishes in a few minutes
subset_indices = list(range(32))
smoke_test_dataset = Subset(full_dataset, subset_indices)
train_loader = DataLoader(smoke_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Running smoke test on {len(smoke_test_dataset)} images...")

# 3. Initialize Model, Loss, and Optimizer
model = get_unet_resnet50(in_channels=13, num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc="Smoke Test Epoch")

    for images, masks, file_paths in loop:
        images = images.to(device)
        masks = masks.to(device)

        # 1. Forward pass
        logits = model(images)

        # 2. Calculate loss
        loss = criterion(logits, masks)

        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"\nSuccess! End of Smoke Test | Average Training Loss: {epoch_loss:.4f}")

    # Verify we can successfully save the checkpoint to your local drive
    torch.save(model.state_dict(), "smoke_test_weights.pth")
    print("Checkpoint saved to smoke_test_weights.pth")