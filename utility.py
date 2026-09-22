import torch
from torch.utils.data import DataLoader
import numpy as np


def compute_global_stats(dataset, batch_size=16):
    """Computes global mean and std across the dataset using Welford's-like batching."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    pixel_count = 0
    sum_bands = np.zeros(13, dtype=np.float64)
    sum_sq_bands = np.zeros(13, dtype=np.float64)

    print("Computing global statistics...")
    for i, (images, _, _) in enumerate(loader):
        # images shape: (batch_size, 13, 512, 512)
        b, c, h, w = images.shape
        pixels_in_batch = b * h * w

        # Reshape to (13, -1) to sum across all pixels for each band
        images_flat = images.numpy().transpose(1, 0, 2, 3).reshape(c, -1)

        sum_bands += images_flat.sum(axis=1)
        sum_sq_bands += (images_flat ** 2).sum(axis=1)
        pixel_count += pixels_in_batch

        if i % 10 == 0:
            print(f"Processed batch {i}...")

    mean = sum_bands / pixel_count
    # Variance = E[X^2] - (E[X])^2
    std = np.sqrt((sum_sq_bands / pixel_count) - (mean ** 2))

    return mean, std


import torch
from torch.utils.data import DataLoader


def compute_full_class_distribution(dataset, batch_size=16):
    print("\n--- Computing Full Pixel Class Distribution ---")
    # Using num_workers=4 to speed up NetCDF I/O
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = {0: 'MISSING', 1: 'CLEAR', 2: 'CLOUD SHADOW',
                   3: 'SEMI TRANSPARENT CLOUD', 4: 'CLOUD', 5: 'UNDEFINED'}

    global_counts = torch.zeros(6, dtype=torch.int64)

    for i, (_, masks, _) in enumerate(loader):
        # Flatten the entire batch of masks and count integer occurrences
        counts = torch.bincount(masks.flatten(), minlength=6)
        global_counts += counts

        if i % 50 == 0:
            print(f"Counted batch {i}...")

    total_pixels = global_counts.sum().item()

    print("\n--- Final Dataset Distribution ---")
    for cls_idx in range(6):
        count = global_counts[cls_idx].item()
        pct = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"{class_names[cls_idx]:<25}: {count:>12} pixels ({pct:>5.2f}%)")

    print("\n--- Recommended E1 Loss Weights (Inverse Frequency) ---")
    # Calculates weights prioritizing minority classes, while masking 0 and 5
    for cls_idx in range(6):
        count = global_counts[cls_idx].item()
        if cls_idx in [0, 5] or count == 0:
            weight = 0.0
        else:
            # We divide by 4.0 because there are 4 valid classes we care about learning
            weight = total_pixels / (4.0 * count)
        print(f"Weight Class {cls_idx} ({class_names[cls_idx]}): {weight:.4f}")

# --- Execution ---
# compute_full_class_distribution(raw_dataset, batch_size=4)

# --- Execution ---
# Ensure you only pass the training split directory here, not the test split!
# train_dataset = Sentinel2CloudMaskDataset(train_dir)
# global_mean, global_std = compute_global_stats(train_dataset)
# print(f"GLOBAL_MEAN = {list(global_mean)}")
# print(f"GLOBAL_STD = {list(global_std)}")
