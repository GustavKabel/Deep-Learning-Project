import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import random
from pathlib import Path

def set_seed(seed):
    """Makes weight initialization and batch order reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(path, model, optimizer, epoch, config_dict=None):
    """Saves model and optimizer state to survive disconnects."""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config_dict
    }, path)

def load_checkpoint(path, model, optimizer, device):
    """Loads state to resume training."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('config', {})

# ==========================================
# 1. PyTorch Dataset Wrapper for NetCDF (13 Bands)
# ==========================================
class Sentinel2CloudMaskDataset(Dataset):
    def __init__(self, root_dir, global_mean=None, global_std=None):
        """
        Custom PyTorch Dataset for loading Sentinel-2 NetCDF datacubes.
        """
        # Recursively find all .nc files in the provided directory (e.g., train/ or test/)
        self.files = sorted(list(Path(root_dir).rglob("*.nc")))
        self.global_mean = global_mean
        self.global_std = global_std

        # The 13 spectral bands provided in the KappaZeta dataset
        self.bands = ["B01", "B02", "B03", "B04", "B05", "B06",
                      "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        try:
            # Open the NetCDF file using the h5netcdf engine
            with xr.open_dataset(file_path, engine="h5netcdf") as ds:
                # Extract each band and stack them into a (13, Height, Width) tensor
                band_arrays = [ds[band].values for band in self.bands]
                img_data = np.stack(band_arrays, axis=0).astype(np.float32)

                # Extract the ground truth label mask (Height, Width)
                mask_data = ds['Label'].values.astype(np.int64)

                # Apply Z-score standardization if stats are provided via config.py
                if self.global_mean is not None and self.global_std is not None:
                    mean_arr = np.array(self.global_mean).reshape(-1, 1, 1)
                    std_arr = np.array(self.global_std).reshape(-1, 1, 1)
                    img_data = (img_data - mean_arr) / (std_arr + 1e-8)

        except Exception as e:
            # Fallback for corrupted NetCDF files to prevent the DataLoader from crashing
            print(f"Error loading {file_path}: {e}")
            img_data = np.zeros((13, 512, 512), dtype=np.float32)
            mask_data = np.zeros((512, 512), dtype=np.int64)

        return torch.tensor(img_data, dtype=torch.float32), torch.tensor(mask_data, dtype=torch.long), str(file_path)


# ==========================================
# 2. Dataset Statistic Utilities
# ==========================================
def compute_global_stats(dataset, batch_size=16):
    """Computes global mean and std across the dataset using Welford's-like batching."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    pixel_count = 0
    sum_bands = np.zeros(13, dtype=np.float64)
    sum_sq_bands = np.zeros(13, dtype=np.float64)

    print("Computing global statistics...")
    for i, (images, _, _) in enumerate(loader):
        b, c, h, w = images.shape
        pixels_in_batch = b * h * w

        images_flat = images.numpy().transpose(1, 0, 2, 3).reshape(c, -1)

        sum_bands += images_flat.sum(axis=1)
        sum_sq_bands += (images_flat ** 2).sum(axis=1)
        pixel_count += pixels_in_batch

        if i % 10 == 0:
            print(f"Processed batch {i}...")

    mean = sum_bands / pixel_count
    std = np.sqrt((sum_sq_bands / pixel_count) - (mean ** 2))

    return mean, std


def compute_full_class_distribution(dataset, batch_size=16):
    """Computes class imbalance weights based on inverse frequency."""
    print("\n--- Computing Full Pixel Class Distribution ---")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = {0: 'MISSING', 1: 'CLEAR', 2: 'CLOUD SHADOW',
                   3: 'SEMI TRANSPARENT CLOUD', 4: 'CLOUD', 5: 'UNDEFINED'}

    global_counts = torch.zeros(6, dtype=torch.int64)

    for i, (_, masks, _) in enumerate(loader):
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
    for cls_idx in range(6):
        count = global_counts[cls_idx].item()
        # Classes 0 and 5 are zeroed out (masked) per the project proposal
        if cls_idx in [0, 5] or count == 0:
            weight = 0.0
        else:
            weight = total_pixels / (4.0 * count)
        print(f"Weight Class {cls_idx} ({class_names[cls_idx]}): {weight:.4f}")