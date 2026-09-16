import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from pathlib import Path


# ==========================================
# 1. PyTorch Dataset Wrapper for NetCDF
# ==========================================
class Sentinel2CloudMaskDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(list(Path(root_dir).rglob("*.nc")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        try:
            with xr.open_dataset(file_path, engine="h5netcdf") as ds:
                # Stack the True Color Image (RGB) bands and scale to [0, 1]
                r = ds['TCI_R'].values
                g = ds['TCI_G'].values
                b = ds['TCI_B'].values
                img_data = np.stack([r, g, b], axis=0) / 255.0

                # Extract the cloud mask
                mask_data = ds['Label'].values

        except Exception as e:
            # Fallback for corrupt files to prevent crashes
            img_data = np.zeros((3, 512, 512))
            mask_data = np.zeros((512, 512))

        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_data, dtype=torch.long)

        return img_tensor, mask_tensor, str(file_path)


# ==========================================
# 2. Adapted Lab Checks for Semantic Segmentation
# ==========================================

def check_pixel_class_distribution(dataset, sample_size=100):
    """Adapted Check 2: Counts pixel-wise class distribution across all KZ classes."""
    print("\n--- Check 2: Pixel Class Distribution ---")

    class_names = {
        0: 'MISSING',
        1: 'CLEAR',
        2: 'CLOUD SHADOW',
        3: 'SEMI TRANSPARENT CLOUD',
        4: 'CLOUD',
        5: 'UNDEFINED'
    }

    # Initialize a counter for classes 0 through 5
    class_counts = {k: 0 for k in class_names.keys()}

    num_to_check = min(sample_size, len(dataset))
    for i in range(num_to_check):
        _, mask, _ = dataset[i]

        # Flatten the 512x512 mask and count occurrences of each integer (0-5)
        counts = torch.bincount(mask.flatten(), minlength=6)
        for cls_idx in range(6):
            class_counts[cls_idx] += counts[cls_idx].item()

    total = sum(class_counts.values())
    if total == 0:
        print("Error: No pixels found. Check variable mapping.")
        return

    # Print the distribution breakdown
    for cls_idx, count in class_counts.items():
        pct = (count / total) * 100
        print(f"{class_names[cls_idx]:<25}: {count:>10} pixels ({pct:>5.1f}%)")

    # Check general imbalance (Clear vs All Cloud types)
    clear_px = class_counts[1]
    cloud_px = class_counts[2] + class_counts[3] + class_counts[4]

    if clear_px > 0 and cloud_px > 0:
        ratio = max(clear_px, cloud_px) / max(min(clear_px, cloud_px), 1)
        print(f"\nClear vs. All Clouds Ratio: {ratio:.1f}x")
        if ratio > 5:
            print("WARNING: High class imbalance between clear and cloudy pixels — consider weighted loss.")
        else:
            print("Class balance (Clear vs Cloudy) looks reasonable.")


def check_nc_image_sizes(dataset_dir, sample_size=200):
    """Adapted Check 3: Checks spatial dimensions of NetCDF files."""
    print("\n--- Check 3: Image Size Distribution ---")
    files = list(Path(dataset_dir).rglob("*.nc"))
    import random
    sampled = random.sample(files, min(sample_size, len(files)))

    widths, heights = [], []
    for path in sampled:
        try:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                dims = list(ds.dims)
                h_dim = 'y' if 'y' in dims else dims[1]
                w_dim = 'x' if 'x' in dims else dims[2]

                heights.append(ds.sizes[h_dim])
                widths.append(ds.sizes[w_dim])
        except Exception:
            pass

    if widths and heights:
        print(f"Width:  min={min(widths)}, max={max(widths)}, median={int(np.median(widths))}")
        print(f"Height: min={min(heights)}, max={max(heights)}, median={int(np.median(heights))}")
        if max(widths) / max(min(widths), 1) > 1.2:
            print("WARNING: Variation in image sizes — resizing or cropping may be needed.")
    else:
        print("Could not extract dimensions. Please check your NetCDF dimension names.")


def check_nc_tensor_stats(loader, n_batches=5):
    """Adapted Check 4: Checks tensor value ranges after scaling."""
    print("\n--- Check 4: Tensor Value Range ---")
    all_means, all_stds = [], []
    for i, (images, _, _) in enumerate(loader):
        if i >= n_batches: break
        all_means.append(images.mean(dim=[0, 2, 3]).numpy())
        all_stds.append(images.std(dim=[0, 2, 3]).numpy())

    mean = np.array(all_means).mean(axis=0)
    std = np.array(all_stds).mean(axis=0)

    print('After scaling (Values should now be between 0.0 and 1.0):')
    for c, name in enumerate(['Band 1 (R)', 'Band 2 (G)', 'Band 3 (B)'][:len(mean)]):
        print(f'  {name}: mean={mean[c]:.3f}, std={std[c]:.3f}')

    if abs(mean).max() > 1.0 or mean.min() < 0:
        print('WARNING: Means are outside the [0, 1] range. Check standardisation.')


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    base_dir = Path("/home/gustav/Desktop/sentinel2_cloudmask_kz")

    check_nc_image_sizes(base_dir, sample_size=100)

    dataset = Sentinel2CloudMaskDataset(base_dir)

    if len(dataset) > 0:
        check_pixel_class_distribution(dataset, sample_size=50)

        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        check_nc_tensor_stats(loader, n_batches=5)