import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from pathlib import Path

import config


# ==========================================
# 1. PyTorch Dataset Wrapper for NetCDF (13 Bands)
# ==========================================
class Sentinel2CloudMaskDataset(Dataset):
    def __init__(self, root_dir, global_mean=None, global_std=None):
        self.files = sorted(list(Path(root_dir).rglob("*.nc")))
        self.global_mean = global_mean
        self.global_std = global_std
        self.bands = ["B01", "B02", "B03", "B04", "B05", "B06",
                      "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        try:
            with xr.open_dataset(file_path, engine="h5netcdf") as ds:
                band_arrays = [ds[band].values for band in self.bands]
                img_data = np.stack(band_arrays, axis=0).astype(np.float32)
                mask_data = ds['Label'].values.astype(np.int64)

                if self.global_mean is not None and self.global_std is not None:
                    mean_arr = np.array(self.global_mean).reshape(-1, 1, 1)
                    std_arr = np.array(self.global_std).reshape(-1, 1, 1)
                    img_data = (img_data - mean_arr) / (std_arr + 1e-8)

        except Exception:
            img_data = np.zeros((13, 512, 512), dtype=np.float32)
            mask_data = np.zeros((512, 512), dtype=np.int64)

        return torch.tensor(img_data), torch.tensor(mask_data), str(file_path)


# ==========================================
# 2. Fast Lab Checks for Semantic Segmentation
# ==========================================
def check_pixel_class_distribution_fast(dataset, sample_size=50):
    print("\n--- Check 2: Fast Pixel Class Distribution (Subset) ---")
    class_names = {0: 'MISSING', 1: 'CLEAR', 2: 'CLOUD SHADOW',
                   3: 'SEMI TRANSPARENT CLOUD', 4: 'CLOUD', 5: 'UNDEFINED'}
    class_counts = {k: 0 for k in class_names.keys()}

    num_to_check = min(sample_size, len(dataset))
    for i in range(num_to_check):
        _, mask, _ = dataset[i]
        counts = torch.bincount(mask.flatten(), minlength=6)
        for cls_idx in range(6):
            class_counts[cls_idx] += counts[cls_idx].item()

    total = sum(class_counts.values())
    for cls_idx, count in class_counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        print(f"{class_names[cls_idx]:<25}: {count:>10} pixels ({pct:>5.1f}%)")


def check_nc_image_sizes(dataset_dir, sample_size=100):
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
    else:
        print("Could not extract dimensions.")


def check_nc_tensor_stats(loader, n_batches=5):
    print("\n--- Check 4: Tensor Value Range ---")
    all_means, all_stds = [], []

    for i, (images, _, _) in enumerate(loader):
        if i >= n_batches: break
        all_means.append(images.mean(dim=[0, 2, 3]).numpy())
        all_stds.append(images.std(dim=[0, 2, 3]).numpy())

    mean = np.array(all_means).mean(axis=0)
    std = np.array(all_stds).mean(axis=0)

    print('Tensor statistics per band (Values should now be mean ≈ 0.0, std ≈ 1.0):')
    for c in range(len(mean)):
        print(f'  Band {c + 1:<2}: mean={mean[c]:>6.3f}, std={std[c]:>6.3f}')

    if abs(mean).max() > 0.1 or abs(std.mean() - 1.0) > 0.2:
        print('WARNING: Data is not cleanly standardized (mean ≈ 0, std ≈ 1).')
    else:
        print('SUCCESS: Tensors are successfully Z-score standardized.')


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    base_dir = Path("/home/gustav/Desktop/sentinel2_cloudmask_kz")

    check_nc_image_sizes(base_dir, sample_size=100)

    print("\n--- Initializing Dataset with Config Stats ---")
    norm_dataset = Sentinel2CloudMaskDataset(
        base_dir,
        global_mean=config.GLOBAL_MEAN,
        global_std=config.GLOBAL_STD
    )

    if len(norm_dataset) > 0:
        check_pixel_class_distribution_fast(norm_dataset, sample_size=50)

        loader = DataLoader(norm_dataset, batch_size=4, shuffle=True)
        check_nc_tensor_stats(loader, n_batches=5)