import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


def create_leakage_free_splits(raw_data_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Groups subscenes by their parent satellite product and splits them
    into train, val, and test folders to prevent spatial data leakage.
    """
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)

    # 1. Group files by parent scene ID
    # Sentinel-2/KappaZeta patches usually share the parent product name as a prefix,
    # followed by patch coordinates (e.g., S2A_MSIL1C_202005..._patch_1_2.nc)
    parent_groups = defaultdict(list)

    for file_path in raw_path.rglob("*.nc"):
        # We assume the parent ID is everything before the first "_patch" or similar suffix.
        # If your filenames are formatted differently, adjust this split logic!
        filename = file_path.name
        if "_patch" in filename:
            parent_id = filename.split("_patch")[0]
        else:
            # Fallback: group by the first 30 characters (usually captures the date/tile)
            parent_id = filename[:30]

        parent_groups[parent_id].append(file_path)

    parent_ids = list(parent_groups.keys())
    print(
        f"Found {len(parent_ids)} unique parent scenes containing {sum(len(v) for v in parent_groups.values())} total patches.")

    # 2. Shuffle parent IDs safely
    random.seed(42)  # Set seed for reproducibility!
    random.shuffle(parent_ids)

    # 3. Calculate split indices based on parent IDs
    num_parents = len(parent_ids)
    train_idx = int(num_parents * train_ratio)
    val_idx = train_idx + int(num_parents * val_ratio)

    train_parents = parent_ids[:train_idx]
    val_parents = parent_ids[train_idx:val_idx]
    test_parents = parent_ids[val_idx:]

    # 4. Create directories and copy files
    splits = {
        "train": train_parents,
        "val": val_parents,
        "test": test_parents
    }

    for split_name, parents in splits.items():
        split_dir = out_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        patch_count = 0
        for parent_id in parents:
            for file_path in parent_groups[parent_id]:
                # Copy the file to its new home
                shutil.copy(file_path, split_dir / file_path.name)
                patch_count += 1

        print(f"Created '{split_name}' split: {len(parents)} scenes, {patch_count} patches.")


if __name__ == "__main__":
    # Point directly to where the raw NetCDF files actually live
    RAW_DATA = "/home/gustav/Desktop/sentinel2_cloudmask_kz/L1C"

    # This will dump the split files straight into your empty train/, val/, and test/ folders
    OUTPUT_DATA = "/home/gustav/Desktop/sentinel2_cloudmask_kz"

    create_leakage_free_splits(RAW_DATA, OUTPUT_DATA)