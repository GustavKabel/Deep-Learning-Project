import os
from pathlib import Path
import xarray as xr
import numpy as np


def sanity_check_netcdf(base_path):
    dataset_dir = Path(base_path)

    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_dir} not found.")
        return

    # Find all NetCDF files recursively
    nc_files = sorted(list(dataset_dir.rglob("*.nc")))
    print(f"Found {len(nc_files)} NetCDF files.\n")

    if not nc_files:
        print("No .nc files found in the specified directory.")
        return

    # Inspect the first file as a baseline sample
    sample_file = nc_files[0]
    print(f"--- Inspecting Sample: {sample_file.name} ---")

    try:
        # Load the dataset
        ds = xr.open_dataset(sample_file, engine="h5netcdf")

        # 1. Print internal data variables (e.g., bands, masks)
        print("Dataset Variables:")
        for var_name, var_data in ds.data_vars.items():
            print(f" - {var_name}: Shape {var_data.shape}, Type {var_data.dtype}")
            # Sanity check: Min and Max values help confirm if masks are binary (0/1) or categorical
            print(f"   Min: {var_data.min().values}, Max: {var_data.max().values}")

        # 2. Print spatial coordinates
        print("\nCoordinates:")
        for coord_name, coord_data in ds.coords.items():
            print(f" - {coord_name}: Shape {coord_data.shape}")

        # 3. Check for global attributes (metadata like CRS, creation date)
        print("\nGlobal Attributes:")
        for attr_name, attr_val in list(ds.attrs.items())[:5]:  # Showing top 5
            print(f" - {attr_name}: {attr_val}")

        ds.close()

    except Exception as e:
        print(f"Error reading {sample_file.name}: {e}")


if __name__ == "__main__":
    # Update <your_username> to match your exact home directory path
    dataset_path = "/home/gustav/Desktop/sentinel2_cloudmask_kz"
    sanity_check_netcdf(dataset_path)