from utility import Sentinel2CloudMaskDataset, compute_global_stats

if __name__ == "__main__":
    # Point ONLY to the new training split
    train_dataset = Sentinel2CloudMaskDataset(
        root_dir="/home/gustav/Desktop/sentinel2_cloudmask_kz/train"
    )

    # We use a batch size of 4 so your laptop doesn't run out of memory
    mean, std = compute_global_stats(train_dataset, batch_size=4)

    print("\n--- Copy these arrays into config.py ---")
    print(f"GLOBAL_MEAN = {mean.tolist()}")
    print(f"GLOBAL_STD = {std.tolist()}")