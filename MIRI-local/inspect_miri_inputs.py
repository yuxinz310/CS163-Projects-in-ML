import json
from pathlib import Path

# Locate the data relative to this script, not your terminal's directory.
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "HIP65426-F1140C"

with (data_dir / "config.json").open() as f:
    config = json.load(f)

print("Instrument:", config["instrument"])
print("Filter:", config["filter"])
print("Coronagraph:", config["image_mask"])
print("Pixel scale:", config["pixel_scale_arcsec"])
print("Full image shape:", config["subarray_shape_yx"])
print("Crop size:", config["num_det_px"])
print("Crop center:", config["crop_center_xy_0based"])

files = {
    "Target": "demo_data/HIP-65426/full_im/target_004_00001COADDALL.npy",
    "Reference": "demo_data/HIP-68245/full_im/reference_006_00001COADDALL.npy",
    "Target noise": "demo_data/HIP-65426/noise_maps/noisemap_target_004_00001COADDALL.npy",
    "Pixel mask": config["pixel_mask"],
}

for name, filename in files.items():
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    print(f"{name}: found")
    