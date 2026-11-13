import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from miri_model_inputs import PROJECT_DIR


def load_single_image(path, expected_shape):
    """Read a saved COADDALL image without combining it again."""
    array = np.load(path)

    # These files contain one image with a leading frame dimension.
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.shape != expected_shape:
        raise ValueError(
            f"{path}: expected {expected_shape}, got {array.shape}"
        )

    return array.astype(np.float64)


def load_miri_observations(settings_path=None):
    """Load the selected target/reference images and their fitting masks."""
    settings = json.loads(
        Path(settings_path or PROJECT_DIR / "miri_settings.json").read_text()
    )

    data_dir = (
        PROJECT_DIR
        / "data"
        / "HIP65426-MIRI"
        / f"HIP65426-{settings['filter']}"
    )

    config = json.loads(
        (data_dir / "config.json").read_text()
    )
    observations = json.loads(
        (data_dir / "observation_metadata.json").read_text()
    )

    if config["instrument"] != settings["instrument"]:
        raise ValueError("Observation and model instruments differ.")

    if config["filter"] != settings["filter"]:
        raise ValueError("Observation and model filters differ.")

    if not np.isclose(
        config["pixel_scale_arcsec"],
        settings["pixel_scale_arcsec"],
    ):
        raise ValueError("Observation and model pixel scales differ.")

    size = settings["detector_pixels"]
    if size != config["num_det_px"]:
        raise ValueError(
            "Model size differs from the supplied observation crop."
        )

    full_shape = tuple(config["subarray_shape_yx"])

    center_x, center_y = config["crop_center_xy_0based"]
    if center_x != int(center_x) or center_y != int(center_y):
        raise ValueError("Crop center must use integer pixel indices.")

    x0 = int(center_x) - size // 2
    y0 = int(center_y) - size // 2
    x1 = x0 + size
    y1 = y0 + size

    if not (
        0 <= x0 < x1 <= full_shape[1]
        and 0 <= y0 < y1 <= full_shape[0]
    ):
        raise ValueError("Crop extends outside the observation.")

    crop = np.s_[y0:y1, x0:x1]

    pixel_mask = np.load(data_dir / config["pixel_mask"])
    if pixel_mask.shape != (size, size):
        raise ValueError("Unexpected cropped pixel-mask shape.")

    if not np.all(np.isin(pixel_mask, [0, 1])):
        raise ValueError("Expected a binary pixel mask.")

    full_mask = np.load(data_dir / "pixel_masks_full.npy")
    if full_mask.shape != full_shape:
        raise ValueError("Unexpected full pixel-mask shape.")

    if not np.array_equal(pixel_mask, full_mask[crop]):
        raise ValueError("Full and cropped pixel masks disagree.")

    pixel_mask = pixel_mask.astype(bool)

    # Location of the coronagraph center inside the observation crop.
    coron_center = (
        np.asarray(config["coron_center_xy_0based"], dtype=float)
        - np.array([x0, y0])
    )

    # An 80-pixel array has geometric center (39.5, 39.5).
    geometric_center = np.array(
        [(size - 1) / 2, (size - 1) / 2]
    )
    registration_offset = coron_center - geometric_center

    result = {
        "settings": settings,
        "crop_bounds_yx": [y0, y1, x0, x1],
        "coronagraph_center_xy": coron_center,
        "registration_offset_xy_pixels": registration_offset,
        "pixel_mask": pixel_mask,
    }

    for role in ("target", "reference"):
        source_file = observations[role]["source_file"]

        entries = [
            entry
            for entry in config["files"]
            if entry["file"] == source_file
            and entry["role"] == role
        ]
        if len(entries) != 1:
            raise ValueError(
                f"Cannot uniquely identify the selected {role}."
            )

        entry = entries[0]
        star = "HIP-65426" if role == "target" else "HIP-68245"

        filename = (
            f"{role}_{entry['obs']:03d}_"
            f"{entry['exposure']:05d}COADDALL.npy"
        )

        image_path = (
            data_dir / "demo_data" / star / "full_im" / filename
        )
        noise_path = (
            data_dir
            / "demo_data"
            / star
            / "noise_maps"
            / f"noisemap_{filename}"
        )

        image = load_single_image(image_path, full_shape)[crop].copy()
        noise = load_single_image(noise_path, full_shape)[crop].copy()

        valid = (
            pixel_mask
            & np.isfinite(image)
            & np.isfinite(noise)
            & (noise > 0)
        )

        if not np.any(valid):
            raise ValueError(f"No usable pixels for {role}.")

        # Safe placeholder values outside the fitting mask.
        # The loss must still explicitly select valid pixels.
        image = np.where(valid, image, 0.0)
        noise = np.where(valid, noise, 1.0)

        header = fits.getheader(
            data_dir / config["fits_dir"] / source_file,
            extname="SCI",
        )

        result[role] = {
            "image": image,
            "noise": noise,
            "valid_mask": valid,
            "unit": header.get("BUNIT", "unknown"),
            "image_file": str(image_path),
            "noise_file": str(noise_path),
            "source_fits": source_file,
        }

    return result


def main():
    observations = load_miri_observations()

    print("Crop [y0, y1, x0, x1]:",
          observations["crop_bounds_yx"])
    print("Coronagraph center in crop:",
          observations["coronagraph_center_xy"])
    print("Offset from geometric center:",
          observations["registration_offset_xy_pixels"])

    for role in ("target", "reference"):
        observation = observations[role]

        print(f"\n{role}:")
        print("  Image:", observation["image"].shape)
        print("  Noise:", observation["noise"].shape)
        print("  Unit:", observation["unit"])
        print("  Usable pixels:",
              int(observation["valid_mask"].sum()))
        print("  Source:", observation["source_fits"])


if __name__ == "__main__":
    main()
