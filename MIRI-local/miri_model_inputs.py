import argparse
import json
from pathlib import Path

import numpy as np
from miri_paths import model_input_dir, telescope_input_dir


PROJECT_DIR = Path(__file__).resolve().parent


def load_pupil_array(path, expected_shape):
    """Load a pupil-sized map and check its dimensions and values."""
    array = np.load(path)

    if array.shape != expected_shape:
        raise ValueError(f"{path}: unexpected shape {array.shape}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: contains invalid values")

    # FITS-derived arrays may retain big-endian storage, which PyTorch
    # cannot ingest directly. Convert in memory to native float64.
    return np.asarray(array, dtype=np.float64)


def load_miri_inputs(role="reference", settings_path=None):
    """Load saved optical inputs without changing their orientations."""
    if role not in ("reference", "target"):
        raise ValueError("Role must be reference or target.")

    settings = json.loads(
        Path(settings_path or PROJECT_DIR / "miri_settings.json").read_text()
    )
    source = settings["source_spectrum"]

    name = (
        f"{settings['filter']}_{source['catalog']}_"
        f"{source['spectral_type']}_n{settings['num_wavelengths']}"
    )
    input_dir = model_input_dir(settings)
    telescope_dir = telescope_input_dir(settings, role)
    pupil_dir = input_dir / "pupil_masks"
    instrument_dir = input_dir / "instrument_opds"
    focal_dir = input_dir / f"focal_mask_fft{settings['fft_oversample']}"

    # Prevent mixing inputs generated with different settings.
    metadata = {}
    for label, folder in {
        "telescope": telescope_dir,
        "pupils": pupil_dir,
        "instrument": instrument_dir,
        "focal_mask": focal_dir,
    }.items():
        record = json.loads((folder / "metadata.json").read_text())
        saved_settings = dict(record["settings"])
        requested_settings = dict(settings)
        if label == "telescope":
            # The telescope pupil map is independent of spectral sampling
            # and of FFT/detector padding. Pupil size and optical settings
            # still have to match exactly.
            for key in ("num_wavelengths", "fft_oversample", "detector_oversample"):
                saved_settings.pop(key, None)
                requested_settings.pop(key, None)
        if saved_settings != requested_settings:
            raise ValueError(
                f"{label}: saved settings differ from the selected settings. "
                "Regenerate the affected inputs before continuing."
            )
        metadata[label] = record

    wavelength_weights = np.load(input_dir / "lambda_weights.npy")
    if wavelength_weights.shape != (2, settings["num_wavelengths"]):
        raise ValueError("Unexpected wavelength/weight array shape.")

    wavelengths, weights = wavelength_weights
    if not (
        np.all(np.isfinite(wavelength_weights))
        and np.all(wavelengths > 0)
        and np.all(np.diff(wavelengths) > 0)
        and np.all(weights >= 0)
        and np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Invalid wavelengths or weights.")

    # Check that saved map indices correspond to these wavelengths.
    for label in ("instrument", "focal_mask"):
        entries = metadata[label]["maps"]
        recorded = np.array([entry["wavelength_m"] for entry in entries])
        if recorded.shape != wavelengths.shape or not np.allclose(
            recorded, wavelengths, rtol=1e-12, atol=0
        ):
            raise ValueError(f"{label}: wavelength ordering does not match.")

    pupil_shape = (settings["pupil_pixels"],) * 2

    inputs = {
        "settings": settings,
        "role": role,
        "metadata": metadata,
        "wavelengths": wavelengths,
        "weights": weights,
        "telescope_opd": load_pupil_array(
            telescope_dir / "telescope_opd.npy", pupil_shape
        ),
    }

    for name in (
        "primary_transmission",
        "lyot_transmission",
        "instrument_transmission",
    ):
        inputs[name] = load_pupil_array(
            pupil_dir / f"{name}.npy", pupil_shape
        )

    inputs["instrument_opds"] = np.stack([
        load_pupil_array(instrument_dir / entry["file"], pupil_shape)
        for entry in metadata["instrument"]["maps"]
    ])

    # Keep file paths so large focal masks can be loaded one at a time.
    inputs["focal_mask_files"] = [
        focal_dir / entry["file"]
        for entry in metadata["focal_mask"]["maps"]
    ]
    for path in inputs["focal_mask_files"]:
        if not path.is_file():
            raise FileNotFoundError(path)

    return inputs


def load_focal_mask(inputs, index):
    """Load the transmission and phase for one wavelength."""
    with np.load(inputs["focal_mask_files"][index]) as saved:
        transmission = saved["transmission"]
        phase = saved["phase_radians"]

    settings = inputs["settings"]
    size = settings["pupil_pixels"] * settings["fft_oversample"]

    for name, array in (("transmission", transmission), ("phase", phase)):
        if array.shape != (size, size):
            raise ValueError(f"Unexpected focal-mask {name} shape.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid focal-mask {name} values.")

    return transmission, phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=["reference", "target"],
        default="reference",
    )
    args = parser.parse_args()

    inputs = load_miri_inputs(args.role)

    print("Role:", inputs["role"])
    print("Wavelength count:", len(inputs["wavelengths"]))
    print("Weight sum:", inputs["weights"].sum())
    print("Telescope OPD:", inputs["telescope_opd"].shape)
    print("Instrument OPDs:", inputs["instrument_opds"].shape)

    for name in (
        "primary_transmission",
        "lyot_transmission",
        "instrument_transmission",
    ):
        print(name, inputs[name].shape)

    transmission, phase = load_focal_mask(inputs, 0)
    print("First focal-mask transmission:", transmission.shape)
    print("First focal-mask phase:", phase.shape)
    print("Inputs loaded successfully; no propagation performed.")


if __name__ == "__main__":
    main()
