import argparse
import json
import os
from pathlib import Path

import numpy as np
from miri_paths import load_opd_selections, telescope_input_dir


PROJECT_DIR = Path(__file__).resolve().parent
OPD_DIR = PROJECT_DIR / "calibration" / "measured_opds"

# Locate the STPSF calibration files before importing STPSF.
os.environ["STPSF_PATH"] = str(
    PROJECT_DIR / "calibration" / "stpsf-data"
)

import stpsf


def create_miri_instrument(settings, opd_file):
    """Configure STPSF MIRI and load a measured wavefront."""
    if settings["instrument"] != "MIRI":
        raise ValueError("This function expects MIRI settings.")

    # This implementation uses STPSF's default 1024-pixel pupil.
    if settings["pupil_pixels"] != 1024:
        raise ValueError("Other pupil sizes are not implemented yet.")

    opd_file = Path(opd_file)
    if not opd_file.is_file():
        raise FileNotFoundError(opd_file)

    miri = stpsf.MIRI()
    miri.filter = settings["filter"]
    miri.image_mask = settings["image_mask"]
    miri.pupil_mask = settings["pupil_mask"]
    miri.pixelscale = settings["pixel_scale_arcsec"]

    # Load the measurement and subtract the modeled sensing contributions.
    # This does not overwrite the downloaded FITS file.
    miri.load_wss_opd(
        str(opd_file),
        backout_si_wfe=True,
        verbose=True,
        plot=False,
    )

    return miri


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and save the telescope OPD for MIRI."
    )
    parser.add_argument("--settings", type=Path, default=PROJECT_DIR / "miri_settings.json")
    parser.add_argument(
        "--role",
        choices=["target", "reference"],
        default="reference",
    )
    args = parser.parse_args()

    settings = json.loads(
        args.settings.read_text()
    )
    selections = load_opd_selections(settings)

    selection = selections[args.role]
    opd_file = OPD_DIR / selection["opd_file"]

    miri = create_miri_instrument(settings, opd_file)

    # Includes telescope field dependence at the selected MIRI position.
    # Does not include MIRI's own instrument OPD.
    telescope_opd = np.asarray(miri.get_wfe(kind="ote"))

    expected_shape = (settings["pupil_pixels"],) * 2
    if telescope_opd.shape != expected_shape:
        raise ValueError(
            f"Expected OPD shape {expected_shape}, "
            f"got {telescope_opd.shape}"
        )

    if not np.all(np.isfinite(telescope_opd)):
        raise ValueError("The telescope OPD contains invalid values.")

    # Save the processed telescope map.
    output_dir = telescope_input_dir(settings, args.role)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "telescope_opd.npy"
    np.save(output_file, telescope_opd)

    # Record how the map was prepared.
    metadata = {
        "role": args.role,
        "source_selection": selection,
        "settings": settings,
        "stpsf_version": stpsf.__version__,
        "unit": "meter",
        "shape": list(telescope_opd.shape),
        "orientation": "STPSF OTE entrance pupil orientation",
        "includes_telescope_field_dependence": True,
        "includes_miri_instrument_opd": False,
        "backout_si_wfe": True,
        "aperture": miri.aperturename,
        "position_source": "STPSF nominal coronagraph position",
        "detector_position": [
            float(value) for value in miri.detector_position
        ],
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\nObservation role:", args.role)
    print("Measured OPD:", opd_file.name)
    print("MIRI aperture:", miri.aperturename)
    print("Nominal detector position:", miri.detector_position)
    print("Telescope OPD shape:", telescope_opd.shape)
    print("OPD unit: meters")
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
