import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = PROJECT_DIR / "calibration" / "stpsf-data"

# Set the calibration location before importing STPSF.
os.environ["STPSF_PATH"] = str(CALIBRATION_DIR)

import stpsf

# configure the entire model from jason and download stpsf


def main():
    with (PROJECT_DIR / "miri_settings.json").open() as f:
        settings = json.load(f)

    miri = stpsf.MIRI()
    miri.filter = settings["filter"]
    miri.image_mask = settings["image_mask"]
    miri.pupil_mask = settings["pupil_mask"]
    miri.pixelscale = settings["pixel_scale_arcsec"]

    print("STPSF version:", stpsf.__version__)
    print("Calibration folder:", CALIBRATION_DIR)
    print("Instrument:", miri.name)
    print("Filter:", miri.filter)
    print("Coronagraph:", miri.image_mask)
    print("Lyot stop:", miri.pupil_mask)
    print("Pixel scale:", miri.pixelscale)


if __name__ == "__main__":
    main()