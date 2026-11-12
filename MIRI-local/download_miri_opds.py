import hashlib
import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    PROJECT_DIR / "data" / "HIP65426-MIRI" / "HIP65426-F1140C"
)
OPD_DIR = PROJECT_DIR / "calibration" / "measured_opds"

os.environ["STPSF_PATH"] = str(
    PROJECT_DIR / "calibration" / "stpsf-data"
)

import stpsf
from stpsf import mast_wss
from astropy.io import fits


def download_nearest_opd(observation_time, output_dir):
    """Find and download the measured OPD nearest this UTC time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = mast_wss.get_opd_at_time(
        observation_time,
        choice="closest",
        verbose=True,
        output_path=str(output_dir),
    )

    path = Path(filename)

    # Check that the downloaded file contains wavefront data.
    with fits.open(path) as hdul:
        if hdul["RESULT_PHASE"].data is None:
            raise ValueError(f"Missing OPD data: {path}")

    return path


def main():
    metadata_file = DATA_DIR / "observation_metadata.json"
    observations = json.loads(metadata_file.read_text())

    selections = {}

    for role in ("target", "reference"):
        observation_time = observations[role]["start_time_utc"]
        print(f"\nFinding OPD for {role}: {observation_time} UTC")

        path = download_nearest_opd(observation_time, OPD_DIR)

        selections[role] = {
            "observation_time_utc": observation_time,
            "time_basis": "observation start",
            "selection_method": "closest",
            "opd_file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stpsf_version": stpsf.__version__,
        }

        print("Downloaded:", path)

    manifest = OPD_DIR / "opd_selection.json"
    manifest.write_text(json.dumps(selections, indent=2) + "\n")
    print("\nSaved selection record:", manifest)


if __name__ == "__main__":
    main()