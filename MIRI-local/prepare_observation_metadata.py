import argparse
import json
from pathlib import Path

from astropy.io import fits


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    PROJECT_DIR / "data" / "HIP65426-MIRI" / "HIP65426-F1140C"
)

# Original FITS files corresponding to our selected demo observations.
FILES = {
    "target": "jw01386004001_04101_00001_mirimage_calints.fits",
    "reference": "jw01386006001_04101_00001_mirimage_calints.fits",
}


def read_observation_metadata(filename):
    """Read observation details from the primary FITS header."""
    header = fits.getheader(filename, ext=0)

    return {
        "source_file": filename.name,
        "instrument": header["INSTRUME"],
        "filter": header["FILTER"],
        "coronagraph": header["CORONMSK"],
        "aperture": header["APERNAME"],
        "start_time_utc": header["DATE-BEG"],
        "end_time_utc": header["DATE-END"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Save MIRI observation metadata from FITS headers."
    )
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=DATA_DIR / "fits_mast",
        help="Folder containing the original FITS observations.",
    )
    args = parser.parse_args()

    observations = {}

    for role, filename in FILES.items():
        fits_path = args.fits_dir / filename

        if not fits_path.is_file():
            raise FileNotFoundError(f"Cannot find observation: {fits_path}")

        metadata = read_observation_metadata(fits_path)

        if metadata["instrument"] != "MIRI":
            raise ValueError(f"Expected MIRI data: {fits_path}")
        if metadata["filter"] != "F1140C":
            raise ValueError(f"Expected F1140C data: {fits_path}")

        observations[role] = metadata

        print(f"{role}:")
        print(f"  Instrument: {metadata['instrument']}")
        print(f"  Filter: {metadata['filter']}")
        print(f"  Start: {metadata['start_time_utc']} UTC")
        print(f"  End: {metadata['end_time_utc']} UTC")

    output = DATA_DIR / "observation_metadata.json"
    output.write_text(json.dumps(observations, indent=2) + "\n")

    print("Saved:", output)


if __name__ == "__main__":
    main()