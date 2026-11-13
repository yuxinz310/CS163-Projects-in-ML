import argparse
import json
from pathlib import Path

from astropy.io import fits
from miri_paths import FILTER_MASKS


PROJECT_DIR = Path(__file__).resolve().parent


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
    parser.add_argument("--filter", choices=sorted(FILTER_MASKS), default="F1140C")
    parser.add_argument(
        "--fits-dir",
        type=Path,
        help="Folder containing the original FITS observations.",
    )
    args = parser.parse_args()
    data_dir = PROJECT_DIR / "data" / "HIP65426-MIRI" / f"HIP65426-{args.filter}"
    config = json.loads((data_dir / "config.json").read_text())
    fits_dir = args.fits_dir or data_dir / config["fits_mast_dir"]
    # Preserve the demo's choice of the first listed target/reference exposure.
    files = {"target": config["science_files"][0], "reference": config["reference_files"][0]}

    observations = {}

    for role, filename in files.items():
        fits_path = fits_dir / filename

        if not fits_path.is_file():
            raise FileNotFoundError(f"Cannot find observation: {fits_path}")

        metadata = read_observation_metadata(fits_path)

        if metadata["instrument"] != "MIRI":
            raise ValueError(f"Expected MIRI data: {fits_path}")
        if metadata["filter"] != args.filter:
            raise ValueError(f"Expected {args.filter} data: {fits_path}")

        observations[role] = metadata

        print(f"{role}:")
        print(f"  Instrument: {metadata['instrument']}")
        print(f"  Filter: {metadata['filter']}")
        print(f"  Start: {metadata['start_time_utc']} UTC")
        print(f"  End: {metadata['end_time_utc']} UTC")

    output = data_dir / "observation_metadata.json"
    output.write_text(json.dumps(observations, indent=2) + "\n")

    print("Saved:", output)


if __name__ == "__main__":
    main()
