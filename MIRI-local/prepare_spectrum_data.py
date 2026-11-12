from pathlib import Path
from urllib.request import urlretrieve

from astropy.io import fits

PROJECT_DIR = Path(__file__).resolve().parent
CATALOG_DIR = (
    PROJECT_DIR / "calibration" / "synphot-data" / "grid" / "phoenix"
)

CATALOG_URL = "https://ssb.stsci.edu/cdbs/grid/phoenix/catalog.fits"


# Downloads the PHOENIX catalog and the 9200 K / 9400 K spectrum files
def main():
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    catalog_file = CATALOG_DIR / "catalog.fits"

    if not catalog_file.exists():
        print("Downloading PHOENIX catalog index...")
        temporary_file = CATALOG_DIR / "catalog.fits.part"
        urlretrieve(CATALOG_URL, temporary_file)
        temporary_file.replace(catalog_file)

    # STPSF maps A1V to 9250 K, metallicity 0, and log(g) 4.
    # Inspect nearby catalog entries before downloading spectra.
    with fits.open(catalog_file) as hdul:
        print("Available nearby spectra:")
        for row in hdul[1].data:
            temperature, metallicity, log_g = map(
                float, row["INDEX"].split(",")
            )
            if (
                8750 <= temperature <= 9750
                and metallicity == 0
                and log_g == 4
            ):
                print(temperature, row["FILENAME"])
    # Download the two spectra needed for our A1V choice.
    spectrum_files = [
        "phoenixm00/phoenixm00_9200.fits",
        "phoenixm00/phoenixm00_9400.fits",
    ]

    base_url = "https://ssb.stsci.edu/cdbs/grid/phoenix/"

    for filename in spectrum_files:
        destination = CATALOG_DIR / filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            print("Already downloaded:", filename)
            continue

        print("Downloading:", filename)
        temporary_file = destination.with_suffix(".fits.part")
        urlretrieve(base_url + filename, temporary_file)

        # Check that the downloaded file contains the required spectrum.
        with fits.open(temporary_file) as hdul:
            if "g40" not in [
                name.lower() for name in hdul[1].columns.names
            ]:
                raise ValueError(f"Missing g40 spectrum in {filename}")

        temporary_file.replace(destination)
        print("Saved:", destination)


if __name__ == "__main__":
    main()