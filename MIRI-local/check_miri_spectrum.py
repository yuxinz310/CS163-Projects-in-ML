import json
from pathlib import Path

import astropy.units as u
from miri_spectrum import create_stellar_spectrum

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    with (PROJECT_DIR / "miri_settings.json").open() as f:
        settings = json.load(f)

    source = settings["source_spectrum"]

    spectrum = create_stellar_spectrum(
        spectral_type=source["spectral_type"],
        catalog=source["catalog"],
    )

    print("Catalog:", source["catalog"])
    print("Spectral type:", source["spectral_type"])

    # Evaluate the spectrum at a few example wavelengths.
    wavelengths = [11.0, 11.4, 11.8] * u.um
    flux = spectrum(wavelengths)

    for wavelength, value in zip(wavelengths, flux):
        print(f"{wavelength}: {value}")


if __name__ == "__main__":
    main()