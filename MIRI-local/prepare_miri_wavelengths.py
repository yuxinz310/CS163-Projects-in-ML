import json
from pathlib import Path

import numpy as np

# This import also sets our calibration-data paths.
from miri_spectrum import create_stellar_spectrum

import stpsf


PROJECT_DIR = Path(__file__).resolve().parent


def generate_wavelength_weights(filter_name, spectrum, num_wavelengths):
    """Return wavelengths in meters and normalized weights."""
    if not isinstance(num_wavelengths, int) or num_wavelengths < 1:
        raise ValueError("num_wavelengths must be a positive integer")

    miri = stpsf.MIRI()
    miri.filter = filter_name

    # Internal method also called by STPSF's PSF calculation.
    wavelengths, weights = miri._get_weights(
        source=spectrum,
        nlambda=num_wavelengths,
    )

    wavelengths = np.asarray(wavelengths, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("Invalid wavelength weights")

    return wavelengths, weights


def main():
    with (PROJECT_DIR / "miri_settings.json").open() as f:
        settings = json.load(f)

    source = settings["source_spectrum"]
    spectrum = create_stellar_spectrum(
        spectral_type=source["spectral_type"],
        catalog=source["catalog"],
    )

    wavelengths, weights = generate_wavelength_weights(
        filter_name=settings["filter"],
        spectrum=spectrum,
        num_wavelengths=settings["num_wavelengths"],
    )






    ## file management

    # Separate outputs for different filters, spectra, and sample counts.
    name = (
        f"{settings['filter']}_{source['catalog']}_"
        f"{source['spectral_type']}_n{settings['num_wavelengths']}"
    )
    output_dir = PROJECT_DIR / "model_inputs" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match pyflip's array layout: row 0 = meters, row 1 = weights.
    output_file = output_dir / "lambda_weights.npy"
    np.save(output_file, np.vstack([wavelengths, weights]))

    metadata = {
        "filter": settings["filter"],
        "source_spectrum": source,
        "num_wavelengths": settings["num_wavelengths"],
        "wavelength_unit": "meter",
        "stpsf_version": stpsf.__version__,
        "method": "STPSF/POPPY _get_weights",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("Wavelength (micrometers)    Weight")
    for wavelength, weight in zip(wavelengths, weights):
        print(f"{wavelength * 1e6:12.6f}          {weight:.8f}")

    print("Weight sum:", weights.sum())
    print("Saved:", output_file)


if __name__ == "__main__":
    main()