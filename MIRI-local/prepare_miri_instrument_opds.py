import argparse
import json
from pathlib import Path
from miri_paths import model_input_dir, load_opd_selections

import numpy as np
from miri_instrument import create_miri_instrument

import stpsf
from stpsf.optics import MIRIFieldDependentAberrationAndObscuration


PROJECT_DIR = Path(__file__).resolve().parent


def generate_instrument_opds(miri, wavelengths, settings):
    """Evaluate MIRI's instrument OPD at each wavelength."""
    optical_system = miri.get_optical_system(
        fft_oversample=settings["fft_oversample"],
        detector_oversample=settings["detector_oversample"],
        fov_pixels=settings["detector_pixels"],
    )

    # Locate the MIRI aberration component by its type.
    # The NIRCam generator's hardcoded plane number need not match MIRI.
    matches = [
        plane
        for plane in optical_system.planes
        if isinstance(
            plane, MIRIFieldDependentAberrationAndObscuration
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one MIRI aberration component, found {len(matches)}"
        )

    instrument_optic = matches[0]
    maps = []

    for wavelength in wavelengths:
        wave = optical_system.input_wavefront(
            wavelength=float(wavelength)
        )
        opd = np.asarray(instrument_optic.get_opd(wave)).copy()

        expected_shape = (settings["pupil_pixels"],) * 2
        if opd.shape != expected_shape:
            raise ValueError(f"Unexpected instrument OPD shape: {opd.shape}")

        if not np.all(np.isfinite(opd)):
            raise ValueError("Instrument OPD contains invalid values.")

        maps.append(opd)

    return np.stack(maps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=PROJECT_DIR / "miri_settings.json")
    args = parser.parse_args()
    settings = json.loads(
        args.settings.read_text()
    )
    source = settings["source_spectrum"]

    name = (
        f"{settings['filter']}_{source['catalog']}_"
        f"{source['spectral_type']}_n{settings['num_wavelengths']}"
    )
    input_dir = model_input_dir(settings)

    wavelength_weights = np.load(input_dir / "lambda_weights.npy")
    if wavelength_weights.shape != (2, settings["num_wavelengths"]):
        raise ValueError("Saved wavelengths do not match the settings.")

    wavelengths = wavelength_weights[0]

    opd_dir = PROJECT_DIR / "calibration" / "measured_opds"
    selections = load_opd_selections(settings)

    # Reuse our instrument setup.
    # MIRI's SI map comes from instrument calibration, not this WSS file.
    miri = create_miri_instrument(
        settings,
        opd_dir / selections["reference"]["opd_file"],
    )

    maps = generate_instrument_opds(miri, wavelengths, settings)

    output_dir = input_dir / "instrument_opds"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, (wavelength, opd) in enumerate(zip(wavelengths, maps)):
        filename = f"instrument_opd_{index}.npy"
        np.save(output_dir / filename, opd)

        entries.append({
            "file": filename,
            "wavelength_m": float(wavelength),
        })
        print(f"Saved {filename}: {wavelength * 1e6:.6f} micrometers")

    metadata = {
        "settings": settings,
        "stpsf_version": stpsf.__version__,
        "unit": "meter",
        "shape_per_map": list(maps.shape[1:]),
        "component": "MIRI instrument OPD; excludes telescope OPD",
        "orientation": "Native STPSF MIRI aberration-plane orientation",
        "aperture": miri.aperturename,
        "position_source": "STPSF nominal coronagraph position",
        "detector_position": [
            float(value) for value in miri.detector_position
        ],
        "maps": entries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("Maximum difference from first map:",
          np.max(np.abs(maps - maps[0])), "meters")
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()