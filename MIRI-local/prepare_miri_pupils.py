import json
from pathlib import Path

import numpy as np
from miri_instrument import create_miri_instrument

import stpsf
from stpsf.optics import MIRIFieldDependentAberrationAndObscuration

## Build the three transmission mask


PROJECT_DIR = Path(__file__).resolve().parent


def generate_pupil_maps(miri, settings, wavelength):
    system = miri.get_optical_system(
        fft_oversample=settings["fft_oversample"],
        detector_oversample=settings["detector_oversample"],
        fov_pixels=settings["detector_pixels"],
    )

    wave = system.input_wavefront(wavelength=float(wavelength))

    lyot_matches = [
        plane for plane in system.planes
        if plane.name == settings["pupil_mask"]
    ]
    instrument_matches = [
        plane for plane in system.planes
        if isinstance(
            plane, MIRIFieldDependentAberrationAndObscuration
        )
    ]

    if len(lyot_matches) != 1 or len(instrument_matches) != 1:
        raise ValueError("Could not uniquely identify the pupil components.")

    components = {
        "primary_transmission": system.planes[0],
        "lyot_transmission": lyot_matches[0],
        "instrument_transmission": instrument_matches[0],
    }

    maps = {}
    for name, component in components.items():
        transmission = np.asarray(
            component.get_transmission(wave)
        ).copy()

        expected_shape = (settings["pupil_pixels"],) * 2
        if transmission.shape != expected_shape:
            raise ValueError(
                f"{name}: unexpected shape {transmission.shape}"
            )
        if not np.all(np.isfinite(transmission)):
            raise ValueError(f"{name}: invalid transmission values")

        maps[name] = transmission

    return maps


def main():
    settings = json.loads(
        (PROJECT_DIR / "miri_settings.json").read_text()
    )
    source = settings["source_spectrum"]

    name = (
        f"{settings['filter']}_{source['catalog']}_"
        f"{source['spectral_type']}_n{settings['num_wavelengths']}"
    )
    input_dir = PROJECT_DIR / "model_inputs" / name
    wavelengths = np.load(input_dir / "lambda_weights.npy")[0]

    opd_dir = PROJECT_DIR / "calibration" / "measured_opds"
    selections = json.loads(
        (opd_dir / "opd_selection.json").read_text()
    )

    miri = create_miri_instrument(
        settings,
        opd_dir / selections["reference"]["opd_file"],
    )

    # These pupil transmissions are wavelength-independent in this setup.
    maps = generate_pupil_maps(miri, settings, wavelengths[0])

    output_dir = input_dir / "pupil_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, transmission in maps.items():
        np.save(output_dir / f"{name}.npy", transmission)
        print(
            name,
            "shape:", transmission.shape,
            "range:", transmission.min(), transmission.max(),
        )

    metadata = {
        "settings": settings,
        "stpsf_version": stpsf.__version__,
        "quantity": "Dimensionless electric-field amplitude transmission",
        "sampling_wavelength_m": float(wavelengths[0]),
        "aperture": miri.aperturename,
        "nominal_detector_position": [
            float(value) for value in miri.detector_position
        ],
        "orientation": {
            "primary_transmission": "STPSF entrance-pupil orientation",
            "lyot_transmission": "Native STPSF Lyot-plane orientation",
            "instrument_transmission": "Native STPSF MIRI aberration-plane orientation",
        },
        "extra_export_flips_or_rotations": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()