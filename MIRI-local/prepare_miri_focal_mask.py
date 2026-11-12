import json
from pathlib import Path

import astropy.units as u
import numpy as np

# Sets the STPSF calibration path before importing STPSF/POPPY.
from miri_instrument import create_miri_instrument

import poppy
import stpsf
from poppy.poppy_core import PlaneType


PROJECT_DIR = Path(__file__).resolve().parent


def sample_focal_mask(system, focal_mask, wavelength, settings):
    """Sample mask transmission and phase on the FQPM focal-plane grid."""
    wavelength = float(wavelength)
    if wavelength <= 0 or not np.isfinite(wavelength):
        raise ValueError("Wavelength must be finite and positive.")

    pupil_pixels = settings["pupil_pixels"]
    fft_oversample = settings["fft_oversample"]
    size = pupil_pixels * fft_oversample

    # Obtain the physical pupil diameter from STPSF.
    pupil_wave = system.input_wavefront(wavelength=wavelength)

    if pupil_wave.shape != (pupil_pixels, pupil_pixels):
        raise ValueError("STPSF pupil sampling does not match the settings.")

    # Intermediate focal-plane spacing: wavelength / diameter / oversampling.
    pixel_scale_arcsec = (
        (wavelength * u.m / pupil_wave.diam)
        .to_value(u.dimensionless_unscaled)
        / fft_oversample
        * u.rad
    ).to_value(u.arcsec)

    # This grid is already sampled at the requested FFT resolution.
    # Its center is between the four central pixels, matching the
    # coordinate convention used with STPSF's FQPM FFT aligners.
    image_wave = poppy.Wavefront(
        wavelength=wavelength,
        npix=size,
        pixelscale=pixel_scale_arcsec,
        oversample=1,
    )

    transmission = np.asarray(
        focal_mask.get_transmission(image_wave)
    ).copy()

    # This is the focal-mask delay, not telescope or instrument aberration.
    mask_opd = np.asarray(focal_mask.get_opd(image_wave))
    phase = 2 * np.pi * mask_opd / wavelength

    if transmission.shape != (size, size):
        raise ValueError(
            f"Unexpected transmission shape: {transmission.shape}"
        )
    if phase.shape != (size, size):
        raise ValueError(f"Unexpected phase shape: {phase.shape}")

    if not (
        np.all(np.isfinite(transmission))
        and np.all(np.isfinite(phase))
    ):
        raise ValueError("The focal mask contains invalid values.")

    return transmission, phase, float(pixel_scale_arcsec)


def main():
    settings = json.loads(
        (PROJECT_DIR / "miri_settings.json").read_text()
    )

    if settings["image_mask"] != "FQPM1140":
        raise ValueError("This first exporter supports FQPM1140.")

    fft_oversample = settings["fft_oversample"]
    if not isinstance(fft_oversample, int) or fft_oversample < 1:
        raise ValueError("fft_oversample must be a positive integer.")

    source = settings["source_spectrum"]
    name = (
        f"{settings['filter']}_{source['catalog']}_"
        f"{source['spectral_type']}_n{settings['num_wavelengths']}"
    )
    input_dir = PROJECT_DIR / "model_inputs" / name

    wavelength_weights = np.load(input_dir / "lambda_weights.npy")
    if wavelength_weights.shape != (2, settings["num_wavelengths"]):
        raise ValueError("Saved wavelengths do not match the settings.")

    wavelengths = wavelength_weights[0]

    # Reuse our configured STPSF instrument.
    opd_dir = PROJECT_DIR / "calibration" / "measured_opds"
    selections = json.loads(
        (opd_dir / "opd_selection.json").read_text()
    )

    miri = create_miri_instrument(
        settings,
        opd_dir / selections["reference"]["opd_file"],
    )

    system = miri.get_optical_system(
        fft_oversample=fft_oversample,
        detector_oversample=settings["detector_oversample"],
        fov_pixels=settings["detector_pixels"],
    )

    # Select STPSF's focal-plane component, including its field stop.
    image_planes = [
        plane for plane in system.planes
        if plane.planetype == PlaneType.image
    ]
    if len(image_planes) != 1:
        raise ValueError(
            f"Expected one image-plane component, found {len(image_planes)}"
        )

    focal_mask = image_planes[0]

    output_dir = input_dir / f"focal_mask_fft{fft_oversample}"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    for index, wavelength in enumerate(wavelengths):
        transmission, phase, pixel_scale = sample_focal_mask(
            system, focal_mask, wavelength, settings
        )

        filename = f"focal_mask_{index}.npz"

        # Lossless compression of the two arrays.
        np.savez_compressed(
            output_dir / filename,
            transmission=transmission,
            phase_radians=phase,
        )

        entries.append({
            "file": filename,
            "wavelength_m": float(wavelength),
            "pixel_scale_arcsec": pixel_scale,
        })

        print(
            f"Saved {filename}: "
            f"{wavelength * 1e6:.6f} micrometers, "
            f"shape={transmission.shape}, "
            f"pixel scale={pixel_scale:.8f} arcsec"
        )

        # Keep only one wavelength's output arrays in memory at a time.
        del transmission, phase

    size = settings["pupil_pixels"] * fft_oversample

    metadata = {
        "settings": settings,
        "stpsf_version": stpsf.__version__,
        "poppy_version": poppy.__version__,
        "component": focal_mask.name,
        "design_wavelength_m": 11.4e-6,
        "shape": [size, size],
        "transmission_quantity": "Dimensionless electric-field amplitude",
        "phase_unit": "radian",
        "centering": "Array center, between four central pixels",
        "orientation": "Native STPSF focal-mask coordinates",
        "extra_export_flips_or_rotations": False,
        "propagation_requires_fqpm_alignment": True,
        "maps": entries,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()