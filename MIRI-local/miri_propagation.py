"""Differentiable MIRI optical propagation using exported STPSF maps.

Includes optical propagation, wavelength weighting, and detector binning.
Detector response and observation registration are separate operations.

Output normalization: unit incident pupil power, not MJy/sr.
"""

from miri_paths import load_opd_selections

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import spline_filter1d
from scipy.special import cosdg, sindg
import torch
from torch import nn
from torch.nn import functional as F

from miri_model_inputs import (
    PROJECT_DIR,
    load_miri_inputs,
    load_focal_mask,
)
from miri_wavefront import create_entrance_field


ARCSEC_TO_RAD = np.pi / (180 * 3600)


def apply_coordinate_inversion(field):
    """Apply STPSF's telescope exit-pupil y inversion."""
    return torch.flip(field, dims=(-2,))


def apply_fqpm_alignment(
    field,
    pupil_pixels,
    fft_oversample,
    direction="forward",
):
    """Apply the half-focal-pixel alignment phase ramp.

    pupil_pixels is the original, unpadded pupil size.
    This function also works on the padded pupil after the inverse FFT.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("Direction must be forward or backward.")

    size = field.shape[-1]
    coordinates = torch.arange(
        size,
        dtype=field.real.dtype,
        device=field.device,
    )
    coordinates = coordinates - (size - 1) / 2

    phase = torch.pi * (
        coordinates[:, None] + coordinates[None, :]
    )
    phase = phase / (pupil_pixels * fft_oversample)

    if direction == "backward":
        phase = -phase

    return field * torch.exp(1j * phase)


def pupil_to_focal(field, fft_oversample):
    """Propagate toward the focal plane using POPPY's FFT convention."""
    size = field.shape[-1]
    padding = size * (fft_oversample - 1) // 2

    field = F.pad(field, (padding,) * 4)

    # POPPY and pyflip use the positive-exponent Fourier convention:
    # optical forward propagation corresponds to torch's inverse FFT.
    return torch.fft.fftshift(
        torch.fft.ifft2(field, norm="ortho"),
        dim=(-2, -1),
    )


def focal_to_pupil(field):
    """Propagate back to the padded pupil."""
    return torch.fft.fft2(
        torch.fft.ifftshift(field, dim=(-2, -1)),
        norm="ortho",
    )


class FixedPupilRotation(nn.Module):
    """Match POPPY's cubic B-spline rotation.

    SciPy constructs fixed interpolation coefficients.
    All operations on the light field use PyTorch, preserving gradients.
    """

    def __init__(self, size, poppy_angle_deg):
        super().__init__()

        # POPPY calls scipy.ndimage.rotate with the negative angle.
        angle = -float(poppy_angle_deg)
        cosine, sine = cosdg(angle), sindg(angle)

        prefilter = spline_filter1d(
            np.eye(size),
            order=3,
            axis=0,
            mode="mirror",
        )
        self.register_buffer(
            "prefilter", torch.from_numpy(prefilter)
        )

        y, x = np.indices((size, size), dtype=float)
        center = (size - 1) / 2

        input_y = (
            cosine * (y - center)
            + sine * (x - center)
            + center
        )
        input_x = (
            -sine * (y - center)
            + cosine * (x - center)
            + center
        )

        valid = (
            (input_y >= 0)
            & (input_y <= size - 1)
            & (input_x >= 0)
            & (input_x <= size - 1)
        )
        self.register_buffer("valid", torch.from_numpy(valid))

        for axis, coordinates in (
            ("y", input_y),
            ("x", input_x),
        ):
            start = np.floor(coordinates).astype(np.int64) - 1
            indices = []
            weights = []

            for offset in range(4):
                index = start + offset
                distance = np.abs(coordinates - index)

                weight = np.where(
                    distance < 1,
                    (4 - 6 * distance**2 + 3 * distance**3) / 6,
                    np.where(
                        distance < 2,
                        (2 - distance)**3 / 6,
                        0,
                    ),
                )

                # Mirror extension of spline coefficients.
                index %= 2 * (size - 1)
                index = np.where(
                    index >= size,
                    2 * (size - 1) - index,
                    index,
                )

                indices.append(index)
                weights.append(weight)

            self.register_buffer(
                axis + "_indices",
                torch.from_numpy(np.stack(indices)),
            )
            self.register_buffer(
                axis + "_weights",
                torch.from_numpy(np.stack(weights)),
            )

    def forward(self, field):
        parts = torch.stack((field.real, field.imag))

        coefficients = (
            self.prefilter @ parts @ self.prefilter.T
        )

        output = torch.zeros_like(parts)

        for iy in range(4):
            for ix in range(4):
                output = output + (
                    coefficients[
                        :,
                        self.y_indices[iy],
                        self.x_indices[ix],
                    ]
                    * self.y_weights[iy]
                    * self.x_weights[ix]
                )

        return (
            torch.complex(output[0], output[1]) * self.valid
        )


def read_stpsf_geometry(inputs):
    """Read diameter and rotation from our configured STPSF system."""
    from miri_instrument import create_miri_instrument

    opd_dir = PROJECT_DIR / "calibration" / "measured_opds"
    settings = inputs["settings"]
    selections = load_opd_selections(settings)

    miri = create_miri_instrument(
        settings,
        opd_dir / selections[inputs["role"]]["opd_file"],
    )

    system = miri.get_optical_system(
        fft_oversample=settings["fft_oversample"],
        detector_oversample=settings["detector_oversample"],
        fov_pixels=settings["detector_pixels"],
    )

    expected = [
        "OTE_Linear_Model_WSS",
        "CoordinateInversion",
        "FQPM_FFT_aligner",
        "CompoundAnalyticOptic",
        "FQPM_FFT_aligner",
        "FITSOpticalElement",
        "Rotation",
        "MIRIFieldDependentAberrationAndObscuration",
        "Detector",
    ]

    actual = [
        type(plane).__name__ for plane in system.planes
    ]
    if actual != expected:
        raise ValueError(
            "STPSF optical sequence differs from this implementation."
        )

    if system.planes[1].axis != "y":
        raise ValueError("Expected telescope y inversion.")

    wave = system.input_wavefront(
        wavelength=float(inputs["wavelengths"][0])
    )

    return {
        "pupil_diameter_m": float(wave.diam.to_value("m")),
        "poppy_rotation_deg": float(system.planes[6].angle),
    }


class MIRIPropagator(nn.Module):
    """Generate a monochromatic or broadband optical PSF.

    Optional telescope and instrument OPD corrections are in meters.
    Each correction is shared across wavelengths.

    source_xy_arcsec specifies tilt BEFORE telescope coordinate inversion.
    It is not a position measured in the observed detector image.
    """

    def __init__(self, inputs, geometry):
        super().__init__()

        self.inputs = inputs
        self._fixed_cache = {}
        self.settings = inputs["settings"]

        self.n = self.settings["pupil_pixels"]
        self.fft_os = self.settings["fft_oversample"]
        self.det_os = self.settings["detector_oversample"]
        self.det_n = self.settings["detector_pixels"]
        self.diameter = geometry["pupil_diameter_m"]

        if self.n % 2 or self.n < 2:
            raise ValueError(
                "This implementation requires an even pupil size."
            )

        for value in (
            self.fft_os,
            self.det_os,
            self.det_n,
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    "Sampling settings must be positive integers."
                )

        for name in (
            "primary_transmission",
            "telescope_opd",
            "lyot_transmission",
            "instrument_transmission",
            "instrument_opds",
            "weights",
        ):
            self.register_buffer(
                name,
                torch.as_tensor(
                    inputs[name],
                    dtype=torch.float64,
                ),
            )

        self.rotation = FixedPupilRotation(
            self.n,
            geometry["poppy_rotation_deg"],
        )

        coordinates = (
            torch.arange(self.n, dtype=torch.float64)
            - (self.n - 1) / 2
        )
        self.register_buffer(
            "pupil_coordinate_m",
            coordinates * self.diameter / self.n,
        )

    def _apply(self, fn, recurse=True):
        # Cached tensors are derived, nontrainable, and excluded from checkpoints.
        # Rebuild after a device/dtype change instead of retaining old allocations.
        self._fixed_cache.clear()
        return super()._apply(fn, recurse=recurse)

    @torch.no_grad()
    def _fixed_optics(self, index, opd):
        key = (opd.device, opd.dtype, index)
        if key in self._fixed_cache:
            return self._fixed_cache[key]
        common_key = (opd.device, opd.dtype, "alignment")
        if common_key not in self._fixed_cache:
            phasors = []
            for size, direction in ((self.n, "forward"),
                                    (self.n * self.fft_os, "backward")):
                coordinates = torch.arange(size, dtype=opd.dtype, device=opd.device)
                coordinates = coordinates - (size - 1) / 2
                phase = torch.pi * (coordinates[:, None] + coordinates[None, :])
                phase = phase / (self.n * self.fft_os)
                if direction == "backward":
                    phase = -phase
                phasors.append(torch.exp(1j * phase))
            self._fixed_cache[common_key] = phasors
        alignment, unalignment = self._fixed_cache[common_key]
        transmission, phase = load_focal_mask(self.inputs, index)
        transmission = torch.as_tensor(transmission, dtype=opd.dtype, device=opd.device)
        phase = torch.as_tensor(phase, dtype=opd.dtype, device=opd.device)
        wavelength = float(self.inputs["wavelengths"][index])
        output_size = self.det_n * self.det_os

        fov_lambda_over_d = (
            self.det_n
            * self.settings["pixel_scale_arcsec"]
            * ARCSEC_TO_RAD
            * self.diameter
            / wavelength
        )

        pupil_coordinates = (
            self.pupil_coordinate_m / self.diameter
        )

        detector_coordinates = torch.arange(
            output_size,
            dtype=opd.dtype,
            device=opd.device,
        )
        detector_coordinates = (
            detector_coordinates - (output_size - 1) / 2
        ) * fov_lambda_over_d / output_size

        matrix = torch.exp(
            2j
            * torch.pi
            * torch.outer(
                pupil_coordinates,
                detector_coordinates,
            )
        )

        normalization = (
            fov_lambda_over_d / (self.n * output_size)
        )
        fixed = dict(alignment=alignment, unalignment=unalignment,
                     transmission=transmission, phase_phasor=torch.exp(1j * phase),
                     matrix=matrix, normalization=normalization)
        self._fixed_cache[key] = fixed
        return fixed

    def monochromatic(
        self,
        index,
        telescope_delta_m=None,
        instrument_delta_m=None,
        source_xy_arcsec=(0.0, 0.0),
    ):
        """Return intensity on the oversampled detector grid."""
        wavelength = float(
            self.inputs["wavelengths"][index]
        )

        # 0. Entrance pupil: transmission and telescope OPD.
        opd = self.telescope_opd
        if telescope_delta_m is not None:
            opd = opd + telescope_delta_m

        field = create_entrance_field(
            self.primary_transmission,
            opd,
            wavelength,
        )

        # Match POPPY normalize="first": unit incident pupil power.
        field = field / (
            self.primary_transmission.square().sum().sqrt()
        )

        # Optional source tilt in entrance-pupil coordinates.
        offset = torch.as_tensor(
            source_xy_arcsec,
            dtype=opd.dtype,
            device=opd.device,
        )
        q = self.pupil_coordinate_m

        tilt_opd = -(
            q[None, :] * offset[0]
            + q[:, None] * offset[1]
        ) * ARCSEC_TO_RAD

        field = field * torch.exp(
            2j * torch.pi * tilt_opd / wavelength
        )

        # 1. Telescope coordinate inversion.
        field = apply_coordinate_inversion(field)

        # Fixed optical tensors are constructed once on the active device.
        fixed = self._fixed_optics(index, opd)
        field = field * fixed["alignment"]
        field = pupil_to_focal(field, self.fft_os)
        # Keep the original multiplication order for numerical equivalence.
        field = field * fixed["transmission"] * fixed["phase_phasor"]
        field = focal_to_pupil(field)
        field = field * fixed["unalignment"]

        # 5. Lyot stop.
        # Its transmission is zero outside the saved pupil grid.
        # POPPY removes padding before rotating the resulting pupil.
        padding = self.n * (self.fft_os - 1) // 2
        field = field[
            padding:padding + self.n,
            padding:padding + self.n,
        ]
        field = field * self.lyot_transmission

        # 6. Rotate into the instrument coordinate frame.
        field = self.rotation(field)

        # 7. MIRI internal transmission and instrument OPD.
        instrument_opd = self.instrument_opds[index]
        if instrument_delta_m is not None:
            instrument_opd = (
                instrument_opd + instrument_delta_m
            )

        field = field * self.instrument_transmission
        field = field * torch.exp(
            2j * torch.pi * instrument_opd / wavelength
        )

        # 8. Fixed matrix Fourier transform to the detector grid.
        matrix = fixed["matrix"]
        field = (matrix.T @ field @ matrix) * fixed["normalization"]

        return field.abs().square()

    def forward(
        self,
        telescope_delta_m=None,
        instrument_delta_m=None,
        source_xy_arcsec=(0.0, 0.0),
    ):
        """Combine wavelength intensities using the saved weights."""
        image = None

        for index in range(
            len(self.inputs["wavelengths"])
        ):
            contribution = (
                self.weights[index]
                * self.monochromatic(
                    index,
                    telescope_delta_m,
                    instrument_delta_m,
                    source_xy_arcsec,
                )
            )

            image = (
                contribution
                if image is None
                else image + contribution
            )

        return image

    def bin_detector(self, oversampled_intensity):
        """Sum subpixels into detector pixels."""
        return oversampled_intensity.reshape(
            self.det_n,
            self.det_os,
            self.det_n,
            self.det_os,
        ).sum(dim=(1, 3))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--role",
        choices=["reference", "target"],
        default="reference",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--wavelength-index",
        type=int,
        default=None,
        help="Omit to use all saved wavelengths and weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "No GPU is available. Run on an allocated GPU node, "
            "or select --device cpu."
        )

    inputs = load_miri_inputs(args.role)

    if args.wavelength_index is not None:
        if not (
            0 <= args.wavelength_index
            < len(inputs["wavelengths"])
        ):
            raise ValueError(
                "Wavelength index is outside the saved list."
            )

    geometry = read_stpsf_geometry(inputs)
    model = MIRIPropagator(
        inputs, geometry
    ).to(args.device)

    # Rendering only; later fitting will enable gradients.
    with torch.inference_mode():
        if args.wavelength_index is None:
            fine_image = model()
        else:
            fine_image = model.monochromatic(
                args.wavelength_index
            )

        detector_image = model.bin_detector(fine_image)

    if not torch.isfinite(detector_image).all():
        raise ValueError(
            "Propagation produced invalid values."
        )

    tag = (
        "broadband"
        if args.wavelength_index is None
        else f"wavelength_{args.wavelength_index}"
    )

    output_dir = args.output_dir or (
        PROJECT_DIR
        / "outputs"
        / "optical_psf"
        / args.role
        / tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "oversampled_intensity.npy",
        fine_image.cpu().numpy(),
    )
    np.save(
        output_dir / "detector_intensity.npy",
        detector_image.cpu().numpy(),
    )

    metadata = {
        "settings": inputs["settings"],
        "role": args.role,
        "geometry": geometry,
        "wavelength_index": args.wavelength_index,
        "normalization": (
            "Unit incident pupil power; "
            "dimensionless flux fractions"
        ),
        "detector_response_applied": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(
        "Oversampled image:",
        tuple(fine_image.shape),
    )
    print(
        "Detector image:",
        tuple(detector_image.shape),
    )
    print(
        "Flux within detector field:",
        detector_image.sum().item(),
    )
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()