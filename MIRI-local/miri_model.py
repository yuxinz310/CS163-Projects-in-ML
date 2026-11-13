"""Assemble the MIRI optical and detector models for optimization."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from miri_model_inputs import load_miri_inputs
from miri_propagation import (
    MIRIPropagator,
    read_stpsf_geometry,
)
from miri_detector import GaussianDetector
from miri_observations import load_miri_observations


def shift_image(image, offset_xy):
    """Shift an image using bicubic interpolation.

    Positive x moves right.
    Positive y moves toward larger row indices.
    """
    height, width = image.shape

    offset = torch.as_tensor(
        offset_xy,
        dtype=image.dtype,
        device=image.device,
    )

    one = image.new_tensor(1.0)
    zero = image.new_tensor(0.0)

    transform = torch.stack((
        torch.stack((
            one,
            zero,
            -2 * offset[0] / width,
        )),
        torch.stack((
            zero,
            one,
            -2 * offset[1] / height,
        )),
    ))[None]

    grid = F.affine_grid(
        transform,
        (1, 1, height, width),
        align_corners=False,
    )

    return F.grid_sample(
        image[None, None],
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )[0, 0]


class MIRIModel(nn.Module):
    def __init__(
        self,
        inputs,
        geometry,
        registration_offset_xy_pixels,
        sigma_detector_pixels=0.28,
        train_flat_field=False,
        checkpoint_wavelengths=True,
    ):
        super().__init__()

        self.role = inputs["role"]
        self.settings = inputs["settings"]
        self.checkpoint_wavelengths = checkpoint_wavelengths

        self.propagator = MIRIPropagator(inputs, geometry)

        self.detector = GaussianDetector(
            detector_pixels=self.settings["detector_pixels"],
            detector_oversample=self.settings["detector_oversample"],
            sigma_detector_pixels=sigma_detector_pixels,
            trainable_sigma=False,
        )

        pupil_size = self.settings["pupil_pixels"]

        # Additive OPD corrections, in meters.
        # Each map is shared across all wavelengths.
        self.telescope_delta_m = nn.Parameter(
            torch.zeros(
                (pupil_size, pupil_size),
                dtype=torch.float64,
            )
        )
        self.instrument_delta_m = nn.Parameter(
            torch.zeros(
                (pupil_size, pupil_size),
                dtype=torch.float64,
            )
        )

        # Physical source pointing in entrance-pupil coordinates,
        # before the telescope inversion and instrument rotation.
        self.source_offset_arcsec = nn.Parameter(
            torch.zeros(2, dtype=torch.float64)
        )

        detector_size = self.settings["detector_pixels"]

        # Optional multiplicative detector-pixel correction.
        # Initially one everywhere.
        self.flat_field = nn.Parameter(
            torch.ones(
                (detector_size, detector_size),
                dtype=torch.float64,
            ),
            requires_grad=train_flat_field,
        )

        # The fitting loop updates flux explicitly, as in pyflip.
        self.register_buffer(
            "flux_scale",
            torch.tensor(1.0, dtype=torch.float64),
        )

        # Fixed image registration from the observation's crop geometry.
        self.register_buffer(
            "registration_offset_xy_pixels",
            torch.as_tensor(
                registration_offset_xy_pixels,
                dtype=torch.float64,
            ).clone(),
        )

    def forward(self):
        fine_image = None

        for index in range(
            len(self.propagator.inputs["wavelengths"])
        ):
            arguments = (
                index,
                self.telescope_delta_m,
                self.instrument_delta_m,
                self.source_offset_arcsec,
            )

            if (
                self.checkpoint_wavelengths
                and torch.is_grad_enabled()
            ):
                # Recompute intermediate fields during backpropagation
                # instead of retaining all nine propagation graphs.
                monochromatic = checkpoint(
                    self.propagator.monochromatic,
                    *arguments,
                    use_reentrant=False,
                )
            else:
                monochromatic = (
                    self.propagator.monochromatic(*arguments)
                )

            contribution = (
                self.propagator.weights[index] * monochromatic
            )

            fine_image = (
                contribution
                if fine_image is None
                else fine_image + contribution
            )

        # Apply Gaussian blur and bin to detector pixels.
        image = self.detector(fine_image)

        # Align the simulated coronagraph center with the observed crop.
        image = shift_image(
            image,
            self.registration_offset_xy_pixels,
        )

        return self.flux_scale * image * self.flat_field

    @torch.no_grad()
    def set_flux_scale(self, value):
        """Set a positive brightness scale without gradient optimization."""
        value = torch.as_tensor(
            value,
            dtype=self.flux_scale.dtype,
            device=self.flux_scale.device,
        )

        if (
            value.numel() != 1
            or not torch.isfinite(value).all()
            or value.item() <= 0
        ):
            raise ValueError(
                "Flux scale must be a finite positive scalar."
            )

        self.flux_scale.copy_(value.reshape(()))


def build_miri_model(
    role="reference",
    device="cpu",
    train_flat_field=False,
    settings_path=None,
):
    inputs = load_miri_inputs(role, settings_path=settings_path)
    observations = load_miri_observations(settings_path=settings_path)

    if inputs["settings"] != observations["settings"]:
        raise ValueError(
            "Optical and observation settings differ."
        )

    geometry = read_stpsf_geometry(inputs)

    model = MIRIModel(
        inputs,
        geometry,
        observations["registration_offset_xy_pixels"],
        train_flat_field=train_flat_field,
    ).to(device)

    selected = observations[role]
    observation = dict(selected)

    for name in ("image", "noise", "valid_mask"):
        dtype = (
            torch.bool
            if name == "valid_mask"
            else torch.float64
        )

        observation[name] = torch.as_tensor(
            selected[name],
            dtype=dtype,
            device=device,
        )

    return model, observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=["reference", "target"],
        default="reference",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    args = parser.parse_args()

    model, observation = build_miri_model(
        args.role,
        args.device,
    )

    print("Role:", model.role)
    print("Observed image:", tuple(observation["image"].shape))
    print("Observation unit:", observation["unit"])
    print(
        "Gaussian width (detector pixels):",
        model.detector.sigma_detector_pixels.item(),
    )
    print(
        "Registration offset (x, y):",
        model.registration_offset_xy_pixels.tolist(),
    )

    print("Trainable parameters:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(" ", name, tuple(parameter.shape))

    print(
        "Model assembled; no PSF rendered "
        "and no optimization performed."
    )


if __name__ == "__main__":
    main()
