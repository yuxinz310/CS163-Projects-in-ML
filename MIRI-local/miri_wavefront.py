import argparse

import torch

from miri_model_inputs import load_miri_inputs


def create_entrance_field(
    transmission,
    telescope_opd,
    wavelength_m,
):
    """Construct the complex light field in entrance-pupil coordinates.

    transmission: 2D torch tensor, electric-field amplitude transmission.
    telescope_opd: 2D torch tensor, OPD in meters.
    wavelength_m: positive wavelength in meters.
    """
    if transmission.ndim != 2:
        raise ValueError("Transmission must be a 2D array.")

    if transmission.shape != telescope_opd.shape:
        raise ValueError("Transmission and OPD shapes must match.")

    if wavelength_m <= 0:
        raise ValueError("Wavelength must be positive.")

    phase = 2 * torch.pi * telescope_opd / wavelength_m

    return transmission * torch.exp(1j * phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=["reference", "target"],
        default="reference",
    )
    args = parser.parse_args()

    inputs = load_miri_inputs(args.role)

    # Use CPU and double precision for this small initial check.
    transmission = torch.as_tensor(
        inputs["primary_transmission"],
        dtype=torch.float64,
        device="cpu",
    )
    telescope_opd = torch.as_tensor(
        inputs["telescope_opd"],
        dtype=torch.float64,
        device="cpu",
    )

    # Check one wavelength first.
    wavelength = float(inputs["wavelengths"][0])

    field = create_entrance_field(
        transmission,
        telescope_opd,
        wavelength,
    )

    intensity = field.abs().square()

    # OPD changes phase, so intensity at this plane should remain T².
    intensity_error = (
        intensity - transmission.square()
    ).abs().max()

    print("Role:", args.role)
    print("Wavelength:", wavelength * 1e6, "micrometers")
    print("Field shape:", tuple(field.shape))
    print("Field dtype:", field.dtype)
    print("All values finite:", torch.isfinite(field).all().item())
    print("Maximum intensity error:", intensity_error.item())


if __name__ == "__main__":
    main()