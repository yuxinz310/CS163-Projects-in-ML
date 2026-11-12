import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianDetector(nn.Module):
    """Apply Gaussian blur, then sum subpixels into detector pixels.

    sigma_detector_pixels is expressed in detector pixels.
    This is a simplified response model, not a calibrated MIRI response.
    """

    def __init__(
        self,
        detector_pixels=80,
        detector_oversample=4,
        sigma_detector_pixels=0.28,
        trainable_sigma=False,
    ):
        super().__init__()

        for value in (detector_pixels, detector_oversample):
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    "Detector sampling must use positive integers."
                )

        if (
            not math.isfinite(sigma_detector_pixels)
            or sigma_detector_pixels <= 0
        ):
            raise ValueError("Gaussian width must be finite and positive.")

        self.detector_pixels = detector_pixels
        self.detector_oversample = detector_oversample

        # Store the width in detector-pixel units.
        # Logarithmic representation keeps the width positive.
        self.log_sigma = nn.Parameter(
            torch.tensor(
                math.log(sigma_detector_pixels),
                dtype=torch.float64,
            ),
            requires_grad=trainable_sigma,
        )

        sigma_fine = (
            sigma_detector_pixels * detector_oversample
        )

        # Centered, odd-sized kernel extending approximately 4 sigma.
        # Its support remains fixed if sigma is later made trainable.
        radius = max(1, math.ceil(4 * sigma_fine))
        self.radius = radius
        self.kernel_size = 2 * radius + 1

        self.register_buffer(
            "coordinates",
            torch.arange(
                -radius,
                radius + 1,
                dtype=torch.float64,
            ),
        )

    @property
    def sigma_detector_pixels(self):
        return self.log_sigma.exp()

    def blur(self, intensity):
        """Blur an oversampled 2D intensity image."""
        expected_size = (
            self.detector_pixels * self.detector_oversample
        )

        if intensity.shape != (expected_size, expected_size):
            raise ValueError(
                f"Expected {(expected_size, expected_size)}, "
                f"got {tuple(intensity.shape)}."
            )

        sigma = (
            self.sigma_detector_pixels
            * self.detector_oversample
        ).to(dtype=intensity.dtype)

        coordinates = self.coordinates.to(
            dtype=intensity.dtype
        )

        kernel_1d = torch.exp(
            -0.5 * (coordinates / sigma).square()
        )
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = torch.outer(kernel_1d, kernel_1d)

        blurred = F.conv2d(
            intensity[None, None],
            kernel_2d[None, None],
            padding=self.radius,
        )

        return blurred[0, 0]

    def bin_pixels(self, intensity):
        """Sum each oversampled block into one detector pixel."""
        n = self.detector_pixels
        s = self.detector_oversample

        return intensity.reshape(n, s, n, s).sum(dim=(1, 3))

    def forward(self, oversampled_intensity):
        blurred = self.blur(oversampled_intensity)
        return self.bin_pixels(blurred)