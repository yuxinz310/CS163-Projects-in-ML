"""Plot MIRI fit results and evaluate the submission-demo SNR statistic.

The estimator follows rferrerc/pyflip submission e0edc43, dl_utils.calc_snr:
pyklip gaussfit2d signal amplitude / annular pixel standard deviation.
MIRI adaptations: pixel scale, coronagraph center, expected planet location.
This is not a small-sample-corrected detection significance or a throughput test.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from pyklip.fakes import gaussfit2d
from pyklip.klip import nan_gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miri_model_inputs import PROJECT_DIR

UPSTREAM_REVISION = "e0edc43a6e517925d9a98edc20edfc654c8db8e3"
ASTROMETRY_SOURCE = "https://doi.org/10.3847/2041-8213/acd93e"


class DetectorDisplay:
    """Original array orientation: x right, y up, no coordinate transform."""

    def __init__(self, manifest):
        self.size = manifest["settings"]["detector_pixels"]
        self.center = ((self.size - 1) / 2
                       + np.asarray(manifest["registration_offset_xy_pixels"]))
        edges = np.arange(self.size + 1) - .5
        self.x_edges, self.y_edges = np.meshgrid(edges, edges)

    def draw(self, axis, image, **kwargs):
        image = np.asarray(image)
        if image.shape != (self.size, self.size):
            raise ValueError("Image shape does not match the detector crop.")
        artist = axis.pcolormesh(self.x_edges, self.y_edges,
                                np.ma.masked_invalid(image), shading="flat", **kwargs)
        axis.set_facecolor("#303030")
        axis.set_aspect("equal")
        axis.set_xlim(-.5, self.size - .5)
        axis.set_ylim(-.5, self.size - .5)
        axis.set_xlabel("Crop x pixel (column)")
        axis.set_ylabel("Crop y pixel (row)")
        axis.plot(*self.center, marker="+", color="white", ms=7, mew=.8)
        return artist

    def metadata(self):
        return {"orientation": "original detector orientation", "axis_unit": "pixel",
                "origin": "zero-based crop coordinates; y increases upward on display",
                "method": "identity mapping; no rotation, reflection or resampling",
                "snr_coordinate_system": "original detector crop pixels",
                "color_scale": {**self.color_scale(), "unit": "MJy/sr",
                                "reference": "Carter et al. 2023, Figure 3, page 8, MIRI F1140C row",
                                "note": "Same display limits also used for F1550C; values outside limits saturate visually."}}

    def color_scale(self):
        """Match the paper's MIRI display; leave the science arrays unchanged."""
        return {"cmap": "inferno", "vmin": -1., "vmax": 2.}


def expected_planet_position(manifest, separation_arcsec=.819, pa_deg=149.8):
    """Approximate crop position from published astrometry and FITS orientation.

    Uses plate scale and orientation, not a full distortion transformation.
    This is a location prior, not an astrometric measurement from our residual.
    """
    settings = manifest["settings"]
    data_dir = PROJECT_DIR / "data" / "HIP65426-MIRI" / f"HIP65426-{settings['filter']}"
    config = json.loads((data_dir / "config.json").read_text())
    filename = manifest["observations"]["target"]["source_fits"]
    header = fits.getheader(data_dir / config["fits_dir"] / filename, extname="SCI")
    orientation = float(header["ROLL_REF"] + header["V3I_YANG"])
    parity = int(header["VPARITY"])
    center = ((settings["detector_pixels"] - 1) / 2
              + np.asarray(manifest["registration_offset_xy_pixels"], dtype=float))
    angle = np.deg2rad(pa_deg - orientation)
    offset = np.array([parity * separation_arcsec * np.sin(angle),
                       separation_arcsec * np.cos(angle)]) / settings["pixel_scale_arcsec"]
    return center + offset, center, {
        "source": ASTROMETRY_SOURCE, "separation_arcsec": separation_arcsec,
        "position_angle_deg": pa_deg, "fits_orientation_deg": orientation,
        "fits_parity": parity, "source_fits": filename,
        "method": "published position prior, FITS orientation and nominal plate scale",
    }


def measure_pyflip_snr(image, valid_mask, planet_xy, center_xy, pixel_scale,
                       signal_radius_arcsec=1.4, annulus_half_width_arcsec=.5,
                       smoothing_sigma_pixels=1., centroid_tolerance_pixels=1.):
    """Preserve the original demo's masks, smoothing, fitting and noise definition."""
    image = np.asarray(image, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or valid_mask.shape != image.shape:
        raise ValueError("Expected a square residual and matching validity mask.")
    n = image.shape[0]
    planet_xy = np.asarray(planet_xy, dtype=float)
    center_xy = np.asarray(center_xy, dtype=float)
    if planet_xy.shape != (2,) or center_xy.shape != (2,) or not np.isfinite(planet_xy).all() or not np.isfinite(center_xy).all():
        raise ValueError("Positions must be finite (x, y) pairs.")
    if min(pixel_scale, signal_radius_arcsec, annulus_half_width_arcsec, centroid_tolerance_pixels) <= 0 or smoothing_sigma_pixels < 0:
        raise ValueError("Invalid SNR geometry or smoothing settings.")
    if np.any(np.rint(planet_xy) - 5 < 0) or np.any(np.rint(planet_xy) + 5 >= n):
        raise ValueError("The Gaussian fitting box extends outside the crop.")
    image = np.where(valid_mask & np.isfinite(image), image, np.nan)
    smoothed = nan_gaussian_filter(image, smoothing_sigma_pixels)
    shift = center_xy - (n - 1) / 2
    # Intentionally retain pyflip's linspace convention for comparability.
    axis = np.linspace(-int(n / 2) * pixel_scale, int(n / 2) * pixel_scale, n)
    x, y = np.meshgrid(axis - shift[0] * pixel_scale, axis - shift[1] * pixel_scale)
    offset = (planet_xy - center_xy) * pixel_scale
    inside_signal = (x - offset[0])**2 + (y - offset[1])**2 <= signal_radius_arcsec**2
    blob = np.where(inside_signal, smoothed, np.nan)
    if not np.isfinite(blob).any():
        raise ValueError("The signal region has no usable pixels.")
    signal, fwhm, fit_x, fit_y = gaussfit2d(
        blob, *planet_xy, searchrad=5, guessfwhm=3,
        guesspeak=np.nanmax(blob), refinefit=True,
    )
    radius = float(np.linalg.norm(offset))
    in_annulus = ((x*x + y*y > (radius - annulus_half_width_arcsec)**2)
                  & (x*x + y*y < (radius + annulus_half_width_arcsec)**2))
    annulus = np.where(in_annulus & np.isnan(blob), smoothed, np.nan)
    count = int(np.isfinite(annulus).sum())
    noise = float(np.nanstd(annulus)) if count >= 2 else float("nan")
    if not np.isfinite([signal, fwhm, fit_x, fit_y, noise]).all() or noise <= 0:
        raise ValueError("Gaussian fit or annular-noise estimate is invalid.")
    displacement = float(np.linalg.norm(np.array([fit_x, fit_y]) - planet_xy))
    snr = float(signal / noise)
    consistent = displacement <= centroid_tolerance_pixels and signal > 0 and fwhm > 0
    result = {
        "raw_pyflip_snr": snr, "signal": float(signal), "annular_noise": noise,
        "expected_xy": planet_xy.tolist(), "fitted_xy": [float(fit_x), float(fit_y)],
        "fitted_fwhm_pixels": float(fwhm), "centroid_displacement_pixels": displacement,
        "centroid_consistent_with_prior": bool(consistent),
        "centroid_consistent_snr": snr if consistent else None, "noise_pixels": count,
    }
    return result, annulus, blob


def read_snapshot(path):
    with np.load(path) as saved:
        return {name: saved[name].copy() for name in saved.files}


def add_planet_arrow(axis, planet_xy, fit_xy=None):
    axis.annotate("Expected planet", xy=planet_xy, xytext=(.03, .95),
                  textcoords="axes fraction", color="lime", fontsize=8,
                  arrowprops=dict(arrowstyle="->", color="lime", lw=.7, shrinkB=4))
    if fit_xy is not None:
        axis.annotate("Fitted peak", xy=fit_xy, xytext=(.68, .08),
                      textcoords="axes fraction", color="orange", fontsize=8,
                      arrowprops=dict(arrowstyle="->", color="orange", lw=.7, shrinkB=4))


def plot_final(snapshot, result, planet_xy, output, unit, display):
    valid = snapshot["valid_mask"].astype(bool)
    observed = np.where(valid, snapshot["observed"], np.nan)
    predicted = np.where(valid, snapshot["prediction"], np.nan)
    residual = np.where(valid, snapshot["residual"], np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, title, image in zip(axes[:2], ("Observation", "Fitted stellar model"), (observed, predicted)):
        artist = display.draw(ax, image, **display.color_scale())
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, label=unit)
    artist = display.draw(axes[2], residual, **display.color_scale())
    axes[2].set_title("Residual: observation − model")
    fig.colorbar(artist, ax=axes[2], label=unit)
    fitted = result.get("fitted_xy")
    add_planet_arrow(axes[2], planet_xy, None if fitted is None else fitted)
    fig.suptitle(f"Target after update {int(snapshot['iteration'])}; original detector orientation; unsmoothed")
    fig.savefig(output / "target_summary.png", dpi=180); plt.close(fig)


def plot_residual_evolution(paths, planet_xy, output, unit, display):
    indices = np.unique(np.linspace(0, len(paths) - 1, min(4, len(paths)), dtype=int))
    snapshots = [read_snapshot(paths[index]) for index in indices]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False, constrained_layout=True)
    for ax, snapshot, index in zip(axes[0], snapshots, indices):
        image = np.where(snapshot["valid_mask"], snapshot["residual"], np.nan)
        artist = display.draw(ax, image, **display.color_scale())
        ax.set_title("Before target fitting" if paths[index].stem == "target_initial" else f"After update {int(snapshot['iteration'])}")
        add_planet_arrow(ax, planet_xy)
    fig.colorbar(artist, ax=list(axes[0]), label=unit)
    fig.suptitle("Target residual evolution — original detector orientation; common color scale, unsmoothed")
    fig.savefig(output / "residual_evolution.png", dpi=180); plt.close(fig)


def plot_history(history, measurements, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, role in zip(axes[0], ("reference", "target")):
        rows = [row for row in history if row["role"] == role]
        ax.plot([r["iteration"] for r in rows], [r["data_loss_before_update"] for r in rows])
        ax.set_title(f"{role.capitalize()} data loss before update")
        ax.set_ylabel("Smooth L1" if role == "reference" else "L1")
    for role in ("reference", "target"):
        rows = [row for row in history if row["role"] == role]
        axes[1, 0].plot([r["iteration"] for r in rows], [r["rms_before_update"] for r in rows], label=role)
    axes[1, 0].set_title("Masked residual RMS before update")
    axes[1, 0].legend()
    for consistent, marker, label in [(True, "o", "Centroid within tolerance"), (False, "x", "Centroid/fit check failed")]:
        rows = [r for r in measurements if "raw_pyflip_snr" in r and r["centroid_consistent_with_prior"] == consistent]
        axes[1, 1].scatter([r["iteration"] for r in rows], [r["raw_pyflip_snr"] for r in rows], marker=marker, s=18, label=label)
    axes[1, 1].set_title("Target pyflip-style SNR statistic after update")
    axes[1, 1].set_ylabel("Signal / annular standard deviation")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Iteration"); ax.grid(alpha=.2)
    fig.savefig(output / "optimization_history.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--planet-xy", type=float, nargs=2, help="Override expected crop position, x y.")
    parser.add_argument("--separation-arcsec", type=float, default=.819)
    parser.add_argument("--pa-deg", type=float, default=149.8)
    parser.add_argument("--signal-radius", type=float, default=1.4)
    parser.add_argument("--annulus-half-width", type=float, default=.5)
    parser.add_argument("--smoothing-sigma", type=float, default=1.)
    parser.add_argument("--centroid-tolerance", type=float, default=1.)
    parser.add_argument("--video-fps", type=float, default=5., help="MP4 playback frames per second (default: 5).")
    parser.add_argument("--skip-video", action="store_true", help="Generate only the plots and SNR report.")
    args = parser.parse_args()
    if not np.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("--video-fps must be finite and positive")
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    if not history:
        raise ValueError("No completed iteration records found.")
    settings = manifest["settings"]
    display = DetectorDisplay(manifest)
    if args.planet_xy is None:
        planet, center, position_info = expected_planet_position(manifest, args.separation_arcsec, args.pa_deg)
    else:
        planet = np.asarray(args.planet_xy, dtype=float)
        center = (settings["detector_pixels"] - 1) / 2 + np.asarray(manifest["registration_offset_xy_pixels"])
        position_info = {"method": "user supplied crop coordinates"}
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    paths = sorted(p for p in run.glob("target_*.npz") if p.stem.removeprefix("target_").isdigit())
    if (run / "target_initial.npz").exists():
        paths.insert(0, run / "target_initial.npz")
    measurements = []
    final_snapshot = None
    for path in paths:
        snapshot = read_snapshot(path)
        row = {"file": path.name, "iteration": int(snapshot["iteration"]), "before_target_fitting": path.stem == "target_initial"}
        try:
            result, annulus, blob = measure_pyflip_snr(
                snapshot["residual"], snapshot["valid_mask"], planet, center,
                settings["pixel_scale_arcsec"], args.signal_radius, args.annulus_half_width,
                args.smoothing_sigma, args.centroid_tolerance,
            )
            row.update(result)
            if path == paths[-1]:
                np.savez_compressed(output / "snr_regions.npz", annulus=annulus, signal_region=blob)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            row.update(error=str(error), centroid_consistent_snr=None)
        measurements.append(row)
        final_snapshot = snapshot
    report = {
        "method": "pyflip submission Gaussian-fit amplitude / annular pixel standard deviation",
        "upstream_revision": UPSTREAM_REVISION,
        "pixel_scale_arcsec": settings["pixel_scale_arcsec"], "coronagraph_center_xy": center.tolist(),
        "expected_planet_xy": planet.tolist(), "position_prior": position_info,
        "signal_radius_arcsec": args.signal_radius, "annulus_half_width_arcsec": args.annulus_half_width,
        "smoothing_sigma_detector_pixels": args.smoothing_sigma, "centroid_tolerance_pixels": args.centroid_tolerance,
        "blur_annulus": True, "blur_before_signal": True,
        "mask_grid": "upstream linspace convention with MIRI scale and center",
        "gaussfit2d": {"searchrad": 5, "guessfwhm": 3, "refinefit": True},
        "noise_ddof": 0, "small_sample_correction": False, "throughput_calibrated": False,
        "interpretation": "Centroid consistency is a diagnostic, not proof of a detection.",
        "image_display": display.metadata(),
        "measurements": measurements, "last_saved_target": measurements[-1] if measurements else None,
        "versions": {name: importlib.metadata.version(name) for name in ("pyklip", "numpy", "scipy", "astropy")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "snr_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_history(history, measurements, output)
    if final_snapshot is not None:
        unit = manifest["observations"]["target"]["unit"]
        plot_final(final_snapshot, measurements[-1], planet, output, unit, display)
        plot_residual_evolution(paths, planet, output, unit, display)
        if not args.skip_video:
            from miri_video import write_residual_video
            video = write_residual_video(paths, measurements, planet, output, unit, display, args.video_fps)
            print("Saved video:", video)
    print("Expected planet crop position:", planet)
    print("Last saved target measurement:", json.dumps(report["last_saved_target"], indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
