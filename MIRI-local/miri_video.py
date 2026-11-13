"""Render saved target residuals to MP4 without changing science arrays."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def write_residual_video(paths, measurements, planet_xy, output, unit, display, fps=5):
    """One frame per snapshot; hold iteration 11000 and the last frame for 2 s."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive.")
    if not paths:
        raise ValueError("No saved target residuals are available for the video.")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to enable MP4 output.") from error
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    paths = [Path(p) for p in paths]
    records = {row["file"]: row for row in measurements}
    valid_count, iterations = 0, []
    for path in paths:
        with np.load(path) as saved:
            residual = saved["residual"]
            valid = saved["valid_mask"].astype(bool) & np.isfinite(residual)
            if residual.shape != (display.size, display.size):
                raise ValueError(f"Unexpected residual shape in {path}.")
            valid_count += int(valid.sum())
            iterations.append(int(saved["iteration"]))
    if any(b < a for a, b in zip(iterations, iterations[1:])):
        raise ValueError("Video snapshots must be in chronological order.")
    if valid_count == 0:
        raise ValueError("No finite, valid residual pixels to display.")
    color_scale = display.color_scale()
    output = Path(output)
    destination = output / "residual_evolution.mp4"
    temporary = output / "residual_evolution.tmp.mp4"
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(left=.13, right=.85, bottom=.14, top=.86)
    artist = display.draw(ax, np.full((display.size, display.size), np.nan),
                      **color_scale)
    fig.colorbar(artist, ax=ax, label=unit, fraction=.045, pad=.04)
    ax.annotate("Expected planet", xy=planet_xy, xytext=(.03, .95),
                textcoords="axes fraction", color="lime", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="lime", lw=.7, shrinkB=4))
    title = fig.suptitle("")
    status = fig.text(.5, .035, "", ha="center", fontsize=9)
    timeline = []
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400,
                         extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"],
                         metadata={"title": "MIRI target residual evolution",
                                   "comment": "Original detector orientation; fixed color scale; saved iterations only"})
    try:
        with matplotlib.rc_context({"animation.ffmpeg_path": encoder}):
            with writer.saving(fig, str(temporary), dpi=120):
                for index, (path, iteration) in enumerate(zip(paths, iterations)):
                    with np.load(path) as saved:
                        image = np.where(saved["valid_mask"], saved["residual"], np.nan)
                    artist.set_array(np.ma.masked_invalid(image).ravel())
                    baseline = path.stem == "target_initial"
                    label = "Before target fitting" if baseline else f"After iteration {iteration}"
                    title.set_text(f"{label}\nOriginal detector orientation — fixed color scale, unsmoothed residual")
                    row = records.get(path.name, {})
                    snr = row.get("raw_pyflip_snr")
                    if snr is not None and np.isfinite(snr):
                        check = "centroid within tolerance" if row.get("centroid_consistent_with_prior") else "centroid/fit check failed"
                        snr_label = f"SNR statistic: {snr:.2f} ({check})"
                    else:
                        snr_label = "SNR statistic unavailable"
                    status.set_text(f"{snr_label}\nSaved snapshot {index + 1}/{len(paths)}; playback is not wall-clock time")
                    repeats = max(1, round(2 * fps)) if iteration == 11000 or index == len(paths) - 1 else 1
                    start = sum(row["frame_count"] for row in timeline)
                    timeline.append({"file": path.name, "iteration": iteration,
                                     "before_target_fitting": baseline,
                                     "first_frame": start, "frame_count": repeats})
                    for _ in range(repeats):
                        writer.grab_frame()
        temporary.replace(destination)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    report = {"file": destination.name, "fps": fps,
              "frame_count": sum(row["frame_count"] for row in timeline),
              "color_limits": [color_scale["vmin"], color_scale["vmax"]],
              "colormap": color_scale["cmap"],
              "color_scale_method": "Fixed MIRI display limits from Carter et al. 2023, Figure 3, page 8",
              "brightness_unit": unit, "image_display": display.metadata(),
              "timing": "one frame per saved snapshot; 11000 and final snapshot held for two seconds",
              "frames": timeline}
    (output / "residual_video.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return destination
