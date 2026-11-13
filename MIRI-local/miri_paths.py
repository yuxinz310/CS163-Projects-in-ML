"""Shared names for prepared optical inputs."""
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent
FILTER_MASKS = {"F1140C": "FQPM1140", "F1550C": "FQPM1550"}


def settings_for_filter(settings, filter_name, project_dir=PROJECT_DIR):
    """Use the supplied observation config for filter-dependent geometry."""
    if filter_name not in FILTER_MASKS:
        raise ValueError(f"Unsupported filter: {filter_name}")
    config = json.loads((Path(project_dir) / "data" / "HIP65426-MIRI" /
                         f"HIP65426-{filter_name}" / "config.json").read_text())
    if config["instrument"] != "MIRI" or config["filter"] != filter_name:
        raise ValueError("Observation configuration does not match the selected filter.")
    return dict(settings, instrument="MIRI", filter=filter_name,
                image_mask=FILTER_MASKS[filter_name], pupil_mask="MASKFQPM",
                pixel_scale_arcsec=config["pixel_scale_arcsec"],
                detector_pixels=config["num_det_px"])


def telescope_input_dir(settings, role, project_dir=PROJECT_DIR):
    return Path(project_dir) / "model_inputs" / "telescope_opd" / settings["filter"] / role


def opd_selection_path(filter_name, project_dir=PROJECT_DIR):
    if filter_name not in FILTER_MASKS:
        raise ValueError(f"Unsupported filter: {filter_name}")
    # Retain the original selection filename for F1140C compatibility.
    name = "opd_selection.json" if filter_name == "F1140C" else f"opd_selection_{filter_name}.json"
    return Path(project_dir) / "calibration" / "measured_opds" / name


def load_opd_selections(settings, project_dir=PROJECT_DIR):
    return json.loads(opd_selection_path(settings["filter"], project_dir).read_text())


def model_input_dir(settings):
    source = settings["source_spectrum"]
    name = (f"{settings['filter']}_{source['catalog']}_"
            f"{source['spectral_type']}_n{settings['num_wavelengths']}")
    fft = settings["fft_oversample"]
    detector = settings["detector_oversample"]
    # Preserve the existing 4x input sets. Other samplings get separate folders.
    if (fft, detector) != (4, 4):
        name += f"_fft{fft}_det{detector}"
    return PROJECT_DIR / "model_inputs" / name
