"""Prepare and run one MIRI experiment with selectable wavelength sampling."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

from miri_model_inputs import PROJECT_DIR, load_miri_inputs
from miri_observations import load_miri_observations
from miri_paths import model_input_dir, telescope_input_dir, settings_for_filter, FILTER_MASKS
from run_miri_demo import validate_config
from setup_miri_calibration import ensure_calibration


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def resolve_fit_config(base, args):
    """Apply CLI overrides without modifying the user's base configuration."""
    cfg = dict(base)
    validate_config(cfg)
    if args.schedule == "proportional":
        if args.iterations is None:
            raise ValueError("--schedule proportional requires --iterations.")
        ratio = args.iterations / cfg["iters"]
        for key in ("ref_cutoff_iter", "stage_fluxpos_cutoff_iter",
                    "stage_fit_flat_fields_start_iter", "stage_fit_flat_fields_end_iter"):
            cfg[key] = round(cfg[key] * ratio)
        if min(cfg["ref_cutoff_iter"], cfg["stage_fluxpos_cutoff_iter"]) < 1:
            raise ValueError("Run is too short for the proportional fitting stages.")
    if args.iterations is not None:
        cfg["iters"] = cfg["forced_stop_iter"] = args.iterations
        # Milestones are absolute iteration labels; keep 11000 when reachable.
        cfg["save_iterations"] = sorted({
            i for i in cfg.get("save_iterations", []) if i <= args.iterations
        } | {args.iterations})
    for argument, key in (("lr", "lr"), ("save_every", "save_every"),
                          ("snr_every", "snr_every"),
                          ("reference_iterations", "ref_cutoff_iter")):
        value = getattr(args, argument)
        if value is not None:
            cfg[key] = value
    validate_config(cfg)
    if cfg["stage_fit_flat_fields_end_iter"] > cfg["forced_stop_iter"]:
        raise ValueError("Run ends before the flat-field stage finishes. Use "
                         "--schedule proportional or supply a compatible --fit-config.")
    return cfg


def save_configuration(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"Saved configuration differs: {path}")
    else:
        path.write_text(json.dumps(settings, indent=2) + "\n")


def execute(script, *args):
    command = [sys.executable, "-u", str(PROJECT_DIR / script), *map(str, args)]
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_DIR)


def prepare_inputs(settings_path):
    settings = json.loads(settings_path.read_text())
    ensure_calibration(PROJECT_DIR)
    metadata = (PROJECT_DIR / "data" / "HIP65426-MIRI" /
                f"HIP65426-{settings['filter']}" / "observation_metadata.json")
    if not metadata.is_file():
        execute("prepare_observation_metadata.py", "--filter", settings["filter"])
    # A fresh clone contains source calibration, not generated telescope maps.
    for role in ("reference", "target"):
        telescope = telescope_input_dir(settings, role, PROJECT_DIR)
        if not all((telescope / name).is_file() for name in ("telescope_opd.npy", "metadata.json")):
            execute("miri_instrument.py", "--role", role, "--settings", settings_path)
    folder = model_input_dir(settings)
    # Never overwrite an input set belonging to a different configuration.
    for metadata in folder.glob("*/metadata.json"):
        saved = json.loads(metadata.read_text())
        if saved.get("settings") != settings:
            raise ValueError(f"Existing inputs have different settings: {metadata}")
    try:
        for role in ("reference", "target"):
            inputs = load_miri_inputs(role, settings_path=settings_path)
            del inputs
    except FileNotFoundError:
        for script in ("prepare_miri_wavelengths.py", "prepare_miri_instrument_opds.py",
                       "prepare_miri_pupils.py", "prepare_miri_focal_mask.py"):
            execute(script, "--settings", settings_path)
        for role in ("reference", "target"):
            inputs = load_miri_inputs(role, settings_path=settings_path)
            del inputs
    load_miri_observations(settings_path=settings_path)
    print("Prepared inputs verified:", folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=sorted(FILTER_MASKS),
                        help="Observation filter; defaults to miri_settings.json (F1140C).")
    parser.add_argument("--wavelengths", type=positive_integer, required=True)
    parser.add_argument("--oversample", type=positive_integer, default=4,
                        help="Set both FFT and detector oversampling (default: 4).")
    parser.add_argument("--fft-oversample", type=positive_integer,
                        help="Override FFT oversampling only.")
    parser.add_argument("--detector-oversample", type=positive_integer,
                        help="Override detector oversampling only.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--fit-config", type=Path, default=PROJECT_DIR / "miri_fit_settings.json",
                        help="Optimizer/schedule settings, shared unless explicitly changed.")
    parser.add_argument("--iterations", type=positive_integer,
                        help="Final iteration label and LR schedule horizon; runs 0..N inclusive, as in pyflip.")
    parser.add_argument("--lr", type=positive_float, help="Base learning rate; retain parameter-group multipliers.")
    parser.add_argument("--schedule", choices=["fixed", "proportional"], default="fixed",
                        help="Keep stage boundaries (default), or scale them with --iterations.")
    parser.add_argument("--reference-iterations", type=positive_integer,
                        help="Override the reference-to-target switch after optional schedule scaling.")
    parser.add_argument("--save-every", type=positive_integer, help="Snapshot/checkpoint interval.")
    parser.add_argument("--snr-every", type=positive_integer, help="Target SNR interval; also saves a snapshot.")
    parser.add_argument("--print-config", action="store_true",
                        help="Print resolved settings and exit without writes, input preparation, or fitting.")
    parser.add_argument("--output-dir", type=Path, help="New result folder; otherwise create a timestamped folder.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and validate inputs without fitting.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip post-fit plots and MP4.")
    parser.add_argument("--max-updates", type=positive_integer, help="Optional short smoke run.")
    parser.add_argument("--resume", type=Path, help="Resume a matching checkpoint with the same sampling flags.")
    args = parser.parse_args()
    settings = json.loads((PROJECT_DIR / "miri_settings.json").read_text())
    if args.filter is not None:
        settings = settings_for_filter(settings, args.filter, PROJECT_DIR)
    settings.update(num_wavelengths=args.wavelengths,
                    fft_oversample=args.fft_oversample or args.oversample,
                    detector_oversample=args.detector_oversample or args.oversample)
    try:
        fit_settings = resolve_fit_config(json.loads(args.fit_config.read_text()), args)
    except ValueError as error:
        parser.error(f"{error} For a shorter run, consider --schedule proportional.")
    if args.print_config:
        print(json.dumps({"settings": settings, "fit_settings": fit_settings}, indent=2))
        return
    tag = f"n{args.wavelengths}_fft{settings['fft_oversample']}_det{settings['detector_oversample']}"
    if settings["filter"] != "F1140C":
        tag = f"{settings['filter']}_{tag}"
    if args.resume:
        output = args.resume.resolve().parent
        if args.output_dir and args.output_dir.resolve() != output:
            parser.error("--output-dir must be the checkpoint's existing directory when resuming")
        if not args.resume.is_file():
            parser.error("The resume checkpoint does not exist")
    else:
        output = (args.output_dir.resolve() if args.output_dir else PROJECT_DIR / "outputs" / "optimization"
                  / f"{tag}_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')}")
        if output.exists() and not args.prepare_only:
            parser.error(f"Output already exists: {output}. Choose a new folder or use --resume.")
    if not args.prepare_only and args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error("No CUDA GPU is available here. Use a GPU machine, --device cpu, or --prepare-only.")
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    settings_path = PROJECT_DIR / "configs" / "generated" / f"{tag}_{digest}.json"
    save_configuration(settings_path, settings)
    fit_digest = hashlib.sha256(json.dumps(fit_settings, sort_keys=True).encode()).hexdigest()[:12]
    fit_path = PROJECT_DIR / "configs" / "generated" / f"fit_{fit_digest}.json"
    save_configuration(fit_path, fit_settings)
    print("Fitting settings:", fit_path, flush=True)
    print(f"Iteration labels: 0..{fit_settings['forced_stop_iter']} inclusive; "
          f"target stage starts at {fit_settings['ref_cutoff_iter']}; "
          f"base LR: {fit_settings['lr']}", flush=True)
    prepare_inputs(settings_path)
    print("Optical settings:", settings_path, flush=True)
    if args.prepare_only:
        return
    command = ["--device", args.device, "--settings", settings_path,
               "--config", fit_path, "--output-dir", output]
    if args.resume:
        command += ["--resume", args.resume.resolve()]
    if args.max_updates:
        command += ["--max-updates", args.max_updates]
    execute("run_miri_demo.py", *command)
    if not args.skip_analysis:
        execute("miri_results.py", output)
    print("Results:", output)


if __name__ == "__main__":
    main()
