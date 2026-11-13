"""Two-stage MIRI fit following the pyflip submission demo's schedule.

Keeps one model and one AdamW optimizer across the reference/target switch.
Uses the original gradient-zeroing freeze behavior, including Adam momentum.
No planet is injected or excluded from the fitting mask.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from miri_model_inputs import PROJECT_DIR, load_miri_inputs
from miri_model import build_miri_model
from miri_observations import load_miri_observations

ARCSEC_PER_RADIAN = 180 * 3600 / math.pi
PARAMETERS = ("telescope_delta_m", "instrument_delta_m", "source_offset_arcsec", "flat_field")


def validate_config(cfg):
    integers = ("iters", "forced_stop_iter", "ref_cutoff_iter", "stage_fluxpos_cutoff_iter",
                "stage_fit_flat_fields_start_iter", "stage_fit_flat_fields_end_iter", "fluxwindsize", "save_every")
    for name in integers:
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if not 0 < cfg["ref_cutoff_iter"] <= cfg["forced_stop_iter"] <= cfg["iters"]:
        raise ValueError("Require 0 < reference switch <= stopping iteration <= schedule horizon.")
    if cfg["save_every"] < 1 or cfg["fluxwindsize"] < 2 or cfg["fluxwindsize"] % 2:
        raise ValueError("save_every must be positive; fluxwindsize must be positive and even.")
    if cfg["stage_fit_flat_fields_start_iter"] > cfg["stage_fit_flat_fields_end_iter"]:
        raise ValueError("Invalid flat-field interval.")
    for name in ("lr", "big_lr_factor", "even_bigger_lr_factor", "sci_lr_weight"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name in ("fit_flat_field", "freeze_position", "freeze_OTE_OPD_sci", "weigh_reference_by_noise", "weigh_loss_by_noise"):
        if type(cfg[name]) is not bool:
            raise ValueError(f"{name} must be a JSON boolean.")
    if cfg["inject_how_many_sigma_flux"] != 0:
        raise ValueError("This driver currently supports injection strength zero only.")
    if type(cfg.get("snr_every", cfg["save_every"])) is not int or cfg.get("snr_every", cfg["save_every"]) < 1:
        raise ValueError("snr_every must be a positive integer.")
    milestones = cfg.get("save_iterations", [])
    if not isinstance(milestones, list) or any(
        type(i) is not int or not 0 <= i <= cfg["forced_stop_iter"] for i in milestones
    ):
        raise ValueError("save_iterations must list iterations within the run.")


def save_plan(iteration, last, cfg, track_snr=True):
    """Milestones force both a save and a target SNR measurement."""
    role, _ = stage_at(iteration, cfg)
    milestone = iteration in cfg.get("save_iterations", [])
    snr_due = track_snr and role == "target" and (
        iteration % cfg.get("snr_every", cfg["save_every"]) == 0
        or milestone or iteration == last or iteration == cfg["ref_cutoff_iter"]
    )
    save_due = (iteration % cfg["save_every"] == 0 or milestone or snr_due
                or iteration == cfg["ref_cutoff_iter"] - 1 or iteration == last)
    return save_due, snr_due, milestone


def stage_at(iteration, cfg):
    target = iteration >= cfg["ref_cutoff_iter"]
    # Preserve upstream's exact off-by-one boundary: cutoff = ref_cutoff - 1.
    position_end = (cfg["ref_cutoff_iter"] - 1 if target else 0) + cfg["stage_fluxpos_cutoff_iter"]
    return ("target" if target else "reference"), iteration < position_end


def fitting_loss(prediction, observation, role, cfg):
    valid = observation["valid_mask"]
    difference = prediction[valid] - observation["image"][valid]
    noise = observation["noise"][valid]
    if role == "reference":
        values = F.smooth_l1_loss(prediction[valid], observation["image"][valid], reduction="none", beta=1.)
        if cfg["weigh_reference_by_noise"]:
            values = values / (noise.square() + 1e-8)
    else:
        values = difference.abs()
        if cfg["weigh_loss_by_noise"]:
            values = values / (noise + 1e-8)
    return values.mean()


def make_optimizer(model, cfg):
    groups = [
        {"params": [model.telescope_delta_m, model.instrument_delta_m], "name": "opd", "factor": 1., "unit_scale": 1.},
        {"params": [model.source_offset_arcsec], "name": "pointing", "factor": cfg["big_lr_factor"],
         "unit_scale": ARCSEC_PER_RADIAN, "eps": 1e-8 / ARCSEC_PER_RADIAN},
    ]
    if cfg["fit_flat_field"]:
        groups.append({"params": [model.flat_field], "name": "flat_field", "factor": cfg["even_bigger_lr_factor"], "unit_scale": 1.})
    for group in groups:
        group["lr"] = cfg["lr"] * group["factor"] * group["unit_scale"]
    return torch.optim.AdamW(groups, weight_decay=0., eps=1e-8)


def set_learning_rates(optimizer, iteration, cfg):
    # Same cosine curve as upstream, with its shared minimum LR in upstream units.
    cosine = (1 + math.cos(math.pi * iteration / cfg["iters"])) / 2
    for group in optimizer.param_groups:
        initial = cfg["lr"] * group["factor"]
        group["lr"] = (cfg["lr"] + (initial - cfg["lr"]) * cosine) * group["unit_scale"]


def zero_gradient(parameter):
    if parameter.grad is not None:
        parameter.grad.zero_()


def apply_stage_rules(model, iteration, cfg):
    role, flux_position = stage_at(iteration, cfg)
    if role == "target" and cfg["freeze_OTE_OPD_sci"]:
        zero_gradient(model.telescope_delta_m)
    if flux_position:
        zero_gradient(model.telescope_delta_m)
        zero_gradient(model.instrument_delta_m)
    elif cfg["freeze_position"]:
        zero_gradient(model.source_offset_arcsec)
    if not (cfg["stage_fit_flat_fields_start_iter"] <= iteration <= cfg["stage_fit_flat_fields_end_iter"]):
        zero_gradient(model.flat_field)


@torch.no_grad()
def match_flux(model, prediction, observation, window):
    n = prediction.shape[-1]
    if window > n:
        raise ValueError("Flux window is larger than the image.")
    start = n // 2 - window // 2
    region = torch.zeros_like(observation["valid_mask"])
    region[start:start + window, start:start + window] = True
    region &= observation["valid_mask"]
    predicted_sum = prediction.detach()[region].sum()
    observed_sum = observation["image"][region].sum()
    if not (torch.isfinite(predicted_sum) and torch.isfinite(observed_sum)) or min(predicted_sum.item(), observed_sum.item()) <= 0:
        raise ValueError("Cannot normalize flux: central valid-pixel sums must be finite and positive.")
    model.set_flux_scale(model.flux_scale * observed_sum / predicted_sum)


@torch.no_grad()
def save_snapshot(model, observation, path, iteration):
    prediction = model()
    if not torch.isfinite(prediction).all():
        raise ValueError("Snapshot contains nonfinite predictions.")
    valid = observation["valid_mask"].cpu().numpy()
    observed = observation["image"].cpu().numpy()
    predicted = prediction.cpu().numpy()
    snapshot = dict(iteration=iteration, observed=observed, prediction=predicted,
                    residual=np.where(valid, observed - predicted, np.nan), valid_mask=valid,
                    noise=observation["noise"].cpu().numpy(), flux_scale=model.flux_scale.item())
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **snapshot)
    temporary.replace(path)
    return snapshot


def create_snr_tracker(manifest):
    """Use exactly the results script's estimator and position prior."""
    from miri_results import expected_planet_position, measure_pyflip_snr

    planet, center, prior = expected_planet_position(manifest)
    settings = dict(signal_radius_arcsec=1.4, annulus_half_width_arcsec=.5,
                    smoothing_sigma_pixels=1., centroid_tolerance_pixels=1.)
    manifest["snr_tracking"] = {
        **settings, "expected_planet_xy": planet.tolist(),
        "coronagraph_center_xy": center.tolist(), "position_prior": prior,
        "snr_every": manifest["fit_settings"].get("snr_every", manifest["fit_settings"]["save_every"]),
        "measurement_timing": "after the saved optimizer update; also before target fitting",
        "metric": "pyflip Gaussian-fit signal / annular standard deviation",
    }

    def measure(snapshot):
        result, _, _ = measure_pyflip_snr(
            snapshot["residual"], snapshot["valid_mask"], planet, center,
            manifest["settings"]["pixel_scale_arcsec"], **settings,
        )
        return result

    return measure


def record_snr(tracker, snapshot, path, history, before_target_fitting=False):
    record = {"iteration": int(snapshot["iteration"]), "role": "target", "file": path.name,
              "before_target_fitting": before_target_fitting}
    try:
        record.update(tracker(snapshot))
        # Nonfinite fit statistics must never masquerade as a valid result.
        json.dumps(record, allow_nan=False)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        record = {"iteration": int(snapshot["iteration"]), "role": "target", "file": path.name,
                  "before_target_fitting": before_target_fitting, "error": str(error),
                  "raw_pyflip_snr": None, "centroid_consistent_snr": None,
                  "centroid_consistent_with_prior": False}
    text = json.dumps(record, allow_nan=False)
    result_path = path.with_name(path.stem + "_snr.json")
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(text + "\n")
    temporary.replace(result_path)
    history.write(text + "\n")
    history.flush()
    return record


def save_checkpoint(path, model, optimizer, cfg, next_iteration, signature):
    state = {name: getattr(model, name).detach().clone() for name in PARAMETERS}
    state["flux_scale"] = model.flux_scale.detach().clone()
    temporary = path.with_suffix(".tmp")
    torch.save({"state": state, "optimizer": optimizer.state_dict(), "config": cfg,
                "next_iteration": next_iteration, "signature": signature}, temporary)
    temporary.replace(path)


def run_fit(model, observations, cfg, output, signature, resume=None, max_updates=None, snr_tracker=None):
    validate_config(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    snr_history_path = output / "snr_history.jsonl"
    optimizer = make_optimizer(model, cfg)
    first = 0
    if resume is not None:
        saved = torch.load(resume, map_location=model.flux_scale.device, weights_only=True)
        if saved["config"] != cfg or saved["signature"] != signature:
            raise ValueError("Checkpoint settings or input/code fingerprint differ.")
        with torch.no_grad():
            for name, value in saved["state"].items():
                getattr(model, name).copy_(value)
        optimizer.load_state_dict(saved["optimizer"])
        first = saved["next_iteration"]
        # Remove history newer than the recovered checkpoint, if a run was interrupted.
        if history_path.exists():
            lines = [line for line in history_path.read_text().splitlines() if json.loads(line)["iteration"] < first]
            history_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        if snr_history_path.exists():
            records = [json.loads(line) for line in snr_history_path.read_text().splitlines() if line.strip()]
            records = [row for row in records if row["iteration"] < first and not (
                row.get("before_target_fitting", False) and first <= cfg["ref_cutoff_iter"]
            )]
            snr_history_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    else:
        if history_path.exists() or snr_history_path.exists() or (output / "latest.pt").exists():
            raise ValueError("Output already contains a run; use --resume or a new directory.")
        # MIRI's unit-power renderer needs an initial brightness scale.
        with torch.no_grad():
            match_flux(model, model(), observations["reference"], cfg["fluxwindsize"])
        save_snapshot(model, observations["reference"], output / "reference_initial.npz", -1)
    last = cfg["forced_stop_iter"]
    if max_updates is not None:
        last = min(last, first + max_updates - 1)
    if first > last:
        raise ValueError("No iterations remain in the requested range.")
    started = time.time()
    with history_path.open("a") as history, snr_history_path.open("a") as snr_history:
        for iteration in range(first, last + 1):
            role, flux_position = stage_at(iteration, cfg)
            observation = observations[role]
            if iteration == cfg["ref_cutoff_iter"]:
                path = output / "target_initial.npz"
                snapshot = save_snapshot(model, observation, path, iteration - 1)
                if snr_tracker is not None:
                    record_snr(snr_tracker, snapshot, path, snr_history, before_target_fitting=True)
            set_learning_rates(optimizer, iteration, cfg)
            optimizer.zero_grad(set_to_none=True)
            prediction = model()
            data_loss = fitting_loss(prediction, observation, role, cfg)
            multiplier = cfg["sci_lr_weight"] if role == "target" and not flux_position else 1.
            loss = multiplier * data_loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at iteration {iteration}.")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Nonfinite gradient: {name}, iteration {iteration}.")
            # Updating this buffer AFTER backward avoids invalidating autograd's saved values.
            if flux_position:
                match_flux(model, prediction, observation, cfg["fluxwindsize"])
            apply_stage_rules(model, iteration, cfg)
            optimizer.step()
            valid = observation["valid_mask"]
            rms = (prediction.detach()[valid] - observation["image"][valid]).square().mean().sqrt().item()
            record = {"iteration": iteration, "role": role, "flux_position_stage": flux_position,
                      "data_loss_before_update": data_loss.item(), "loss_before_update": loss.item(),
                      "rms_before_update": rms, "flux_scale_after_update": model.flux_scale.item(),
                      "source_offset_arcsec_after_update": model.source_offset_arcsec.detach().cpu().tolist(),
                      "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups}}
            history.write(json.dumps(record) + "\n")
            history.flush()
            save_now, snr_due, milestone = save_plan(iteration, last, cfg, snr_tracker is not None)
            if save_now:
                path = output / f"{role}_{iteration:06d}.npz"
                snapshot = save_snapshot(model, observation, path, iteration)
                snr_record = record_snr(snr_tracker, snapshot, path, snr_history) if snr_due else None
                if milestone:
                    save_checkpoint(output / f"checkpoint_{iteration:06d}.pt", model, optimizer, cfg, iteration + 1, signature)
                save_checkpoint(output / "latest.pt", model, optimizer, cfg, iteration + 1, signature)
                if iteration == cfg["ref_cutoff_iter"] - 1:
                    save_checkpoint(output / "reference_final.pt", model, optimizer, cfg, iteration + 1, signature)
                snr_text = ""
                if snr_record is not None:
                    value = snr_record.get("raw_pyflip_snr")
                    snr_text = (f" SNR={value:.4g} centroid_ok={snr_record['centroid_consistent_with_prior']}"
                                if value is not None else " SNR unavailable; see saved diagnostic")
                print(f"{iteration:5d} {role:9s} data loss={data_loss.item():.6g} RMS={rms:.6g}{snr_text}", flush=True)
            del prediction, loss, data_loss
    summary = {"first_iteration_this_run": first, "last_iteration": last,
               "completed": last == cfg["forced_stop_iter"], "seconds_this_run": time.time() - started,
               "freeze_behavior": "zero gradients, preserving upstream Adam momentum behavior",
               "injection_strength": 0, "planet_exclusion": False,
               "snr_tracking_enabled": snr_tracker is not None,
               "snr_history": "snr_history.jsonl",
               "milestone_iterations": cfg.get("save_iterations", [])}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def input_signature(inputs, observations, cfg):
    digest = hashlib.sha256(json.dumps({"settings": inputs["settings"], "fit": cfg}, sort_keys=True).encode())
    for name in ("primary_transmission", "telescope_opd", "lyot_transmission", "instrument_transmission", "instrument_opds", "wavelengths", "weights"):
        digest.update(np.ascontiguousarray(inputs[name]).tobytes())
    for path in inputs["focal_mask_files"]:
        digest.update(path.read_bytes())
    for role in ("reference", "target"):
        for name in ("image", "noise", "valid_mask"):
            digest.update(np.ascontiguousarray(observations[role][name]).tobytes())
    digest.update(np.asarray(observations["registration_offset_xy_pixels"]).tobytes())
    for name in ("run_miri_demo.py", "miri_model.py", "miri_propagation.py", "miri_detector.py", "miri_model_inputs.py", "miri_observations.py", "miri_wavefront.py", "miri_results.py"):
        digest.update((PROJECT_DIR / name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "miri_fit_settings.json")
    parser.add_argument("--settings", type=Path, default=PROJECT_DIR / "miri_settings.json", help="Optical settings; selects the matching prepared input set.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a trusted checkpoint created by this script.")
    parser.add_argument("--max-updates", type=int, help="Optional short run; does not alter the stage schedule.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run on an allocated GPU node, or explicitly select --device cpu.")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max-updates must be positive.")
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    observations = load_miri_observations(settings_path=args.settings)
    inputs = load_miri_inputs("reference", settings_path=args.settings)
    target_inputs = load_miri_inputs("target", settings_path=args.settings)
    if not np.array_equal(inputs["telescope_opd"], target_inputs["telescope_opd"]):
        raise ValueError("This shared-baseline demo expects identical initial target/reference telescope OPDs.")
    signature = input_signature(inputs, observations, cfg)
    output = args.output_dir
    if args.resume is not None:
        if output is not None and output.resolve() != args.resume.resolve().parent:
            raise ValueError("Resume into the checkpoint's original output directory.")
        output = args.resume.resolve().parent
    if output is None:
        output = PROJECT_DIR / "outputs" / "optimization" / time.strftime("%Y%m%d_%H%M%S")
    manifest = {"settings": inputs["settings"], "fit_settings": cfg, "signature": signature,
                "observations": {role: {k: v for k, v in observations[role].items() if not isinstance(v, np.ndarray)} for role in ("reference", "target")},
                "registration_offset_xy_pixels": observations["registration_offset_xy_pixels"].tolist(),
                "initial_flux_method": "central-window matching before first update; replaces NIRCam-specific photon scaling",
                "pointing_units": "entrance-pupil arcseconds; Adam LR and epsilon converted from original radians"}
    # Check the SNR dependency and geometry before starting any optimization.
    snr_tracker = create_snr_tracker(manifest)
    model, _ = build_miri_model("reference", args.device, train_flat_field=cfg["fit_flat_field"], settings_path=args.settings)
    tensors = {}
    for role in ("reference", "target"):
        tensors[role] = {
            name: torch.as_tensor(observations[role][name], dtype=torch.bool if name == "valid_mask" else torch.float64, device=args.device)
            for name in ("image", "noise", "valid_mask")
        }
    output.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        if (output / "manifest.json").exists():
            raise ValueError("Output directory already contains a run.")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = run_fit(model, tensors, cfg, output, signature, args.resume, args.max_updates, snr_tracker)
    print(json.dumps(result, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
