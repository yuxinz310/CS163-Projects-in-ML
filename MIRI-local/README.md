# MIRI pyflip demo

Adaptation of [pyflip](https://github.com/rferrerc/pyflip/tree/submission) for
HIP 65426 observations in **F1140C** and **F1550C**. The model fits the reference
star first, then the target star. No artificial planet is injected.

## Install

Requires Linux, Python 3.11, and an NVIDIA GPU compatible with the pinned CUDA
11.8 PyTorch build. Internet access is needed for installation and initial setup.

```bash
git clone --branch yuxin https://github.com/yuxinz310/CS163-Projects-in-ML.git
cd CS163-Projects-in-ML/MIRI-local
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Original observations are included. Calibration downloads and optical-input
preparation run automatically on first use; matching inputs are reused later.

## Run

Example—change the filter, wavelength count, sampling, iterations, and learning
rate as needed:

```bash
python run_miri.py --filter F1140C --wavelengths 5 --oversample 2 --iterations 15000 --lr 4e-9
```

| Option | Purpose |
| --- | --- |
| `--filter F1140C` or `F1550C` | Select the observation filter. |
| `--wavelengths N` | Number of wavelength samples; required. |
| `--oversample N` | Set both FFT and detector sampling; default 4. |
| `--iterations N` | Final iteration label; default 15000. |
| `--lr VALUE` | Base learning rate; default `4e-9`. |
| `--save-every N`, `--snr-every N` | Snapshot and SNR intervals; default 100. |
| `--output-dir PATH` | Use a new results folder; otherwise one is timestamped automatically. |
| `--print-config` | Preview settings without downloading or running. |
| `--prepare-only` | Prepare inputs without fitting. |

For shorter runs, add `--schedule proportional` to scale the fitting stages.
Separate sampling controls are `--fft-oversample` and `--detector-oversample`.
Use `python run_miri.py --help` for all options, including `--resume CHECKPOINT`.
Resuming requires the same inputs, code, and fitting settings.

## Results

Each run saves to `outputs/optimization/<run-directory>/`:

- `manifest.json`, `summary.json`: settings and completion status.
- `history.jsonl`, `snr_history.jsonl`: loss and SNR histories.
- `reference_*.npz`, `target_*.npz`: observed images, predictions, and residuals.
- `latest.pt`, `checkpoint_*.pt`: resumable checkpoints, including 11000 and 15000 with the defaults.
- `analysis/`: residual plots, loss/SNR plots, SNR report, and an MP4 video.

The default schedule fits the reference at iterations 0–5499 and the target at
5500–15000; labels are inclusive (15001 total updates). Five-wavelength, 2× runs
took about **23 minutes per filter on an H200**, excluding preparation and plots.

## Notes

- Each run uses the first target exposure and first reference exposure, averaged into one image each.
- Displays retain detector orientation and use `inferno`, fixed at **−1 to 2 MJy/sr**.
- SNR is a fitted peak divided by annular pixel scatter, with a position check. It is not a calibrated detection significance; the final iteration may not give the best planet recovery.
- Keep the original `data/` folder. Downloads, generated inputs, environments, and outputs are ignored by Git.
