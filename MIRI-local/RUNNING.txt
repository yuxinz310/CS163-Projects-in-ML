MIRI pyflip demo: HIP 65426
==========================

Local adaptation of pyflip for the supplied JWST/MIRI F1140C and F1550C
observations. The model first fits the reference star (HIP 68245), then fits
the science target (HIP 65426). This demo does not inject an artificial planet.

Original pyflip and documentation reference:
https://github.com/rferrerc/pyflip/tree/submission

1. System requirements
----------------------

Operating environment:
- Linux and Python 3.11.
- A compatible NVIDIA GPU and driver for the pinned CUDA 11.8 PyTorch build.
- Internet access for installation and the first calibration download.
- Disk space for observations, calibration, generated optical arrays and outputs.

The requirements.txt file pins these direct dependencies:
- torch 2.4.0+cu118 and torchvision 0.19.0+cu118
- numpy 2.1.0 and scipy 1.17.1
- astropy 8.0.2
- stpsf 2.2.0 and poppy 1.1.2
- synphot 1.8.0, stsynphot 1.5.1 and pysiaf 0.29.0
- pyklip 2.8
- matplotlib 3.11.2 and imageio-ffmpeg 0.6.0

Full F1140C and F1550C fits were tested on an NVIDIA H200. F1550C optical
validation and a short two-stage fit were tested on a Tesla P100 with 16 GB GPU memory.
This does not establish a minimum memory requirement for every configuration.
CPU execution is available using --device cpu, but full fits can be much slower.
The GPU execution path uses CUDA, not Apple MPS or AMD ROCm.

For hardware needing a different PyTorch build, adjust the torch/torchvision
pins and CUDA index together using https://pytorch.org/get-started/locally/ .
Different versions can change numerical results. Transitive dependencies are
not fully locked by this requirements file.

2. Installation guide
---------------------

Clone the repository and create a fresh environment:

  git clone --branch yuxin https://github.com/yuxinz310/CS163-Projects-in-ML.git
  cd CS163-Projects-in-ML/MIRI-local
  python3.11 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip check

Check GPU access:

  python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

CUDA should report True for GPU execution. No HPC account or Slurm script is
needed on a local GPU machine. Installation time depends on download speed and
package availability; a fresh-machine installation time has not been measured.

3. Demo
-------

Run the demo from MIRI-local, with the environment activated. Choose your own
parameters using this command template. Replace the uppercase placeholders
with values before running it:

  python run_miri.py --filter FILTER --wavelengths N --oversample S --iterations I --lr LR

  FILTER  F1140C or F1550C, the two supplied observation filters.
  N       Positive integer: number of wavelength samples within that filter.
  S       Positive integer: both FFT and detector oversampling factors.
  I       Positive integer: final iteration label and learning-rate horizon.
  LR      Positive finite number: base learning rate, in decimal or scientific notation.

You are not restricted to the wavelength count, sampling or learning rate used
in our earlier runs. Filter selection remains limited to the two supported
filters. Larger sampling and wavelength counts need more compute and memory.

For separate FFT and detector sampling, use:

  python run_miri.py --filter FILTER --wavelengths N --fft-oversample FFT --detector-oversample DET --iterations I --lr LR

Replace FFT and DET with positive integers. Add any of these controls as needed:

  --save-every K             Save snapshots every K iterations.
  --snr-every K              Measure target SNR every K iterations.
  --schedule proportional   Scale fitting-stage boundaries to your chosen I.
  --output-dir PATH         Save this experiment in a new directory of your choice.
  --print-config            Preview resolved settings without starting the run.

By default, fitting-stage boundaries remain fixed. For a shorter training
schedule, use --schedule proportional or provide a compatible --fit-config;
the launcher rejects iteration limits that conflict with the fitting stages.

If --output-dir is omitted, each run gets a timestamped directory under
outputs/optimization/. You can repeat the command with different parameter
values without editing the Python source or the base JSON files.

The launcher automatically:
1. Downloads and verifies calibration files listed in calibration_sources.json.
2. Reads the selected filter's original observations and prepares metadata.
3. Generates telescope/instrument OPDs, wavelengths, weights and optical masks.
4. Fits the reference star, then the science target.
5. Saves results, calculates SNR, and generates plots and a residual video.

The first listed target and reference exposures in the supplied data config
are selected. Their prepared COADDALL images, uncertainty maps and pixel masks
are loaded from the data folder. Calibration and generated optical inputs are
cached locally and reused for matching settings.

With the supplied default settings (before user overrides):
- Reference fitting: iteration labels 0 through 5499.
- Target fitting: labels 5500 through 15000.
- Image snapshots every 100 iterations, with additional stage-boundary saves.
- Target SNR every 100 iterations; loss is recorded every iteration.
- Dedicated checkpoints at 11000 and 15000.

Iteration numbering follows the existing pyflip convention: 0..N inclusive.
Therefore --iterations 15000 performs 15001 updates, including iteration zero.

Expected outputs

Results are written to your chosen output directory, or by default:

  outputs/optimization/<run-directory>/
    manifest.json                  Actual optical/fit settings and provenance
    summary.json                   Completion and iteration summary
    history.jsonl                  Loss and training history
    snr_history.jsonl              SNR measurements during fitting
    latest.pt                      Latest resumable checkpoint
    checkpoint_NNNNNN.pt           Dedicated milestone/final checkpoints
    reference_*.npz                Reference-star image snapshots
    target_*.npz                   Target-star image snapshots
    target_*_snr.json               SNR details for saved target snapshots
    analysis/
      target_summary.png           Final observation/model/residual comparison
      residual_evolution.png       Residuals at selected iterations
      optimization_history.png    Loss and SNR plots
      snr_report.json              Detailed SNR measurements
      residual_evolution.mp4       Video made from saved target snapshots

Checkpoint filenames use six-digit iteration labels. The default milestone at
11000 is retained when reachable, and the requested final iteration is saved.

Each snapshot NPZ includes the observed image, model prediction, residual,
valid-pixel mask and uncertainty map. The final snapshot is not necessarily
the iteration with the highest SNR; inspect the SNR history for that comparison.

Runtime

A previous five-wavelength F1140C run at 4x sampling reached iteration 15000
in about 31 minutes of optimization on an H200. Initial input preparation and
final plotting/video generation take additional time. Runtime varies with
hardware, wavelength count and sampling. Five-wavelength runs at 2x sampling
reached iteration 15000 in 22.8 minutes (F1140C) and 22.9 minutes (F1550C) of
optimization on an H200. CPU runs and first-time setup can take much longer.

4. Instructions for use
-----------------------

Common command-line options:

  --filter F1140C|F1550C    Select observations and coronagraph; default F1140C.
  --wavelengths N          Number of wavelength samples (required).
  --oversample N           Set both FFT and detector oversampling; default 4.
  --fft-oversample N       Override FFT oversampling separately.
  --detector-oversample N  Override detector oversampling separately.
  --iterations N          Final iteration label and learning-rate horizon.
  --lr VALUE              Base learning rate; default 4e-9 in the supplied JSON.
  --save-every N          Periodic snapshot/checkpoint interval; default 100.
  --snr-every N           Target SNR interval; default 100. Also saves a snapshot.
  --output-dir PATH       Choose a new output directory.
  --device cuda|cpu       Choose execution device; default cuda.

The base learning rate retains the existing parameter-group multipliers,
pointing-unit conversion and target-stage scaling. For further controls:

  python run_miri.py --help

Preview resolved settings without writing files or starting a run:

  python run_miri.py --filter F1550C --wavelengths 9 --oversample 4 --iterations 15000 --lr 4e-9 --print-config

Prepare and validate inputs without fitting or requiring a GPU:

  python run_miri.py --filter F1550C --wavelengths 5 --oversample 4 --prepare-only

Run a shorter experiment, explicitly scaling fitting-stage boundaries:

  python run_miri.py --filter F1140C --wavelengths 5 --oversample 4 --iterations 3000 --schedule proportional

By default, --schedule fixed preserves stage boundaries from the base fitting
JSON. Proportional scaling requires --iterations and scales stage boundaries
relative to the original schedule horizon; save/SNR intervals do not scale.
--reference-iterations can override the reference-to-target switch afterward.
Milestones beyond a requested final iteration are dropped and the new final
iteration is added. The default 11000 checkpoint remains whenever reachable.

For a quick execution check, --max-updates 2 stops after two updates without
changing the schedule. It does not test the later target-fitting stage.
--skip-analysis skips final plots/video but retains in-training SNR tracking.

Resume an interrupted run using the same sampling and fitting settings:

  python run_miri.py --filter F1550C --wavelengths 5 --oversample 4 --iterations 15000 --lr 4e-9 --resume outputs/optimization/F1550C_demo/latest.pt

Resume checks require matching inputs, settings and source signatures. Changing
the learning rate, schedule or implementation requires a new run rather than
bypassing these checks. For other fitting settings, provide --fit-config PATH.
The base JSON files are not overwritten; resolved settings are saved under
configs/generated/ and recorded in the run's manifest.json.

5. Data, calibration and model scope
-----------------------------------

Keep the original data/HIP65426-MIRI/ folder with the code. To use both filters,
keep its HIP65426-F1140C and HIP65426-F1550C subfolders. The launcher is configured
for this target/reference pair; a different star or filter needs an explicit
adaptation of observation selection, source spectrum and planet-position prior.

Calibration downloads use pinned URLs and SHA256 checksums. To run that setup
separately, or check existing downloads without network access:

  python setup_miri_calibration.py
  python setup_miri_calibration.py --offline

The shared STPSF package and PHOENIX A1V spectra are downloaded automatically.
The measured telescope OPD is pinned separately for each observation role and
filter in calibration_sources.json. The selected measurement is the same file
for these four observations, but telescope field dependence is evaluated at
each filter's configured coronagraph position.

Filter settings follow the supplied data:
- F1140C: FQPM1140, MASKFQPM, 0.1102 arcsec per detector pixel.
- F1550C: FQPM1550, MASKFQPM, 0.1101 arcsec per detector pixel.

Generated telescope maps are separated by filter and role. Other optical
inputs and run outputs are also separated by configuration. These local folders
are generated as needed and should not be included in new Git additions:

  .venv/                  Installed Python environment
  calibration/            Downloaded calibration
  model_inputs/           Generated optical arrays
  configs/generated/      Resolved run settings
  outputs/                Checkpoints, logs, plots and videos

The detector uses a fixed Gaussian approximation with sigma=0.28 detector
pixels. SNR uses a Gaussian-fitted planet peak divided by residual pixel scatter
in an annulus, with the pyflip-style measurement settings retained across both
filters. This is not a throughput-calibrated or small-sample-corrected detection
significance.

F1550C validation used five wavelengths and 4x FFT/detector sampling. Broadband
optical agreement with STPSF had relative error 1.73e-15; a seven-update two-stage
fit produced finite predictions, checkpoints, SNR records, plots and video.
These checks validate baseline propagation and execution, not convergence of a
full F1550C optimization or a measured physical detector response.

Full 15000-iteration runs at five wavelengths and 2x sampling also completed
for both filters. Completion does not establish successful planet recovery:
the highest saved SNR statistic passing the position/positive-fit checks was
17.69 at iteration 6600 for F1140C and 1.62 at iteration 5500 for F1550C.
Both final snapshots failed those checks. Inspect the full SNR history rather
than assuming the last iteration is the best subtraction.

Plot and video coordinates

Images are displayed directly in detector crop coordinates: x increases to the
right and y increases upward. No north-up/east-left display transformation is
applied. Planet arrows use those same crop coordinates. The optical model still
includes the STPSF pupil inversion, FQPM alignment and instrument-frame rotation;
these are propagation operations. FITS orientation is still used to convert the
published planet position into detector coordinates for SNR measurements.

Image plots and residual videos use the inferno color map with fixed linear
limits of -1 to 2 MJy/sr, matching the MIRI F1140C row in Figure 3 (page 8) of
https://doi.org/10.3847/2041-8213/acd93e . The same display limits are used for
F1550C. Values outside this range saturate visually; saved arrays and SNR
calculations are unchanged.
