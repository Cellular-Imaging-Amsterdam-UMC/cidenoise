# Confocal-only Noise2Noise training

## Installed confocal checkpoint (2026-09-17)

The completed 100-epoch run selected epoch **75** (`best.pt`), validation noisy-target
MSE **0.0009274739**. Mean validation MSE was 0.0009282375 at epochs 81–90 and
0.0009282186 at epochs 91–100 (0.00203% improvement): the current schedule plateaued.
This supports using the best checkpoint without extending this run; it does not
prove an optimal architecture, schedule or performance on other specimens.

Select **Noise2Noise confocal only (experimental)** in the launcher, or pass
`--model noise2noise-confocal` to `wrapper.py`. The published mixed-modality FMD model
remains a separate choice. Resource defaults: tile 512, overlap 128, batch 2, FP32.

The local installation copies the original checkpoint unchanged to
`models/confocal/best.pt`. SHA256:
`a0c009dc9e5b863ee8a506b784e277901808e5aa84ccbe607701cff575dfd5c8`.
On another machine, copy that exact file into the model cache at `confocal/best.pt`
(or the corresponding path under `CIDENOISE_MODELS`). Weights are not in Git and
have no public download URL. Model-cache Docker builds include the file when present;
existing containers need rebuilding before this choice is available in Docker.

Training used fixed uint8/255 scaling, retained exactly for uint8 inference.
Uint16 uses its full 65535 range, and float data must already be in [0,1].
Unsigned 12-bit acquisition stored in uint16 is therefore **not** automatically
rescaled to its acquisition range. Such intensity/noise differences from the training
data need review; no per-plane maximum normalization is silently introduced.
Output conversion reverses scaling but does not enforce intensity conservation.

**Brain-data limitation:** the small brain1/brain2 review found a strong positive
background/intensity bias with full-range uint16 scaling: channel mean increases
of approximately 152–156 raw intensity units. The checkpoint is available for
experimentation but is not recommended for quantitative brain analysis in this
configuration. More epochs of the same schedule are not supported by the plateau;
acquisition-range/noise-domain adaptation needs a separate evaluation.
On the fixed 256 held-out FMD patches, noisy-target MSE was 0.001594864 versus
0.002783081 for raw pairs. These are noisy-reference metrics, not clean ground truth.

Reproduce convergence assessment and a fixed held-out FOV19 test using
`pip install -r training/requirements-review.txt` then
`python tools/assess_confocal_training.py`. Local outputs are under
`outputs/confocal-training-review/`; small brain comparisons are under
`outputs/benchmark-confocal/report/`. No test or brain image was used to select
the epoch-75 checkpoint.

This optional local trainer is separate from the pretrained BIOMERO inference job.
It uses the existing `cidenoise` Conda environment, Windows CUDA and the small
Noise2Noise U-Net already ported into this repository. No TensorFlow, notebook or
training of FluoResFM is required. It starts from random weights, so its initialization
does not include widefield or two-photon training data.

## Start on this workstation

The data has been prepared under `F:\noise2noise_data`. Double-click
`F:\noise2noise_data\start_training.cmd`, or run from the CIDenoise checkout:

```powershell
.\train_noise2noise.cmd
```

Default run directory: `F:\noise2noise_data\runs\noise2noise_confocal`.
The trainer refuses to overwrite an existing run. To continue that run:

```powershell
.\train_noise2noise.cmd --resume F:\noise2noise_data\runs\noise2noise_confocal\last.pt
```

The resume script in the data folder does the same. If you changed training options
for the original run, supply those same options when resuming; incompatible changes
are rejected. A saved checkpoint restores model, optimizer, learning-rate schedule,
AMP scaler, epoch, step and deterministic sample positions. Work after the last saved
checkpoint is repeated after an interruption. Numerical bitwise reproducibility across
different hardware/software is not guaranteed.

For a run whose loss becomes unstable, explicitly lower the learning rate on resume:

```powershell
.\train_noise2noise.cmd --resume F:\noise2noise_data\runs\noise2noise_confocal\last.pt --learning-rate 0.0001 --reset-learning-rate
```

This rescales the saved cosine schedule while preserving its position and Adam's
moments. It records the new base learning rate in subsequent checkpoints. Reusing
the command is safe: if the checkpoint already uses 0.0001, the scale factor is 1.
The local `resume_training_recovery.cmd` runs this command. Continue using it after
recovery, since ordinary resume requires the checkpoint's learning-rate setting.
The original 0.001 run showed instability at epoch 15; lowering the rate is an
explicit recovery adjustment, not a guarantee of convergence or denoising quality.

For a short trial with the same full-run learning-rate schedule:

```powershell
.\train_noise2noise.cmd --output F:\noise2noise_data\runs\trial --stop-after-steps 20
.\train_noise2noise.cmd --output F:\noise2noise_data\runs\trial --resume F:\noise2noise_data\runs\trial\last.pt
```

Preparation on a new machine, or after an interrupted download:

```powershell
.\prepare_noise2noise.cmd
```

Archives resume when the server supports byte ranges; a server returning a full file
causes a safe restart of that partial download. Completed archives are verified against
the publisher's MD5 checksums. Extracted PNGs are independently pinned with SHA-256 in
`index.json`, and checked before every training run. No data is downloaded during
training. `--prepare-only` verifies/indexes data without starting optimization.

## Data actually used

[FluoResFM's dataset table](https://github.com/qiqi-lu/fluoresfm#data-collection)
links to the [FMD dataset](https://github.com/yinhaoz/denoising-fluorescence).
The [author-hosted archival record](https://curate.nd.edu/articles/dataset/Fluorescence_Microscopy_Denoising_FMD_dataset/24744648)
provides separately downloadable microscope-specific archives and checksums.

Only these five archives are downloaded (3,443,189,760 bytes total):

| Category | Structures/specimen | Raw captures |
|---|---|---:|
| Confocal_BPAE_B | BPAE nuclei | 1,000 |
| Confocal_BPAE_G | BPAE actin | 1,000 |
| Confocal_BPAE_R | BPAE mitochondria | 1,000 |
| Confocal_FISH | Zebrafish | 1,000 |
| Confocal_MICE | Mouse brain tissue | 1,000 |

Each category has 20 fields and 50 repeated 512x512, 8-bit grayscale captures per
field. This is 5,000 images, **not 5,000 independent specimens**. The corresponding
100 averaged reference images are retained for inspection but never used as training
targets or for selecting checkpoints. Precomputed `avg2/4/8/16` exports are not extracted;
their possibly overlapping constituent frames must not be mistaken for independent pairs.
The mixed-modality test archive is deliberately excluded.

The source metadata, URLs, checksums, DOI and CC BY-SA 4.0 dataset license are preserved
in `sources.json` and `training/confocal_sources.json`. Cite Zhang et al., *A
Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images*, CVPR 2019,
and dataset DOI [10.7274/r0-ed2r-4052](https://doi.org/10.7274/r0-ed2r-4052).
The model architecture derives from Mannam's GPL-3.0 Instant Image Denoising implementation;
its notices remain in `cidenoise/vendor/instant/LICENSE` and `THIRD_PARTY.md`.

Other confocal data linked by FluoResFM are not interchangeable Noise2Noise targets:

| Source | Reason excluded from this trainer |
|---|---|
| CARE Planaria | Published low/high-exposure training pairs; not a confirmed same-exposure independent-repeat collection. Suitable for a separate supervised setup. |
| DeepBacs | Leica SP8 low/high-SNR pairs use different laser power/averaging; live temporal frames can change biologically. |
| SR-CACO-2 | Acquisitions at different spatial resolutions; not matched noisy observations of the same sampling grid. |
| RCAN confocal/STED | Cross-resolution restoration targets, rather than independent confocal noisy repeats. |
| Confocal segmentation collections | Annotated single images do not by themselves supply Noise2Noise pairs. |

The processed FluoResFM training collection is available by author request, not as one
public confocal download. This preparation uses the eligible public source data; it
does not claim to reproduce FluoResFM's full training set.

## Sampling and split

- Training: FOV 1–18 in every category (90 category/FOV groups, 4,500 raw captures).
- Validation: FOV 20 in every category (5 groups, 250 captures).
- Test, unused by optimization or model selection: FOV 19 (5 groups, 250 captures).

The same field numbers are held out across all BPAE channels. Cropping first and then
randomly splitting patches would leak structures; this trainer splits fields first.
brain1, brain2 and their LAS-X references are not included anywhere in the training data.

Each sample selects one field, then disjoint groups of 1, 2 or 4 raw captures. Each
group is averaged independently to provide several noise levels without sharing a
raw capture between input and target. Use `--averages 1` for single-capture pairs only;
counts 8 and 16 are also supported explicitly. Sampling balances the five categories.
Input and target use identical XY crops, rotations and flips. Channels, Z planes and
changing live frames are never used as noisy targets for one another.

Normalization is fixed and shared: `uint8 / 255 - 0.5`, as in the published microscopy
Noise2Noise training recipe. Target-dependent percentile/max normalization is avoided.
MSE training assumes repeated observations have the same underlying signal and
independent noise; drift, bleaching, clipping and fixed-pattern noise can violate
that approximation. Check biological preservation on held-out data after training.

## Defaults and outputs

| Option | Default |
|---|---|
| Epochs / steps per epoch | 100 / 1,000 |
| Patch / batch | 256 / 8 |
| Precision | CUDA FP16 autocast; FP32 loss and optimizer weights |
| CUDA allocator limit | Smaller of 9 GiB or 80% of the device |
| Data-loader workers | 2, bounded image cache and prefetch |
| Optimizer / learning rate | Adam / 0.001, cosine decay to 0.00001 |
| Validation | 32 fixed batches per epoch, from held-out fields |
| Checkpoint frequency | Every 250 steps and at each epoch end |

FP16 gradient overflow automatically reduces the loss scale and retries the same
batch (up to 16 retries). Failed attempts do not advance Adam, the learning-rate
schedule or sample position. Persistent overflow, nonfinite forward loss and
nonfinite float32 gradients still stop with an error. Existing checkpoints remain
compatible; after an interrupted run, use `resume_training.cmd` to continue from
the last saved step.

`last.pt` is the resumable checkpoint; `best.pt` is selected by fixed validation
**noisy-target MSE**, not clean-image PSNR. Both contain `model_state_dict` compatible
with `cidenoise.vendor.instant.Noise2Noise`. `run.json`, `training.log` and
`metrics.jsonl` record provenance and progress. Completed checkpoints are published by
atomic file replacement. No workflow pretrained checkpoint is overwritten or silently
replaced with a newly trained model.

For loading the resulting network in Python:

```python
import torch
from cidenoise.vendor.instant import Noise2Noise
checkpoint = torch.load("best.pt", map_location="cpu", weights_only=True)
model = Noise2Noise()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
```

Training on this dataset is an experiment, not a guarantee of matching Leica LAS-X.
The supplied four-step smoke checkpoint is only a code/gradient/checkpoint test; it is
not a useful trained denoiser. The full default training run has not been started.

Local validation on the RTX A5000: four CUDA optimization steps plus validation at
the default patch/batch used 1.72 GiB peak reserved GPU memory. A separate run paused
after step 2 and resumed through step 4 produced exactly the same model weights
(maximum absolute difference 0). These checks validate execution and resumption,
not denoising quality or convergence. Results are in `runs/training_validation.json`.
