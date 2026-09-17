# CI Denoise

Pretrained fluorescence denoising from OME-Zarr to OME-Zarr, for Bilayers/BIOMERO,
local Windows CUDA and Linux containers. The BIOMERO workflow performs pretrained
inference only; no training, Gradio or Jupyter interface is included.

## Visual benchmark

Choose **Visual Benchmark Gallery** in the launcher or `--benchmark MODE`:

| Mode | XY region | Z processing / gallery view |
|---|---|---|
| Off (`off`, default) | Normal workflow | Selected model/channels; OME-Zarr output |
| 2D Full (`2d-full`) | Full image | Middle Z-plane |
| 2D Crop (`2d-crop`) | Centre 512x512 | Middle Z-plane |
| 3D Full (`3d-full`) | Full image | All Z restored, then maximum projection |
| 3D Crop (`3d-crop`) | Centre 512x512 | All Z restored, then maximum projection |

Enabled modes write **one PNG gallery per image/HCS field/timepoint**, original plus
all six models, with all channels overlaid; no OME-Zarr output. Smaller images keep
available XY dimensions. The middle plane is `Z//2` (upper middle for even stacks).
2D inputs work in all modes as one-plane stacks. 3D restores **every Z-plane before
projection**, rather than denoising an input projection. Crop never shortens Z.
Most models remain plane-wise; Tribolium receives five neighboring Z planes with
reflection in every mode. UniFMIR normalization uses all Z in the selected XY region.
Projections can hide slice-specific artifacts. Filenames include the mode.

Panels share source channel colours and per-channel raw display percentiles 1/99.8,
fitted to the raw plane or maximum projection. No per-model brightness matching.
Model selection, channel selection and output dtype are ignored for galleries.
Each panel gives processing seconds and time divided by the fastest successful model
(1.0x). Times include all requested planes/channels, reads, normalization, inference
and projection, excluding checkpoint/prompt loading and PNG rendering. This is a
single local measurement, not repeated benchmarking. Full/3D can be much slower.
PNG metadata records mode, Z indices, crop, settings, checkpoint hashes and display
ranges. Failed models are marked and the run returns failure. Existing PNGs are
never overwritten. A legacy bare `--benchmark` means `2d-crop`; old saved checkbox
settings migrate to `off` / `2d-crop`.

### Advanced settings and model applicability

Basic order: **Channels, Pretrained Model, Visual Benchmark Gallery**.
Compute Device is under Advanced. Most advanced options apply to all models:

| Advanced setting | Applies to / purpose |
|---|---|
| Compute Device | All: auto selects CUDA when available, otherwise CPU; explicit CUDA fails if unavailable. |
| Tile size | All: XY patch size, larger uses more GPU memory. 0 selects model preset. At least 64 and divisible by 8; Noise2Noise requires multiples of 32. |
| Tile overlap | All: shared margins blended to reduce seams; more overlap increases compute. -1 selects model preset. |
| Tile batch size | All: tiles inferred together; larger batches require more GPU memory. 0 selects model preset. |
| Output data type | All, ordinary OME-Zarr only: source rounds/clips; float32 retains floating values. Ignored by galleries. |
| Inference precision | All: auto uses FP16 for FluoResFM CUDA, FP32 otherwise. Explicit FP16 requires CUDA; FP32 avoids reduced-precision effects. |
| Channel 1-8 structure / --structures | FluoResFM only: optional biological prompt; task-only adds no structure description. Ignored by UniFMIR, Noise2Noise and Cellpose. |

Preset tile/overlap/batch: FluoResFM 64/16/16; UniFMIR 64/16/4; Noise2Noise 512/128/2;
Cellpose 224/64/8. Auto presets resolve separately per gallery model. Manual overrides
apply to every model and must be compatible with all six. No option trains models
or silently substitutes another checkpoint.

## Quick start on this workstation

The `cidenoise` Conda environment lives at
`C:\Users\p000881\AppData\Local\miniconda3\envs\cidenoise`.
Open this folder in VS Code and run `launch.cmd` for the PyQt launcher.
It exposes **Run Locally**, **Run Docker**, settings files and a live execution log.

For a new installation, run `create_env.cmd`. It installs the pinned requirements,
downloads verified pretrained assets and tests all six checkpoints on CUDA.
`requirements-lock-windows.txt` records and constrains the tested transitive versions.
The setup download includes a roughly 4.3 GB FluoResFM release archive; extracted
runtime assets occupy roughly 3.6 GB. Downloading is separate from inference.

```powershell
& "$env:LOCALAPPDATA\miniconda3\envs\cidenoise\python.exe" wrapper.py `
  --infolder inputfolder --outfolder outputfolder --model fluoresfm --device cuda
```

Place only intended inference inputs in `inputfolder`. The normal workflow processes
every top-level `.ome.zarr`. Visual benchmark modes process each input independently.

## Models and parameters

| ID | Input context | Normalization |
|---|---|---|
| `fluoresfm` | One XY plane from one channel | Per-plane 3rd/99.5th percentiles, nonnegative input, task-only prompt by default |
| `noise2noise-fmd` (default) | One XY plane; fast microscopy CNN | Per-plane maximum; input x/max - 0.5, inverse (y+0.5)*max |
| `cellpose-cyto3` / `cellpose-nuclei` | One XY plane; segmentation-oriented denoising | Per-plane 1st/99th percentiles; native pixel scale |
| `unifmir-planaria` | One XY plane from one channel | Per-channel Z-stack 2nd/99.8th percentiles |
| `unifmir-tribolium` | Five neighboring Z planes from one channel | Same stack normalization; reflected Z boundaries |

These are pretrained models from other specimen domains. Their names describe training
data, not detected specimen classes. No model is silently substituted on failure.

Basic parameters are `--channels all` (or one-based `1,3`), `--model`, and
`--benchmark`. Compute device (`--device auto|cuda|cpu`) is advanced. The launcher has eight channel structure menus;
`--structure-1 nuclei` through `--structure-8` provide the same choices in the CLI.
Task-only is the default. Legacy `--structures` JSON remains available in the CLI.
Descriptions affect FluoResFM only and are never inferred from channel colors.

Advanced defaults `--tile-size 0 --overlap -1 --batch-size 0` resolve to model-specific
presets for a 12 GB GPU / 16 GB RAM workstation. Explicit values override presets.
`--precision auto` uses FP16 for FluoResFM on CUDA and FP32 otherwise; use
`--precision float32` for the reference path. Output dtype remains source by default.
Compilation remains disabled. See [performance and presets](docs/performance.md).

For quick visual comparisons, select **2D Crop** in the launcher or run:

```powershell
python wrapper.py --infolder inputfolder --outfolder outputfolder --benchmark 2d-crop
```

Model assets and tokenizer files are verified and loaded from `models/`, or from
`CIDENOISE_MODELS`. Missing/corrupt assets fail with an actionable error. Runtime sets
Hugging Face and Transformers to offline mode; no model weights are fetched by a job.

## OME-Zarr contract

Input: NGFF 0.4, Zarr v2 ordinary images or HCS plates. Output:
`<source>__cidenoise.ome.zarr`, containing denoised intensities in the same layout.
Axis order, dimensions, scales/translations, channel names/colors, well/field hierarchy
and inherited labels are retained. Labels are not recomputed. Timepoints and fluorescence
channels are independent; channels are never interpreted as Z context.

Selected channels are denoised; others are copied unchanged. Source dtype is the default,
with integer rounding/clipping. Float32 affects the entire intensity array, including
unselected channels, because Zarr arrays have one dtype. Input normalization is reversed,
but neural restoration is **not an intensity-conserving operation**.

Only the current spatial context and output plane are held in memory, not whole plates
or time series. Integer stacks up to 16 bits use exact histogram percentiles. Float and
wider-integer stacks use a deterministic sample of at most approximately one million
pixels, recorded in provenance. Existing XY pyramids are regenerated for selected
channels using area averaging. Z-downsampled pyramids, multiple multiscale series in one
image group, unknown axes and other NGFF/Zarr versions are rejected explicitly in v1.

Each store is written transactionally through a unique temporary sibling. Existing
outputs are rejected; original input pixels are never modified. An ordinary exception
removes the temporary output. A force-killed process can leave a hidden `.partial`
directory, which is never treated as a completed result.

The root `cidenoise` attributes record checkpoint hashes, software versions, normalization
per plane, prompts, timings, clipping and GPU memory. Per-store elapsed time excludes
initial model loading, which is recorded separately under model provenance.

## Docker and BIOMERO

Run `builddocker.cmd` to build `w_cidenoise:v0.1.0`. Content-addressed model and runtime
layers are reused. `config.yaml` declares the Bilayers parameters and registry identity
`cellularimagingcf/w_cidenoise:v0.1.0`; the launcher uses the local image name.
`requirements-lock-linux.txt` constrains the tested Linux inference dependencies.

```powershell
docker run --rm --gpus all --network none `
  -v "${PWD}/inputfolder:/data/in:ro" -v "${PWD}/outputfolder:/data/out" `
  w_cidenoise:v0.1.0 --model fluoresfm --device cuda
```

`pushdocker.cmd --skip-build --dry-run` and `release_github.cmd --dry-run` inspect
publication actions. Publishing and creating a GitHub release are explicit maintainer
operations; implementation and testing do not publish anything. The release script
expects a clean, synchronized repository under the configured organization.

For an existing local BIOMERO deployment, `tools/biomero_import_smoke.py` tests a small
output store through both direct OMERO and BIOMERO import paths. It uses credentials
already present inside the importer container and a site-installed metadata probe.
Only dedicated temporary probe targets/imports are removed. This verifies import
compatibility; it is distinct from registering and scheduling the new workflow in Slurm.

## Verification and development

Run `test.cmd` for synthetic I/O and error-path tests, `tools/cuda_smoke.py` for actual
checkpoint inference, and `tools/verify_models.py` for upstream numerical comparisons
(the latter needs the pinned `.cache/upstream` checkouts). Source provenance and
licensing are documented in [THIRD_PARTY.md](THIRD_PARTY.md).

The initial workstation validation passed 23 Windows contract tests and the Linux
GitHub CI suite (the Windows rename test is skipped on Linux). All three checkpoints
passed CPU and CUDA inference checks, plus CUDA checks in Docker with networking
disabled. Adapted UniFMIR forward passes matched upstream exactly on the test patches;
FluoResFM's SDPA path differed by at most `9.93e-5` in normalized units. A small image
and HCS fixture produced pixel-identical local/container UniFMIR results. Direct OMERO
and BIOMERO import probes passed for both fixture types, including first-plane pixel
readback. These import checks do not establish Slurm workflow registration.

See [docs/candidates.md](docs/candidates.md) for deferred denoising methods.

The repository includes a [workflow skill](skills/use-cidenoise-workflow/SKILL.md)
for configuring, running, monitoring and troubleshooting CIDenoise, with parameter
and recovery references. It is scoped to the workflow; there is no measurement-data
analysis skill.
