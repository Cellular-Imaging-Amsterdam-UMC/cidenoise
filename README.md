# CI Denoise

Pretrained fluorescence denoising from OME-Zarr to OME-Zarr, for Bilayers/BIOMERO,
local Windows CUDA and Linux containers. No training, Gradio or Jupyter dependencies.

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
every top-level `.ome.zarr`, including names ending in `-dn`; the benchmark separately
and explicitly pairs raw/reference stores in `localdata`.

## Models and parameters

| ID | Input context | Normalization |
|---|---|---|
| `fluoresfm` (default) | One XY plane from one channel | Per-plane 3rd/99.5th percentiles, nonnegative input, task-only prompt by default |
| `noise2noise-fmd` | One XY plane; fast microscopy CNN | Per-plane maximum; input x/max - 0.5, inverse (y+0.5)*max |
| `cellpose-cyto3` / `cellpose-nuclei` | One XY plane; segmentation-oriented denoising | Per-plane 1st/99th percentiles; native pixel scale |
| `unifmir-planaria` | One XY plane from one channel | Per-channel Z-stack 2nd/99.8th percentiles |
| `unifmir-tribolium` | Five neighboring Z planes from one channel | Same stack normalization; reflected Z boundaries |

These are pretrained models from other specimen domains. Their names describe training
data, not detected specimen classes. No model is silently substituted on failure.

Basic parameters are `--model`, `--channels all` (or one-based `1,3`), and
`--device auto|cuda|cpu`. The launcher has eight channel structure menus;
`--structure-1 nuclei` through `--structure-8` provide the same choices in the CLI.
Task-only is the default. Legacy `--structures` JSON remains available in the CLI.
Descriptions affect FluoResFM only and are never inferred from channel colors.

Advanced defaults `--tile-size 0 --overlap -1 --batch-size 0` resolve to model-specific
presets for a 12 GB GPU / 16 GB RAM workstation. Explicit values override presets.
`--precision auto` uses FP16 for FluoResFM on CUDA and FP32 otherwise; use
`--precision float32` for the reference path. Output dtype remains source by default.
Compilation remains disabled. See [performance and presets](docs/performance.md).

For quick comparisons, run `python tools/make_benchmark_crops.py` once, then:

```powershell
python -m cidenoise.benchmark --localdata outputs/benchmark-small-inputs --output outputs/benchmark-small-fast
```

The four matched crop stores contain all channels: brain1 is 4x6x256x256 and brain2
4x4x256x256, each with its LAS-X reference. Raw inputs alone are denoised by this
benchmark. Spatial coordinates and source pixels are preserved; these are standalone
single-resolution NGFF stores. Their report is `outputs/benchmark-small-fast/report/index.html`.

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

## Local comparison with LAS-X

```powershell
& "$env:LOCALAPPDATA\miniconda3\envs\cidenoise\python.exe" -m cidenoise.benchmark
```

The benchmark explicitly uses `brain1`/`brain1-dn` and `brain2`/`brain2-dn`, with all
channels and Z planes. It writes full stores beneath `outputs/benchmark/<model>/` and
an HTML report at `outputs/benchmark/report/index.html`. Completed stores can be reused
on a resumed run after settings validation. Use a new output directory to change settings.

LAS-X is a processed comparison reference, not ground truth. The supplied LAS-X values
are often about ten times higher than the raw values. A per-channel reference-to-raw
affine mapping is fitted on brain1 and frozen for brain2. Alignment is checked before
similarity metrics. Raw intensity differences are reported separately from PSNR/SSIM
against the scaled reference. No automatic winner is selected.

Panels include central XY, a fixed central crop, XZ, maximum projection and signed
residuals. Display ranges are shared and fixed from brain1. XZ images are stretched for
inspection. Review weak puncta, fine processes, background texture and introduced or
removed structures; similarity alone cannot establish biological fidelity.
Optional manual observations saved as `report/review.html` are linked by the report
generator and retained when the figures and metrics are regenerated.

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
