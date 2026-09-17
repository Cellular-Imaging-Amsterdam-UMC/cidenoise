# CIDenoise parameters

Read the current `config.yaml` and `wrapper.py` before constructing a command. The
launcher and Bilayers descriptor use the same parameter contract.

| Setting | Default | Meaning |
|---|---|---|
| `--infolder` | `/data/in` | Top-level NGFF 0.4/Zarr v2 `.ome.zarr` images or HCS plates; set explicitly for local runs. |
| `--outfolder` | `/data/out` | Separate writable destination; existing output stores are rejected. |
| `--model` | `noise2noise-fmd` | Explicit checkpoint choice; alternatives: `fluoresfm`, `unifmir-planaria`, `unifmir-tribolium`, `cellpose-cyto3`, `cellpose-nuclei`. |
| `--benchmark` | off | off, 2d-full, 2d-crop, 3d-full, 3d-crop. Full keeps XY; crop uses centre 512x512. 2D restores middle Z; 3D restores all Z then maximum projection. One all-model PNG with times; no OME-Zarr. |
| `--channels` | `all` | One-based channel numbers, e.g. `1,3`; keep other channels unchanged. |
| `--device` | `auto` | CUDA when available, otherwise CPU; explicit `cuda` fails if unavailable. |
| `--tile-size` | `0` (model preset) | XY tile edge, at least 64 and divisible by 8. Larger tiles increase memory use. |
| `--overlap` | `-1` (model preset) | Blended XY overlap, at least zero and smaller than tile size. |
| `--batch-size` | `0` (model preset) | Positive number of tiles per inference batch. |
| `--output-dtype` | `source` | Original dtype with integer rounding/clipping, or `float32` for review. |
| `--structure-1` ... `--structure-8` | task-only | Eight optional channel descriptions; exposed as dropdown menus. |
| `--precision` | auto | FP16 FluoResFM on CUDA, FP32 otherwise; explicit float32/float16 supported. |
| `--structures` | `{}` | JSON map of one-based channels to user-supplied FluoResFM descriptions. |

FluoResFM predicts each XY plane independently with task-only denoising conditioning
unless descriptions are supplied. Its plane normalization uses the 3rd/99.5th
percentiles after restricting inputs to nonnegative values. Do not infer biological
structures from display colors. There is no user-selectable deconvolution,
super-resolution, training, or compilation mode.

UniFMIR Planaria uses one plane. Tribolium uses five neighboring Z planes from the
same channel with reflected boundaries. Both use per-channel stack 2nd/99.8th
percentiles, independently for each timepoint. Short and single-plane stacks are
supported through reflection. Do not substitute one checkpoint after another fails.

Use `--structures` as a single argument containing JSON, for example
`'{"1":"nuclei","4":"neuronal processes"}'` in PowerShell. It affects FluoResFM only.
The launcher uses eight structure menus and migrates previously saved JSON descriptions.

Local example, from the repository root:

```powershell
& "$env:LOCALAPPDATA\miniconda3\envs\cidenoise\python.exe" wrapper.py `
  --infolder inputfolder --outfolder outputfolder `
  --model fluoresfm --channels 1,3 --device cuda
```

For Docker, the launcher's GPU checkbox controls `--gpus all`; the `device` parameter
still controls inference device selection inside the container. The local image is
`w_cidenoise:<version.txt>`. The registry organization/name/tag are in `config.yaml`.

Resource presets (tile / overlap / batch): FluoResFM 64/16/16, UniFMIR 64/16/4,
Noise2Noise 512/128/2, Cellpose 224/64/8. CUDA allocator budget is at most 9 GiB
and at most 80% of detected device capacity. This excludes allocations outside PyTorch;
manual overrides can still exhaust memory. Noise2Noise tiles require multiples of 32.
Cellpose uses native pixel scale without diameter rescaling; review fine structures.

## Advanced applicability

Basic order: Channels, Pretrained Model, Visual Benchmark Gallery. Device is advanced.
Device, tile size, overlap, batch size and precision apply to all models. Larger
patches/batches use more memory; overlap reduces seams at extra compute cost.
Structure menus/JSON apply only to FluoResFM. Output dtype only affects ordinary
OME-Zarr, not galleries. Auto presets resolve per model; manual overrides apply to
all gallery models. FP16 requires CUDA; auto uses it for FluoResFM CUDA only.
3D timings include all Z planes and projection. Bare legacy --benchmark means 2d-crop.
