# CIDenoise parameters

Read the current `config.yaml` and `wrapper.py` before constructing a command. The
launcher and Bilayers descriptor use the same parameter contract.

| Setting | Default | Meaning |
|---|---|---|
| `--infolder` | `/data/in` | Top-level NGFF 0.4/Zarr v2 `.ome.zarr` images or HCS plates; set explicitly for local runs. |
| `--outfolder` | `/data/out` | Separate writable destination; existing output stores are rejected. |
| `--model` | `fluoresfm` | Explicit checkpoint choice; alternatives are `unifmir-planaria` and `unifmir-tribolium`. |
| `--channels` | `all` | One-based channel numbers, e.g. `1,3`; keep other channels unchanged. |
| `--device` | `auto` | CUDA when available, otherwise CPU; explicit `cuda` fails if unavailable. |
| `--tile-size` | `64` | XY tile edge, at least 64 and divisible by 8. Larger tiles increase memory use. |
| `--overlap` | `16` | Blended XY overlap, at least zero and smaller than tile size. |
| `--batch-size` | `4` | Positive number of tiles per inference batch. |
| `--output-dtype` | `source` | Original dtype with integer rounding/clipping, or `float32` for review. |
| `--structures` | `{}` | JSON map of one-based channels to user-supplied FluoResFM descriptions. |

FluoResFM predicts each XY plane independently with task-only denoising conditioning
unless descriptions are supplied. Its plane normalization uses the 3rd/99.5th
percentiles after restricting inputs to nonnegative values. Do not infer biological
structures from display colors. There is no user-selectable deconvolution,
super-resolution, training, mixed precision, or compilation mode.

UniFMIR Planaria uses one plane. Tribolium uses five neighboring Z planes from the
same channel with reflected boundaries. Both use per-channel stack 2nd/99.8th
percentiles, independently for each timepoint. Short and single-plane stacks are
supported through reflection. Do not substitute one checkpoint after another fails.

Use `--structures` as a single argument containing JSON, for example
`'{"1":"nuclei","4":"neuronal processes"}'` in PowerShell. It affects FluoResFM only.
The launcher accepts the JSON text without shell quoting.

Local example, from the repository root:

```powershell
& "$env:LOCALAPPDATA\miniconda3\envs\cidenoise\python.exe" wrapper.py `
  --infolder inputfolder --outfolder outputfolder `
  --model fluoresfm --channels 1,3 --device cuda
```

For Docker, the launcher's GPU checkbox controls `--gpus all`; the `device` parameter
still controls inference device selection inside the container. The local image is
`w_cidenoise:<version.txt>`. The registry organization/name/tag are in `config.yaml`.
