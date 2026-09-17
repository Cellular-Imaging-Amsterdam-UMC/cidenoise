# Memory presets and fast pretrained inference

`--tile-size 0 --overlap -1 --batch-size 0` selects these presets. Override any
individual number explicitly. The launcher and Bilayers use the same defaults.

| Model | Tile | Overlap | Batch | Automatic precision |
|---|---:|---:|---:|---|
| FluoResFM | 64 | 16 | 16 | FP16 on CUDA, FP32 on CPU |
| UniFMIR, both checkpoints | 64 | 16 | 4 | FP32 |
| Noise2Noise FMD | 512 | 128 | 2 | FP32 |
| Cellpose denoise cyto3 / nuclei | 224 | 64 | 8 | FP32 |

The CUDA allocator is capped at the smaller of 9 GiB and 80% of device capacity.
This leaves space on a 12 GB card for display/driver overhead; it cannot control other
applications or allocations outside PyTorch. Manual settings can still cause OOM.
Planes are processed sequentially, with only the required neighboring Z context in RAM.
Very large XY planes still increase host memory; these measurements cover the supplied
1024-square images, not arbitrary image sizes.

## Measured on this workstation, 16 September 2026

RTX A5000 (24 GB), CUDA allocator capped at 9 GiB, Windows/PyTorch 2.11 CUDA 12.6.
These are **not measurements on a physical 12 GB GPU**. Timings use the same brain1
C1/Z9 1024-square plane, after model/prompt warmup, including tiling and reconstruction
but excluding checkpoint loading, file I/O and pyramid generation. TF32 is disabled.

| Model/settings | Seconds/plane | Peak reserved GPU GiB |
|---|---:|---:|
| FluoResFM FP32, 64/16/batch4 | 23.20 | 3.43 |
| FluoResFM FP16, 64/16/batch16 | 6.48 | 4.16 |
| FluoResFM FP16, 64/16/batch32 | 6.18 | 5.68 |
| Noise2Noise FMD, preset | 0.186 | 1.08 |
| Cellpose denoise cyto3, preset | 0.197 | 0.69 |
| Cellpose denoise nuclei, preset | 0.155 | 0.67 |

Batch 16 is preferred over 32 for FluoResFM: almost the same speed with more headroom.
Its measured peak process RAM including model/text-encoder setup was 5.56 GiB, leaving
room within a 16 GB machine. Larger FluoResFM tiles were slower in the tested sweep;
its attention cost grows with spatial extent. Compilation is disabled.

FP16 FluoResFM differs from FP32: normalized RMSE 0.000201 and maximum absolute
difference 0.00153 on this plane. Use `--precision float32` for reference comparisons.
These measurements do not establish equivalence on all possible images.

Noise2Noise's PyTorch port matches the original TensorFlow architecture with published
weights: maximum absolute error 2.24e-7 on two seeded 128-square patches. Cellpose's
vendored CPnet is unchanged and both checkpoint forwards match upstream exactly on
two 224-square patches. Whole-image padding/blending is CIDenoise's implementation;
patch equivalence is not a claim of identical output to the original ImageJ/Cellpose UI.
Cellpose is evaluated at native sampling (no diameter rescaling).

## Quick review data

`python tools/make_benchmark_crops.py` makes exactly paired central crops of the raw
and LAS-X stores under `outputs/benchmark-small-inputs`. All four channels are kept:
brain1 CZYX=4x6x256x256 (48x fewer pixels); brain2=4x4x256x256 (32x fewer pixels).
Original source stores are read-only. Crop translations preserve physical location.
NGFF channel metadata is retained; source OME-XML is not copied into these standalone
single-level crops. `pairs.json` records the origins and explicit raw/reference pairs.

Run `python -m cidenoise.benchmark --localdata outputs/benchmark-small-inputs
--output outputs/benchmark-small-fast` (one line) for all six checkpoints.
The report includes matched views, residuals, intensity metrics and fixed brain1-derived
brightness calibration for brain2. Crops are screening data, not representative proof
of full-volume quality. LAS-X is a processed comparator, not ground truth.

Local artifacts: `outputs/validation/fluoresfm-precision-speed.json`,
`fast-model-smoke.json`, `fast-model-equivalence.json`, and
`outputs/benchmark-small-fast/report/index.html`. Data, weights and reports are ignored
by Git. The existing v0.1.0 Docker image predates these additions and must be rebuilt
explicitly before using the new command parameters in Docker.

## Launcher gallery modes

Visual Benchmark Gallery offers Off, 2D Full, 2D Crop, 3D Full, 3D Crop.
2D restores middle Z; 3D restores every Z-plane before maximum projection. Full
keeps XY; crop uses central 512x512 without shortening Z. Each field/timepoint gets
one original-plus-six-model PNG. Times include reads, normalization, inference and
projection for every requested plane/channel, excluding model/prompt loading and
PNG rendering. Ratios divide by the fastest successful model (1.0x). Display limits
are shared from raw plane/projection. Full/3D takes longer and projections can hide
slice-specific artifacts. This is separate from the paired LAS-X HTML benchmark.
See README's applicability table: device/tiles/overlap/batch/precision affect all
models; biological structure prompts only FluoResFM; dtype only OME-Zarr inference.
