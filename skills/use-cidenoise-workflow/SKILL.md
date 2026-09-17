---
name: use-cidenoise-workflow
description: Configure, run, monitor, and troubleshoot the CIDenoise Bilayers/BIOMERO workflow for pretrained fluorescence denoising of OME-Zarr images or HCS plates with FluoResFM, UniFMIR, Noise2Noise and Cellpose.
metadata:
  version: "2"
---

# Use CIDenoise Workflow

Locate the CIDenoise checkout containing `config.yaml`, `wrapper.py`, and
`model_manifest.json`. An installed copy of this skill may live outside that checkout.
Use its current descriptor and model manifest as the source of truth for the workflow
revision, options, assets, and container tag.

## Configure and run

1. Resolve the intended input stores, output folder, model, channels, and execution
   target from the user's request. Inspection or parameter advice alone does not
   imply running a job. An existing request to run the workflow is authorization;
   ask only for missing information that affects the run.
2. Read [PARAMETERS.md](references/PARAMETERS.md) when selecting or changing settings.
   Validate NGFF 0.4/Zarr v2 inputs, one-based channel indices, and the requested
   CPU/GPU resource. Keep fluorescence channels independent from Z context.
3. Use the checkout's `cidenoise` environment for local commands. On the Windows
   workstation its interpreter is
   `%LOCALAPPDATA%\miniconda3\envs\cidenoise\python.exe`. `create_env.cmd` prepares
   the environment, downloads pinned assets, and performs a CUDA smoke test.
   Inference itself must run offline; missing weights require a separate setup step.
4. Choose the execution interface that matches the request:
   - `launch.cmd`: configurable PyQt launcher, **Run Locally**, **Run Docker**, saved
     parameters, and live logs.
   - `wrapper.py`: shared local/container command entrypoint. Local execution needs
     explicit `--infolder` and `--outfolder`; container defaults are `/data/in` and
     `/data/out`. Mount container inputs read-only.
   - Bilayers/BIOMERO: use the configured deployment's workflow registration and
     execution interface. `config.yaml` describes the workflow; a local Docker image
     or successful import test does not prove it is registered with Slurm.
5. State the resolved input, model/checkpoint, channels, device, output location, and
   revision with the run update. Submit once and retain the process/container/job ID.
   Monitor that run and its logs; a delayed status response is not a reason to resubmit.

The wrapper processes every top-level `.ome.zarr` independently. Place only intended
inputs in its input folder. Use the Visual Benchmark Gallery modes to compare the
original image with all six models without producing denoised stores.

## Verify completion

`--benchmark MODE` (Visual Benchmark Gallery) accepts `off`, `2d-full`, `2d-crop`,
`3d-full`, `3d-crop`. Enabled modes write one mode-labelled PNG per image/field/timepoint,
with original plus all six models and every channel. Full keeps XY; crop uses central
512x512 without shortening Z. 2D restores middle Z; 3D restores every Z-plane before
per-channel maximum projection. Small images keep available extent. Tribolium uses
neighboring Z context in every mode. Shared display limits come from raw plane or
projection. Model/channel selection and output dtype are ignored. Times include all
processed planes and projection, exclude checkpoint/prompt loading, and give ratios
to fastest. PNG metadata/logs identify failures. Projections can hide Z-specific defects.

For ordinary inference, expect `<source>__cidenoise.ome.zarr` in the output folder. Hidden
`.partial` stores are incomplete. Check process success and completed stores, rather
than relying only on a scheduler's completion state.

Verify dimensions, axis order, physical coordinates, channel metadata, HCS well/field
layout, finite intensities, and the root `cidenoise` provenance. Selected channels are
replaced in a new store; unselected channels and inherited labels are retained. Lower
XY pyramid levels are regenerated for selected channels. Float32 output converts all
intensity channels because each Zarr array has one dtype.

The workflow produces denoised OME-Zarr intensities, not segmentation, measurements,
or a measurement database. Report the actual checkpoint hashes, settings, device,
timing, clipping, output paths, and final status from the run. Restoring input scaling
does not establish intensity conservation or biological fidelity.

Read [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) when a run or validation fails.
For maintenance, `test.cmd` checks the I/O contract and `tools/cuda_smoke.py` exercises
the real checkpoints. Docker builds, registry pushes, releases, and deployment changes
should follow the requested maintenance scope; configuring or running inference does
not implicitly request publication.
