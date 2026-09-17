# Diagnose and recover a CIDenoise run

Start with the launcher log, the wrapper's `cidenoise_*.log` in the output folder,
or the retained scheduler/container log. Preserve the model, resolved parameters,
error, input paths, and process/job ID when explaining a failure.

| Symptom | Relevant action |
|---|---|
| Missing or corrupt checkpoint/text encoder | Use `tools/download_models.py --verify-only` to check assets. Run the downloader during setup when retrieval is requested; inference never downloads or substitutes weights. Check `CIDENOISE_MODELS` if using a custom cache. |
| CUDA unavailable | Check the selected Conda interpreter and PyTorch installation. For Docker also check GPU exposure. Use CPU if that fits the user's request; do not describe a CPU run as CUDA validation. |
| GPU memory exhausted | Reduce batch size first, then tile size within valid limits. Loading the model itself also needs memory. Retain the chosen checkpoint and explain any parameter change before the retry. |
| Existing output destination | Preserve the completed output and choose a new destination for another run. There is no overwrite flag. The benchmark can reuse completed outputs only after its settings validation. |
| Hidden `.partial` store after interruption | First establish that its owning run is no longer active. A new run uses a new temporary sibling. Remove an abandoned partial only when cleanup is in scope and its resolved path is verified. |
| Windows publication access denied | The workflow briefly retries the final rename. Persistent failure requires checking open handles, permissions, or storage errors; do not loop indefinitely or treat the temporary store as complete. |
| Unsupported Zarr/NGFF or axes | Explain the supported NGFF 0.4/Zarr v2 contract. XY pyramids are supported; Z-downsampled pyramids and multiple multiscale series in one image group are rejected. Format conversion is a separate operation. |
| Nonfinite inputs or predictions | Identify the affected store/channel and retain the error. Do not silently replace NaNs, clip neural output to hide failures, or switch models. |
| Delayed BIOMERO status | Query the original run ID and logs. Confirm whether submission succeeded before considering a retry. |

An ordinary exception cleans up the temporary output owned by that run. A force-killed
process may leave it behind. Completed outputs are published only after processing and
metadata finalization succeed; the original input remains read-only.

For paired LAS-X benchmark failures, retain `outputs/benchmark/frozen.json`: it contains the brain1
calibration, settings, and manifest hash used for brain2. Use a new benchmark output
folder to change settings. Similarity to LAS-X is a processed-reference comparison,
not proof that faint structures are real or preserved.

`tools/biomero_import_smoke.py` is an optional site-specific import probe. It needs an
existing deployment and metadata-probe helper; do not assume those exist elsewhere.
Successful import validates the import path, not Slurm registration or full workflow
scheduling. Report those validation levels separately.
