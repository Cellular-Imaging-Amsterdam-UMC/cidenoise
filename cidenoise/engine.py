"""Transactional image/HCS denoising, bounded to a few planes in RAM."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import logging
import os
import shutil
import time
import uuid
import numpy as np
import zarr
from skimage.transform import resize_local_mean
from . import __version__
from .ome_zarr import open_images, channels, cast_output, update_float_xml
from .normalization import plane_bounds, stack_bounds
from .tiling import predict_plane, reflected_index

log = logging.getLogger(__name__)


@dataclass
class Settings:
    model: str = "fluoresfm"
    channels: str = "all"
    device: str = "auto"
    tile_size: int = 64
    overlap: int = 16
    batch_size: int = 4
    output_dtype: str = "source"
    structures: str = "{}"

    def validate(self):
        if self.tile_size < 64 or self.tile_size % 8 or not 0 <= self.overlap < self.tile_size or self.batch_size < 1:
            raise ValueError("Tile size must be >=64 and divisible by 8; overlap must be smaller; batch size >=1")
        if self.output_dtype not in ("source", "float32"):
            raise ValueError("Output dtype must be source or float32")
        structures = json.loads(self.structures)
        if not isinstance(structures, dict) or any(not str(k).isdigit() or int(k) < 1 or not isinstance(v, str) for k, v in structures.items()):
            raise ValueError('Structures must be JSON mapping one-based channels to descriptions, e.g. {"1":"nuclei"}')
        return structures


def output_name(source):
    return Path(source).name.removesuffix(".ome.zarr") + "__cidenoise.ome.zarr"


def publish_store(temp, destination):
    # Windows indexers/antivirus can briefly hold a newly written directory open.
    for attempt in range(6):
        if destination.exists():
            raise FileExistsError(f"Output appeared during run: {destination}")
        try:
            temp.rename(destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def run_store(source, outfolder, settings, adapter=None):
    structures = settings.validate()
    source = Path(source).resolve()
    outfolder = Path(outfolder).resolve()
    destination = outfolder / output_name(source)
    if outfolder == source or source in outfolder.parents:
        raise ValueError("Output folder must not be inside the input store")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    root, images = open_images(source)
    for image in images:
        channels(settings.channels, image.length("c"))
        if any(int(c) > image.length("c") for c in structures):
            raise ValueError(f"Structure descriptions reference a channel outside 1..{image.length('c')}")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Symlinks inside input stores are not supported")
    if adapter is None:
        from .adapters import Adapter
        adapter = Adapter(settings.model, settings.device)
    adapter.load()
    import torch
    if adapter.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    outfolder.mkdir(parents=True, exist_ok=True)
    temp = outfolder / ("." + destination.name + "." + uuid.uuid4().hex + ".partial")
    started = time.perf_counter()
    records = []
    try:
        shutil.copytree(source, temp)
        # Consolidated metadata must not retain the old array dtype/attributes.
        (temp / ".zmetadata").unlink(missing_ok=True)
        out = zarr.open_group(str(temp), mode="a")
        for image in images:
            target_group = out[image.path] if image.path else out
            selected = channels(settings.channels, image.length("c"))
            # A Zarr array has one dtype for all channels. Re-create each intensity array when requested.
            if settings.output_dtype == "float32":
                for ds in image.datasets:
                    old = image.group[ds["path"]]
                    attrs = dict(old.attrs)
                    target_group.create_dataset(ds["path"], shape=old.shape, chunks=old.chunks,
                        dtype="float32", compressor=old.compressor, overwrite=True, dimension_separator="/")
                    target_group[ds["path"]].attrs.update(attrs)
                    # Copy unselected channels plane by plane at every resolution.
                    for t in range(image.length("t")):
                        for c in set(range(image.length("c"))) - set(selected):
                            for z in range(image.length("z")):
                                sel = image.selection(t, c, z)
                                target_group[ds["path"]][sel] = old[sel].astype(np.float32)
            for t in range(image.length("t")):
                for c in selected:
                    adapter.set_structure(structures.get(str(c + 1), ""))
                    volume_bounds = stack_bounds(image, t, c) if settings.model.startswith("unifmir") else None
                    for z in range(image.length("z")):
                        tick = time.perf_counter()
                        center = image.plane(t, c, z)
                        if not np.isfinite(center).all():
                            raise ValueError("Input contains NaN or infinite intensities")
                        low, high, method = volume_bounds or plane_bounds(center)
                        scale = high - low
                        constant = float(center.min()) == float(center.max())
                        if constant and adapter.context == 1:
                            restored = center
                        else:
                            scale = scale if scale > 0 else 1.0
                            radius = adapter.context // 2
                            indices = [reflected_index(z + d, image.length("z")) for d in range(-radius, radius + 1)]
                            # Cache only the current spatial context, never the whole stack/plate.
                            context = np.stack([center if zi == z else image.plane(t, c, zi) for zi in indices])
                            def read_patch(y0, y1, x0, x1):
                                data = context[:, y0:y1, x0:x1]
                                if settings.model == "fluoresfm":
                                    data = np.maximum(data, 0)
                                if not np.isfinite(data).all():
                                    raise ValueError("Input contains NaN or infinite intensities")
                                return (data - low) / (scale + (1e-20 if settings.model.startswith("unifmir") else 0))
                            restored = predict_plane(read_patch, center.shape, adapter.predict,
                                settings.tile_size, settings.overlap, settings.batch_size) * scale + low
                        array = target_group[image.datasets[0]["path"]]
                        converted, clipped = cast_output(restored, array.dtype)
                        native = converted if image.axes.index("y") < image.axes.index("x") else converted.T
                        array[image.selection(t, c, z)] = native
                        for ds in image.datasets[1:]:
                            lower = target_group[ds["path"]]
                            shape = (lower.shape[image.axes.index("y")], lower.shape[image.axes.index("x")])
                            down = resize_local_mean(converted.astype(np.float32), shape, preserve_range=True)
                            down, _ = cast_output(down, lower.dtype)
                            lower[image.selection(t, c, z)] = down if image.axes.index("y") < image.axes.index("x") else down.T
                        record = dict(field=image.path, t=t, channel=c + 1, z=z,
                            normalization=dict(low=low, high=high, method=method), constant_passthrough=constant and adapter.context == 1,
                            clipped_pixels=clipped, seconds=time.perf_counter() - tick,
                            prompt=adapter.provenance.get("prompt"))
                        records.append(record)
                        log.info("%s field=%s T=%d C=%d Z=%d/%d %.2fs", source.name, image.path or "/", t, c + 1, z + 1, image.length("z"), record["seconds"])
            if "cidenoise" in target_group.attrs:
                target_group.attrs["cidenoise_parent"] = copy.deepcopy(target_group.attrs["cidenoise"])
            target_group.attrs["cidenoise"] = dict(version=__version__, inherited_labels="labels" in target_group, model=copy.deepcopy(adapter.provenance))
        provenance = dict(version=__version__, source_name=source.name, settings=asdict(settings),
            model=copy.deepcopy(adapter.provenance), planes=records, runtime_seconds=time.perf_counter() - started,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if adapter.device == "cuda" else 0,
            versions=dict(numpy=np.__version__, torch=torch.__version__, zarr=zarr.__version__),
            inherited_labels=any("labels" in image.group for image in images), z_boundary="reflect")
        out.attrs["cidenoise"] = provenance
        if settings.output_dtype == "float32":
            update_float_xml(temp, images)
        zarr.consolidate_metadata(str(temp))
        publish_store(temp, destination)
        return destination
    except BaseException:
        # Only remove the unique temporary directory created by this invocation.
        if temp.exists() and temp.parent == outfolder and temp.name.endswith(".partial"):
            shutil.rmtree(temp)
        raise
