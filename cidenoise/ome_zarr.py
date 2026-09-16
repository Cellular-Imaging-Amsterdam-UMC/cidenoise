"""NGFF 0.4 image and HCS I/O with explicit axis handling."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import xml.etree.ElementTree as ET
import numpy as np
import zarr


def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe NGFF relative path: {value}")
    return value


@dataclass
class Image:
    path: str
    group: object
    axes: list
    datasets: list

    @property
    def array(self):
        return self.group[self.datasets[0]["path"]]

    def length(self, axis):
        return self.array.shape[self.axes.index(axis)] if axis in self.axes else 1

    def selection(self, t=0, c=0, z=0, y=slice(None), x=slice(None)):
        values = dict(t=t, c=c, z=z, y=y, x=x)
        return tuple(values[a] for a in self.axes)

    def plane(self, t, c, z, y=slice(None), x=slice(None)):
        data = np.asarray(self.array[self.selection(t, c, z, y, x)], dtype=np.float32)
        remaining = [a for a in self.axes if a in "yx"]
        return data if remaining == ["y", "x"] else data.T


def open_images(path):
    path = Path(path)
    if not (path / ".zgroup").exists() or json.loads((path / ".zgroup").read_text())["zarr_format"] != 2:
        raise ValueError(f"{path}: only NGFF 0.4 in Zarr v2 is supported")
    root = zarr.open_group(str(path), mode="r")
    paths = []
    if "plate" in root.attrs:
        plate = root.attrs["plate"]
        if plate.get("version", "0.4") != "0.4":
            raise ValueError("Only NGFF 0.4 plates are supported")
        for well in plate["wells"]:
            wp = safe_path(well["path"])
            for image in root[wp].attrs["well"]["images"]:
                paths.append(wp + "/" + safe_path(image["path"]))
    elif "multiscales" in root.attrs:
        paths = [""]
    else:
        raise ValueError(f"{path}: expected an NGFF image or HCS plate")
    images = []
    for p in paths:
        group = root[p] if p else root
        multis = group.attrs.get("multiscales", [])
        if len(multis) != 1 or multis[0].get("version") != "0.4":
            raise ValueError(f"{p}: expected one NGFF 0.4 multiscale image")
        meta = multis[0]
        axes = [a["name"] for a in meta["axes"]]
        if len(set(axes)) != len(axes) or not set(axes) <= set("tczyx") or not {"y", "x"} <= set(axes):
            raise ValueError(f"Unsupported axes {axes}")
        datasets = meta["datasets"]
        if not datasets:
            raise ValueError("Empty multiscale datasets")
        for ds in datasets:
            array = group[safe_path(ds["path"])]
            if array.ndim != len(axes) or array.dtype.kind not in "uif" or any(n < 1 for n in array.shape):
                raise ValueError(f"Unsupported image array {p}/{ds['path']}")
        image = Image(p, group, axes, datasets)
        display_channels = group.attrs.get("omero", {}).get("channels")
        if display_channels is not None and len(display_channels) != image.length("c"):
            raise ValueError(f"{p}: OMERO channel metadata count differs from array C dimension")
        for ds in datasets[1:]:
            shape = group[ds["path"]].shape
            if any(shape[i] != image.array.shape[i] for i, a in enumerate(axes) if a not in "yx"):
                raise ValueError("Only XY-downsampled pyramids are supported in v1")
            if any(shape[i] > image.array.shape[i] for i, a in enumerate(axes) if a in "yx"):
                raise ValueError("First multiscale dataset must be highest resolution")
        images.append(image)
    if len(paths) != len(set(paths)) or not images:
        raise ValueError("Empty or duplicate HCS image paths")
    return root, images


def channels(spec, count):
    if spec == "all":
        return list(range(count))
    try:
        values = sorted(set(int(v.strip()) - 1 for v in spec.split(",")))
    except ValueError as exc:
        raise ValueError("Channels must be 'all' or one-based numbers, e.g. 1,3") from exc
    if not values or min(values) < 0 or max(values) >= count:
        raise ValueError(f"Channel selection {spec!r} outside 1..{count}")
    return values


def cast_output(data, dtype):
    dtype = np.dtype(dtype)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite restored intensities cannot be written")
    clipped = 0
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        clipped = int(np.count_nonzero((data < limits.min) | (data > limits.max)))
        data = np.rint(np.clip(data, limits.min, limits.max))
    return data.astype(dtype), clipped


def update_float_xml(store, images):
    """Keep optional OME-XML sidecars consistent with converted intensity arrays.

    Update root/per-image intensity sidecars only; inherited label sidecars stay intact.
    """
    expected = {tuple(image.length(a) for a in "tczyx") for image in images}
    candidates = {Path(store) / "OME/METADATA.ome.xml"}
    candidates.update(Path(store) / image.path / "OME/METADATA.ome.xml" for image in images)
    for path in candidates:
        if not path.is_file():
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag.startswith("{"):
            ET.register_namespace("", root.tag.split("}")[0][1:])
        changed = False
        for pixels in root.iter():
            if pixels.tag.rsplit("}", 1)[-1] != "Pixels":
                continue
            dimensions = tuple(int(pixels.get("Size" + a.upper(), "1")) for a in "tczyx")
            if dimensions in expected:
                pixels.set("Type", "float")
                changed = True
        if changed:
            tree.write(path, encoding="utf-8", xml_declaration=True)
