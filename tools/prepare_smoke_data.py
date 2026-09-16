"""Prepare small regular/HCS fixtures from read-only local brain1 pixels."""
import copy
from pathlib import Path
import sys
import zarr

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = zarr.open(str(ROOT / "localdata/brain1.ome.zarr"), mode="r")
    base = ROOT / ".cache/container-input"
    base.mkdir(parents=True, exist_ok=True)
    def populate(group):
        group.attrs.update(copy.deepcopy(dict(source.attrs)))
        group.create_dataset("0", data=source["0"][:, 7:10, 480:544, 480:544], chunks=(1,1,64,64))
        group["0"].attrs["_ARRAY_DIMENSIONS"] = ["c","z","y","x"]
        meta = group.attrs["multiscales"]
        scales = meta[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        meta[0]["datasets"][0]["coordinateTransformations"].append({"type":"translation","translation":[0,7*scales[1],480*scales[2],480*scales[3]]})
        group.attrs["multiscales"] = meta
    if not (base / "brain-crop.ome.zarr").exists():
        populate(zarr.open_group(str(base / "brain-crop.ome.zarr"), mode="w"))
    if not (base / "plate.ome.zarr").exists():
        plate = zarr.open_group(str(base / "plate.ome.zarr"), mode="w")
        plate.attrs["plate"] = dict(version="0.4", name="CIDenoise smoke plate", rows=[{"name":"A"}], columns=[{"name":"1"}], wells=[{"path":"A/1","rowIndex":0,"columnIndex":0}], field_count=1)
        well = plate.require_group("A/1")
        well.attrs["well"] = {"version":"0.4","images":[{"path":"0"}]}
        populate(well.require_group("0"))
    print(base)


if __name__ == "__main__":
    main()
