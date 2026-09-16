"""Create small, exactly paired brain/LAS-X NGFF crops; never modify sources."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cidenoise.ome_zarr import open_images
from cidenoise.engine import publish_store


def crop(source, destination, origin, shape):
    root, images = open_images(source)
    if len(images) != 1 or images[0].path or images[0].axes != list("czyx"):
        raise ValueError("This local benchmark utility requires ordinary CZYX images")
    image = images[0]
    if destination.exists():
        raise FileExistsError(destination)
    slices = (slice(None),) + tuple(slice(start, start+size) for start,size in zip(origin,shape))
    pixels = image.array[slices]
    if pixels.shape[1:] != tuple(shape):
        raise ValueError("Crop falls outside the source")
    metadata = copy.deepcopy(dict(root.attrs))
    multiscale = metadata["multiscales"][0]
    dataset = copy.deepcopy(image.datasets[0])
    transforms = dataset["coordinateTransformations"]
    scale = np.ones(4)
    translation = np.zeros(4)
    for transform in transforms:
        if transform["type"] == "scale":
            scale *= transform["scale"]
            translation *= transform["scale"]
        elif transform["type"] == "translation":
            translation += transform["translation"]
        else:
            raise ValueError("Unsupported coordinate transform")
    translation += scale * np.array((0,*origin))
    dataset["coordinateTransformations"] = [dict(type="scale",scale=scale.tolist()),
                                             dict(type="translation",translation=translation.tolist())]
    multiscale["datasets"] = [dataset]
    if "omero" in metadata:
        metadata["omero"].setdefault("rdefs", {}).update(defaultZ=shape[0]//2,defaultT=0)
    metadata["cidenoise_benchmark_crop"] = dict(source=source.name, origin_zyx=list(origin),
        shape_zyx=list(shape), selection="fixed central crop, identical for raw and LAS-X", 
        note="Standalone NGFF crop; source OME-XML and original pyramid are not copied")
    temp = destination.with_name("."+destination.name+"."+uuid.uuid4().hex+".partial")
    try:
        output = zarr.open_group(str(temp), mode="w")
        output.attrs.update(metadata)
        array = output.create_dataset(dataset["path"],data=pixels,
            chunks=(1,1,min(256,shape[1]),min(256,shape[2])),
            compressor=image.array.compressor,dimension_separator="/")
        array.attrs.update(dict(image.array.attrs))
        np.testing.assert_array_equal(array[:],pixels)
        open_images(temp)
        publish_store(temp,destination)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return dict(source=source.name,output=destination.name,origin_zyx=list(origin),
                shape=list(pixels.shape),dtype=str(pixels.dtype),pixel_bytes=pixels.nbytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localdata",type=Path,default=ROOT/"localdata")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/benchmark-small-inputs")
    parser.add_argument("--xy",type=int,default=256)
    args = parser.parse_args()
    if args.xy < 32:
        raise ValueError("Use at least 32 pixels in XY")
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,depth in [("brain1",6),("brain2",4)]:
        raw = args.localdata/f"{name}.ome.zarr"
        reference = args.localdata/f"{name}-dn.ome.zarr"
        a,b = open_images(raw)[1][0],open_images(reference)[1][0]
        if a.axes != b.axes or a.array.shape != b.array.shape or a.datasets != b.datasets:
            raise ValueError(f"{name}: raw/reference geometry differs")
        shape=(min(depth,a.length("z")),args.xy,args.xy)
        origin=tuple((a.length(axis)-size)//2 for axis,size in zip("zyx",shape))
        if min(origin)<0:
            raise ValueError("Crop larger than source")
        for source in [raw,reference]:
            records.append(crop(source,args.output/source.name,origin,shape))
    (args.output/"pairs.json").write_text(json.dumps(dict(
        pairs=[dict(raw=f"{n}.ome.zarr",reference=f"{n}-dn.ome.zarr") for n in ("brain1","brain2")],
        crops=records),indent=2)+"\n")
    print(json.dumps(records,indent=2))


if __name__ == "__main__":
    main()
