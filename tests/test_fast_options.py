import json
from pathlib import Path
import numpy as np
import pytest
import zarr
from cidenoise.engine import Settings
from cidenoise.vendor.instant import ArrayUnpickler


def test_resource_presets_and_manual_override():
    assert (Settings().resolved().tile_size, Settings().resolved().batch_size) == (64,16)
    assert Settings(model="noise2noise-fmd").resolved().tile_size == 512
    assert Settings(model="cellpose-cyto3").resolved().tile_size == 224
    assert Settings(batch_size=2).resolved().batch_size == 2
    with pytest.raises(ValueError):
        Settings(model="noise2noise-fmd",tile_size=72).validate()
    with pytest.raises(ValueError):
        Settings(precision="invalid").validate()


def test_eight_channel_menus_and_cli_agree(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import launcher, bilayers_cli, shlex
    from wrapper import parser
    config=launcher.load_config()
    menus=[p for p in config["parameters"] if p["name"].startswith("structure_")]
    assert len(menus)==8 and all(p["type"]=="dropdown" for p in menus)
    values={"structure_1":"nuclei","structure_8":"neuronal processes","model":"noise2noise-fmd","benchmark":True}
    local=launcher.build_local_command(config,values,"/data/in","/data/out","python")
    generated=shlex.split(bilayers_cli.generate_cli_command(config,dict(values,infolder="/data/in",outfolder="/data/out")))
    a,b=parser().parse_args(local[2:]),parser().parse_args(generated[2:])
    assert vars(a)==vars(b)
    assert a.structure_8=="neuronal processes" and a.precision=="auto"


def test_restricted_checkpoint_loader_rejects_code():
    import io,pickle
    with pytest.raises(ValueError,match="Unexpected checkpoint object"):
        ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_launcher_preserves_legacy_custom_descriptions(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import launcher
    app=launcher.QApplication.instance() or launcher.QApplication([])
    window=launcher.Window()
    window._apply_settings({"values":{"structures":'{"2":"fine dendritic processes"}'}})
    assert window.values()["structure_2"]=="fine dendritic processes"
    assert window.values()["structure_1"]=="task-only"
    window.close()


def test_crop_exact_pixels_and_coordinates(tmp_path):
    from tools.make_benchmark_crops import crop
    source=tmp_path/"source.ome.zarr";target=tmp_path/"small.ome.zarr"
    data=np.arange(2*5*12*12,dtype=">u2").reshape(2,5,12,12)
    g=zarr.open_group(str(source),mode="w");g.create_dataset("0",data=data)
    g.attrs["multiscales"]=[dict(version="0.4",axes=[dict(name=a,type="channel" if a=="c" else "space") for a in "czyx"],datasets=[dict(path="0",coordinateTransformations=[dict(type="scale",scale=[1,.3,.07,.07]),dict(type="translation",translation=[0,1,2,3])])])]
    crop(source,target,(1,3,4),(3,6,6))
    out=zarr.open_group(str(target),mode="r")
    np.testing.assert_array_equal(out["0"][:],data[:,1:4,3:9,4:10])
    np.testing.assert_allclose(out.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][1]["translation"],[0,1.3,2.21,3.28])
    np.testing.assert_array_equal(g["0"][:],data)
    with pytest.raises(FileExistsError):
        crop(source,target,(1,3,4),(3,6,6))
