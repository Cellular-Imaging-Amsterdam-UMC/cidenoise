import hashlib
import json
from pathlib import Path
import shlex
import numpy as np
import pytest
import zarr
from cidenoise.engine import Settings, run_store
from cidenoise.ome_zarr import open_images
from cidenoise.tiling import predict_plane, reflected_index
from cidenoise.normalization import histogram_percentile


def image_group(group, data, axes="czyx", pyramid=False):
    axes_meta = [{"name": a, "type": "channel" if a == "c" else "time" if a == "t" else "space"} for a in axes]
    datasets = [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1] * len(axes)}]}]
    group.create_dataset("0", data=data, chunks=tuple(min(32, n) if a in "yx" else 1 for a, n in zip(axes, data.shape)))
    group["0"].attrs["_ARRAY_DIMENSIONS"] = list(axes)
    if pyramid:
        slices = tuple(slice(None, None, 2) if a in "yx" else slice(None) for a in axes)
        group.create_dataset("1", data=data[slices])
        datasets.append({"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2 if a in "yx" else 1 for a in axes]}]})
    group.attrs.update(multiscales=[{"version": "0.4", "axes": axes_meta, "datasets": datasets}],
        omero={"channels": [{"label": f"C{i}", "color": "FF0000"} for i in range(data.shape[axes.index('c')] if 'c' in axes else 1)]})


def fixture_store(tmp_path, axes="czyx", pyramid=False):
    dims = dict(t=2, c=2, z=3, y=70, x=83)
    data = np.random.default_rng(1).integers(0, 1000, tuple(dims[a] for a in axes), dtype=np.uint16).astype(">u2")
    source = tmp_path / "sample.ome.zarr"
    group = zarr.open_group(str(source), mode="w")
    image_group(group, data, axes, pyramid)
    return source, data


class Identity:
    device = "cpu"
    context = 1
    provenance = {"model": "identity-test-only"}
    def load(self): pass
    def set_structure(self, value=""): pass
    def predict(self, x): return x


def digest_tree(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("axes", ["yx", "xy", "zyx", "czyx", "tczyx", "ctzxy"])
def test_axis_roundtrip_and_readonly(tmp_path, axes):
    source, data = fixture_store(tmp_path, axes)
    before = digest_tree(source)
    result = run_store(source, tmp_path / "out", Settings(), Identity())
    np.testing.assert_array_equal(zarr.open(str(result))["0"][:], data)
    assert digest_tree(source) == before
    assert zarr.open(str(result))["0"].dtype == data.dtype


def test_channel_passthrough_pyramid_and_labels(tmp_path):
    source, data = fixture_store(tmp_path, pyramid=True)
    group = zarr.open(str(source), mode="a")
    group.create_dataset("labels/old/0", data=np.ones((3, 70, 83), np.uint8))
    group["labels"].attrs["labels"] = ["old"]
    class Double(Identity):
        def predict(self, x): return x * 2
    result = run_store(source, tmp_path / "out", Settings(channels="1"), Double())
    out = zarr.open(str(result))
    np.testing.assert_array_equal(out["0"][1], data[1])
    np.testing.assert_array_equal(out["1"][1], group["1"][1])
    np.testing.assert_array_equal(out["labels/old/0"][:], group["labels/old/0"][:])
    assert not np.array_equal(out["0"][0], data[0])
    assert out.attrs["omero"] == group.attrs["omero"]
    assert out.attrs["multiscales"] == group.attrs["multiscales"]


def test_float32_and_consolidated_metadata(tmp_path):
    source, data = fixture_store(tmp_path, pyramid=True)
    (source / "OME").mkdir()
    (source / "OME/METADATA.ome.xml").write_text('<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image><Pixels SizeT="1" SizeC="2" SizeZ="3" SizeY="70" SizeX="83" Type="uint16"/></Image></OME>')
    zarr.consolidate_metadata(str(source))
    result = run_store(source, tmp_path / "out", Settings(output_dtype="float32", channels="1"), Identity())
    out = zarr.open_consolidated(str(result))
    assert out["0"].dtype == np.float32
    np.testing.assert_allclose(out["0"][:], data, atol=0.001)
    np.testing.assert_array_equal(out["1"][1], zarr.open(str(source))["1"][1])
    assert 'Type="float"' in (result / "OME/METADATA.ome.xml").read_text()
    assert 'Type="uint16"' in (source / "OME/METADATA.ome.xml").read_text()


def test_failure_cleans_temp_and_refuses_overwrite(tmp_path):
    source, _ = fixture_store(tmp_path)
    class Broken(Identity):
        def predict(self, x): raise RuntimeError("expected failure")
    with pytest.raises(RuntimeError, match="expected failure"):
        run_store(source, tmp_path / "out", Settings(), Broken())
    assert list((tmp_path / "out").iterdir()) == []
    run_store(source, tmp_path / "out", Settings(), Identity())
    with pytest.raises(FileExistsError):
        run_store(source, tmp_path / "out", Settings(), Identity())


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows sharing violation retry")
def test_transient_windows_rename_failure(tmp_path, monkeypatch):
    from cidenoise.engine import publish_store
    temp, destination = tmp_path / "partial", tmp_path / "complete"
    temp.mkdir()
    rename = Path.rename
    attempts = []
    def transient(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("Temporary Windows sharing violation")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", transient)
    publish_store(temp, destination)
    assert len(attempts) == 3 and destination.is_dir() and not temp.exists()


def test_hcs(tmp_path):
    source = tmp_path / "plate.ome.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.attrs["plate"] = {"version": "0.4", "wells": [{"path": "A/1"}, {"path": "B/2"}], "rows": [{"name":"A"},{"name":"B"}], "columns":[{"name":"1"},{"name":"2"}]}
    for well in ("A/1", "B/2"):
        group = root.require_group(well)
        group.attrs["well"] = {"images": [{"path":"0"}, {"path":"1"}]}
        for field in ("0", "1"):
            image_group(group.require_group(field), np.full((2, 1, 16, 16), 27, np.uint16))
    output = run_store(source, tmp_path / "out", Settings(), Identity())
    result, images = open_images(output)
    assert len(images) == 4
    assert result.attrs["plate"] == root.attrs["plate"]
    for image in images:
        assert np.all(image.array[:] == 27)


@pytest.mark.parametrize("shape", [(1, 1), (11, 20), (70, 83), (128, 128)])
def test_tile_blending(shape):
    plane = np.random.default_rng(1).random(shape, dtype=np.float32)
    output = predict_plane(lambda y0,y1,x0,x1: plane[None,y0:y1,x0:x1], shape, lambda x:x)
    np.testing.assert_allclose(output, plane, atol=2e-7)


def test_reflection_and_histogram():
    assert [reflected_index(z, 3) for z in range(-2, 5)] == [2, 1, 0, 1, 2, 1, 0]
    assert reflected_index(-10, 1) == 0
    data = np.array([0, 0, 2, 9, 9, 10])
    hist = np.bincount(data, minlength=11)
    for p in (0, 2, 50, 99.8, 100):
        assert histogram_percentile(hist, p) == pytest.approx(np.percentile(data, p))


def test_invalid_inputs(tmp_path):
    source, _ = fixture_store(tmp_path)
    for settings in (Settings(channels="0"), Settings(tile_size=65), Settings(overlap=64), Settings(structures='[]')):
        with pytest.raises(ValueError):
            run_store(source, tmp_path / "out", settings, Identity())
    group = zarr.open(str(source), mode="a")
    meta = group.attrs["multiscales"]
    meta[0]["version"] = "0.5"
    group.attrs["multiscales"] = meta
    with pytest.raises(ValueError, match="0.4"):
        open_images(source)


def test_checkpoint_and_cuda_errors(tmp_path, monkeypatch):
    from cidenoise.adapters import Adapter, resolve_device
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda:False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")
    with pytest.raises(FileNotFoundError, match="download_models"):
        Adapter("unifmir-planaria", "cpu", tmp_path).load()


def test_launcher_bilayers_equivalence(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import launcher
    import bilayers_cli
    config = launcher.load_config()
    assert not bilayers_cli.validate_config(config)
    values = {"structures": '{"1":"fine neuronal processes"}', "channels": "1,2"}
    local = launcher.build_local_command(config, values, "/data/in", "/data/out", "python")
    generated = shlex.split(bilayers_cli.generate_cli_command(config, dict(values, infolder="/data/in", outfolder="/data/out")))
    from wrapper import parser
    assert vars(parser().parse_args(local[2:])) == vars(parser().parse_args(generated[2:]))
    docker = launcher.build_docker_command(config, values, "/data/in", "/data/out")
    assert any(v.endswith(":/data/in:ro") for v in docker)


def test_five_plane_context_uses_z_not_channels(tmp_path):
    source = tmp_path / "context.ome.zarr"
    data = np.stack([np.stack([np.full((64,64), v, np.uint16) for v in values]) for values in ([10,20,30],[100,200,300])])
    image_group(zarr.open_group(str(source),mode="w"), data)
    seen=[]
    class Context(Identity):
        context=5
        def predict(self,x):
            seen.append(x[:, :, 0, 0].copy())
            return x[:,2:3]
    result=run_store(source,tmp_path / "out",Settings(model="unifmir-tribolium"),Context())
    np.testing.assert_array_equal(zarr.open(str(result))["0"][:],data)
    np.testing.assert_allclose(seen[0],[[1,0.5,0,0.5,1]])


def test_nonfinite_prediction_and_oom_are_actionable(tmp_path):
    source,_=fixture_store(tmp_path)
    class Nonfinite(Identity):
        def predict(self,x): return np.full_like(x,np.nan)
    with pytest.raises(ValueError,match="invalid predictions"):
        run_store(source,tmp_path / "out",Settings(),Nonfinite())
    from cidenoise.adapters import Adapter
    import torch
    adapter=Adapter("unifmir-planaria","cpu",tmp_path)
    adapter.loaded=True
    def oom(x): raise torch.cuda.OutOfMemoryError("test")
    adapter.model=oom
    with pytest.raises(RuntimeError,match="Reduce batch size"):
        adapter.predict(np.zeros((1,1,64,64),np.float32))
