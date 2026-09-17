import pytest
from cidenoise.engine import Settings
from cidenoise.vendor.instant import ArrayUnpickler


def test_resource_presets_and_manual_override():
    assert (Settings().resolved().tile_size, Settings().resolved().batch_size) == (512,2)
    assert Settings(model="noise2noise-fmd").resolved().tile_size == 512
    assert Settings(model="cellpose-cyto3").resolved().tile_size == 224
    assert Settings(batch_size=2).resolved().batch_size == 2
    with pytest.raises(ValueError):
        Settings(model="noise2noise-fmd",tile_size=72).validate()
    with pytest.raises(ValueError):
        Settings(precision="invalid").validate()


@pytest.mark.parametrize('mode',['off','2d-full','2d-crop','3d-full','3d-crop'])
def test_eight_channel_menus_and_cli_agree(monkeypatch,mode):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import launcher, bilayers_cli, shlex
    from wrapper import parser
    config=launcher.load_config()
    menus=[p for p in config["parameters"] if p["name"].startswith("structure_")]
    assert len(menus)==8 and all(p["type"]=="dropdown" for p in menus)
    values={"structure_1":"nuclei","structure_8":"neuronal processes","model":"noise2noise-fmd","benchmark":mode}
    local=launcher.build_local_command(config,values,"/data/in","/data/out","python")
    generated=shlex.split(bilayers_cli.generate_cli_command(config,dict(values,infolder="/data/in",outfolder="/data/out")))
    a,b=parser().parse_args(local[2:]),parser().parse_args(generated[2:])
    assert vars(a)==vars(b)
    assert a.benchmark==mode
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
    window._apply_settings({"values":{"structures":'{"2":"fine dendritic processes"}',"benchmark":True}})
    assert window.values()['benchmark']=='2d-crop'
    assert window.values()["structure_2"]=="fine dendritic processes"
    assert window.values()["structure_1"]=="task-only"
    window.close()
