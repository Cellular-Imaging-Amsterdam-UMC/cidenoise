import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from training.data import NoisyPairs, EpochSamples, verify_index
from training.download_confocal import extract
from training.train_noise2noise import train_batch, rescale_learning_rate


def test_rescale_preserves_cosine_schedule_position():
    p=torch.nn.Parameter(torch.ones(1))
    optimizer=torch.optim.Adam([p],lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=.00001)
    for _ in range(15):
        optimizer.step();scheduler.step()
    saved=scheduler.state_dict()
    old_lr=optimizer.param_groups[0]["lr"]
    rescale_learning_rate(optimizer,scheduler,.1)
    assert scheduler.last_epoch==15
    assert scheduler.get_last_lr()[0]==pytest.approx(old_lr*.1)
    optimizer.step();scheduler.step()
    recovered_lr=optimizer.param_groups[0]["lr"]
    scheduler.load_state_dict(saved);optimizer.param_groups[0]["lr"]=old_lr
    optimizer.step();scheduler.step()
    assert recovered_lr==pytest.approx(optimizer.param_groups[0]["lr"]*.1)


def test_overflow_retries_without_corrupting_optimizer():
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler("cpu",init_scale=1024.)
    calls=[]
    def overflow_once(grad):
        calls.append(1)
        return torch.full_like(grad,float("inf")) if len(calls)==1 else grad
    model.weight.register_hook(overflow_once)
    loss=train_batch(model,optimizer,scaler,torch.ones(2,2),torch.zeros(2,1),"cpu",False)
    assert torch.isfinite(loss) and len(calls)==2 and scaler.get_scale()==512.
    assert all(s["step"].item()==1 for s in optimizer.state.values())
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_nonfinite_float32_gradients_remain_fatal():
    model=torch.nn.Linear(2,1)
    before=model.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters())
    model.weight.register_hook(lambda g: torch.full_like(g,float("inf")))
    with pytest.raises(RuntimeError,match="gradients in float32"):
        train_batch(model,optimizer,torch.amp.GradScaler("cpu",enabled=False),
                    torch.ones(2,2),torch.zeros(2,1),"cpu",False)
    assert torch.equal(before,model.weight) and not optimizer.state


def test_persistent_overflow_stops_without_optimizer_update():
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(model.parameters())
    model.weight.register_hook(lambda g: torch.full_like(g,float("inf")))
    with pytest.raises(RuntimeError,match="Persistent FP16"):
        train_batch(model,optimizer,torch.amp.GradScaler("cpu"),
                    torch.ones(2,2),torch.zeros(2,1),"cpu",False,max_retries=2)
    assert not optimizer.state


def records():
    return {"records":[dict(category="Confocal_BPAE_B",fov=fov,split=split,
        frames=[dict(path=f"{i}.png") for i in range(50)])
        for fov,split in [(1,"train"),(20,"validation"),(19,"test")]]}


def test_noisy_targets_independent_and_held_out():
    pairs=NoisyPairs(".",records(),averages=(1,2,4,8,16))
    for i in range(100):
        record,a,b,*_=pairs.selection(i)
        assert record["fov"]==1
        assert len(a)==len(b) and not set(a)&set(b)
    assert NoisyPairs(".",records(),"validation").selection(0)[0]["fov"]==20
    assert NoisyPairs(".",records(),"test").selection(0)[0]["fov"]==19


def test_pair_augmentation_and_normalization_are_shared():
    pairs=NoisyPairs(".",records(),patch=64,averages=(1,))
    ramp=np.tile(np.arange(512,dtype=np.float32)/512,(512,1))
    pairs._read=lambda path: ramp+int(Path(path).stem)
    _,a,b,*_=pairs.selection(8)
    x,y=pairs[8]
    np.testing.assert_allclose((y-x).numpy(),(float(b[0])-float(a[0]))/255,atol=1e-7)
    np.testing.assert_array_equal(pairs[8][0],x)
    assert x.shape==(1,64,64)


def test_resume_sampler_continues_same_samples():
    sampler=EpochSamples(24);sampler.epoch=2
    entire=list(sampler)
    sampler.skip=8
    assert list(sampler)==entire[8:] and len(sampler)==16


def test_index_rejects_clean_targets(tmp_path):
    index=records()
    index["records"][0]["frames"]=[dict(path="FMD/Confocal_BPAE_B/gt/1/avg50.png",sha256="bad")]
    with pytest.raises(ValueError,match="Only raw"):
        verify_index(tmp_path,index)


def test_confocal_extraction_rejects_traversal(tmp_path):
    archive=tmp_path/"bad.tar"
    with tarfile.open(archive,"w") as tar:
        info=tarfile.TarInfo("../escape.png");info.size=1
        tar.addfile(info,io.BytesIO(b"x"))
    root=tmp_path/"data";root.mkdir()
    with pytest.raises(ValueError,match="Unsafe archive"):
        extract(archive,root,{"name":"Confocal_BPAE_B.tar","computed_md5":"unused"})
    assert not (tmp_path/"escape.png").exists() and not list(root.iterdir())
