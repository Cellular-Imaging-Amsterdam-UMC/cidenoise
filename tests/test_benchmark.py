import numpy as np
import pytest
from cidenoise.benchmark import alignment, metrics, gallery


def test_gain_adjusted_metrics_do_not_hide_direct_intensity_change(tmp_path):
    raw=np.random.default_rng(8).uniform(1,100,(3,32,32)).astype(np.float32)
    reference=raw*10
    calibration=dict(slope=0.1,intercept=0,data_range=100,display_low=0,display_high=100)
    result=metrics(raw,raw,reference,calibration,True)
    assert result["raw_units_mae_vs_unscaled_lasx"] > 100
    assert result["mae_vs_scaled_lasx"] < 1e-5
    assert result["ssim_vs_scaled_lasx"] == pytest.approx(1)
    path=tmp_path / "gallery.png"
    gallery(path,{"Raw":raw,"LAS-X scaled":reference*0.1},calibration)
    assert path.is_file()


def test_misaligned_reference_is_not_scored():
    raw=np.random.default_rng(8).uniform(1,100,(3,32,32)).astype(np.float32)
    reference=np.roll(raw,5,axis=2)
    aligned=alignment(raw,reference)
    assert not aligned["accepted"]
    result=metrics(raw,raw,reference,dict(slope=1,intercept=0,data_range=100),False)
    assert result["ssim_vs_scaled_lasx"] is None
    assert result["psnr_vs_scaled_lasx"] is None
