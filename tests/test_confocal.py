import numpy as np
import pytest
from cidenoise.normalization import confocal_bounds
from cidenoise.engine import Settings


@pytest.mark.parametrize('dtype,maximum',[('uint8',255),('>u2',65535),('float32',1)])
def test_fixed_confocal_normalization_roundtrip(dtype,maximum):
    pixels=np.array([0,maximum//2,maximum],dtype=dtype)
    low,high,_=confocal_bounds(pixels,dtype)
    normalized=(pixels.astype(np.float32)-low)/(high-low)
    np.testing.assert_allclose(normalized,pixels.astype(np.float32)/maximum-.5,atol=1e-7)
    np.testing.assert_allclose(normalized*(high-low)+low,pixels,atol=.002)
    assert confocal_bounds(pixels[:2],dtype)==(low,high,confocal_bounds(pixels,dtype)[2])


def test_confocal_rejects_ambiguous_float_range_and_bad_tiles():
    with pytest.raises(ValueError,match='requires uint8'):
        confocal_bounds(np.array([0.,255.]),'float32')
    with pytest.raises(ValueError,match='divisible by 32'):
        Settings(model='noise2noise-confocal',tile_size=72).validate()
