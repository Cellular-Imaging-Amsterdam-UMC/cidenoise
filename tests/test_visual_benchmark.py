import json
import numpy as np
import pytest
import zarr
from PIL import Image
from cidenoise.engine import Settings
from cidenoise.visual_benchmark import run_benchmark, Crop
from cidenoise.ome_zarr import open_images


def store(path,axes,shape):
    g=zarr.open_group(str(path),mode='w')
    data=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)%256
    g.create_dataset('0',data=data)
    g.attrs['multiscales']=[dict(version='0.4',axes=[dict(name=a) for a in axes],datasets=[dict(path='0')])]
    return data


class Identity:
    calls=[]
    def __init__(self,model,device,precision):
        self.context=5 if model=='unifmir-tribolium' else 1
        self.provenance={'model':model};self.model=model
    def load(self): pass
    def set_structure(self,s): pass
    def predict(self,batch):
        self.calls.append((self.model,batch.shape))
        return batch[:,self.context//2:self.context//2+1]


@pytest.mark.parametrize('axes,shape',[('cyx',(2,40,48)),('czyx',(2,2,40,48))])
def test_one_gallery_all_models_and_channels(tmp_path,axes,shape):
    source=tmp_path/'in.ome.zarr';original=store(source,axes,shape)
    out=tmp_path/'out';Identity.calls=[]
    run_benchmark([source],out,Settings(channels='1'),Identity)
    files=list(out.iterdir());assert len(files)==1 and files[0].suffix=='.png'
    meta=json.loads(Image.open(files[0]).info['cidenoise'])
    assert len(meta['models'])==6 and len(meta['display'])==2
    assert min(r['relative_to_fastest'] for r in meta['models'])==1.
    assert all(r['processing_seconds']>0 for r in meta['models'])
    assert meta['z']==(1 if 'z' in axes else 0)
    assert any(name=='unifmir-tribolium' and shape[1]==5 for name,shape in Identity.calls)
    np.testing.assert_array_equal(zarr.open_group(str(source),mode='r')['0'][:],original)
    with pytest.raises(FileExistsError): run_benchmark([source],out,Settings(),Identity)


def test_crop_coordinates_and_axis_order(tmp_path):
    source=tmp_path/'in.ome.zarr';data=store(source,'xyz',(520,530,3))
    image=open_images(source)[1][0];crop=Crop(image)
    assert (crop.x0,crop.y0)==(4,9)
    np.testing.assert_array_equal(crop.plane(0,0,1),data[4:516,9:521,1].T)


def test_failure_is_visible_and_does_not_skip_other_models(tmp_path):
    class Broken(Identity):
        def load(self):
            if self.model=='fluoresfm': raise FileNotFoundError('missing weights')
    source=tmp_path/'in.ome.zarr';store(source,'yx',(32,32));out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='Some benchmark models failed'):
        run_benchmark([source],out,Settings(),Broken)
    meta=json.loads(Image.open(next(out.glob('*.png'))).info['cidenoise'])
    assert len(meta['models'])==6 and 'error' in meta['models'][0]
