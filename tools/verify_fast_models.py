"""Verify new adapters against upstream inference; never train.

Development-only: supply a separate TensorFlow 2.15 installation with
--tensorflow-site, or install it in a separate verification environment.
Pinned upstream source is downloaded into .cache if absent. Runtime inference
does not use TensorFlow or network access.
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def upstream(relative, url):
    path=ROOT/".cache/verification"/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url,path)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensorflow-site",type=Path,default=ROOT/".cache/tf-reference")
    p.add_argument("--write-reference",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    manifest=json.loads((ROOT/"model_manifest.json").read_text())
    folder=ROOT/"outputs/validation";folder.mkdir(parents=True,exist_ok=True)
    reference_path=folder/"noise2noise-tensorflow-reference.npz"
    if args.write_reference:
        sys.path.insert(0,str(args.tensorflow_site.resolve()))
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
        import tensorflow as tf
        import numpy as np
        from cidenoise.vendor.instant import ArrayUnpickler
        entry=manifest["models"]["noise2noise-fmd"]
        url="https://raw.githubusercontent.com/ND-HowardGroup/Instant-Image-Denoising/"+entry["upstream_revision"]+"/Plugins/Image_Denoising_Plugins_Journal/best_results_with_configurations/Noise2Noise_NBN/Unet_nbn_lr3_200e_changed_lr_weights.py"
        source=upstream("noise2noise.py",url)
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="get_unet")
        namespace={k:getattr(tf.keras.layers,k) for k in ["Conv2D","LeakyReLU","MaxPooling2D","UpSampling2D","concatenate","Activation"]}
        namespace["init_ortho"]=tf.keras.initializers.Orthogonal()
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
        x=tf.keras.Input(shape=(None,None,1),name="img")
        model=tf.keras.Model(x,namespace["get_unet"](x))
        with (ROOT/"models"/entry["checkpoint"]).open("rb") as stream:
            weights=ArrayUnpickler(stream).load()
        assert len(model.layers)==len(weights)
        for layer,(name,values) in zip(model.layers,weights):
            assert layer.name==name
            layer.set_weights(values)
        sample=np.random.default_rng(42).uniform(-.5,.5,(2,128,128,1)).astype("float32")
        np.savez(reference_path,input=sample,output=model(sample,training=False).numpy())
        return
    subprocess.run([sys.executable,__file__,"--write-reference","--tensorflow-site",str(args.tensorflow_site)],check=True)
    import numpy as np
    import torch
    from cidenoise.adapters import Adapter
    results=[]
    data=np.load(reference_path)
    model=Adapter("noise2noise-fmd","cuda")
    result=model.predict(data["input"].transpose(0,3,1,2))
    reference=data["output"].transpose(0,3,1,2)
    np.testing.assert_allclose(result,reference,atol=2e-6,rtol=2e-5)
    results.append(dict(model=model.model_id,max_abs=float(abs(result-reference).max())))
    del model
    rev=manifest["models"]["cellpose-cyto3"]["upstream_revision"]
    source=upstream("cellpose_resnet.py",f"https://raw.githubusercontent.com/MouseLand/cellpose/{rev}/cellpose/resnet_torch.py")
    spec=importlib.util.spec_from_file_location("original_cellpose",source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    for name in ["cellpose-cyto3","cellpose-nuclei"]:
        model=Adapter(name,"cuda");model.load()
        net=original.CPnet([1,32,64,128,256],1,3).cuda().eval()
        net.load_state_dict(model.model.state_dict())
        sample=np.random.default_rng(42).random((2,1,224,224),dtype=np.float32)
        result=model.predict(sample)
        with torch.inference_mode():
            reference=net(torch.from_numpy(sample).cuda())[0].cpu().numpy()
        np.testing.assert_array_equal(result,reference)
        results.append(dict(model=name,max_abs=float(abs(result-reference).max())))
        del model,net
        torch.cuda.empty_cache()
    (folder/"fast-model-equivalence.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
