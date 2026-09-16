"""Compare adapters to original upstream forward passes on identical CUDA inputs."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cidenoise.adapters import Adapter, MODEL_IDS


def main():
    results = []
    for identifier in MODEL_IDS:
        adapter = Adapter(identifier, "cuda")
        adapter.load()
        adapter.set_structure()
        if identifier == "fluoresfm":
            path = ROOT / ".cache/upstream/napari-fluoresfm/src/napari_fluoresfm/fluoresfm/models/unet_sd_c.py"
            source = path.read_text().replace("napari_fluoresfm.fluoresfm.models.unet_attention", "cidenoise.vendor.fluoresfm.unet_attention")
            namespace = {}
            exec(compile(source, str(path), "exec"), namespace)
            with torch.device("meta"):
                original = namespace["UNetModel"](in_channels=1, out_channels=1, channels=320, n_res_blocks=1,
                    attention_levels=[0,1,2,3], channel_multipliers=[1,2,4,4], n_heads=8,tf_layers=1,d_cond=768,pixel_shuffle=False,scale_factor=4)
        else:
            path = ROOT / ".cache/upstream/UniFMIR/model/swinir.py"
            source = path.read_text().replace("from model.enlcn import ENLCN", "")
            namespace = {}
            exec(compile(source,str(path),"exec"),namespace)
            # Small model; CPU initialization avoids .item() on meta tensors upstream.
            original = namespace["swinir"](upscale=1,in_chans=adapter.context)
        original.load_state_dict(adapter.model.state_dict(), assign=True)
        original.eval()
        batch = np.random.default_rng(42).random((1, adapter.context, 64, 64), dtype=np.float32)
        result = adapter.predict(batch)
        with torch.inference_mode():
            x = torch.from_numpy(batch).cuda()
            reference = original(x,None,adapter.embedding) if identifier == "fluoresfm" else original(x)
            reference = reference.cpu().numpy()
        np.testing.assert_allclose(result,reference,rtol=2e-4,atol=2e-5)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        adapter.predict(np.repeat(batch,4,axis=0))
        elapsed = time.perf_counter()-tick
        entry = dict(model=identifier, maximum_absolute_error=float(np.max(np.abs(result-reference))), batch4_seconds=elapsed)
        results.append(entry)
        print(json.dumps(entry),flush=True)
        del adapter,original,x,reference,result
        torch.cuda.empty_cache()
    (ROOT / "outputs/validation").mkdir(parents=True,exist_ok=True)
    (ROOT / "outputs/validation/upstream_equivalence.json").write_text(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
