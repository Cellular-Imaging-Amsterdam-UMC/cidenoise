import sys
from pathlib import Path
import json
import time
import numpy as np
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cidenoise.adapters import Adapter, MODEL_IDS


def main():
    import torch
    parser = argparse.ArgumentParser(description="Smoke-test every bundled checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    device = parser.parse_args().device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requires an available NVIDIA GPU")
    results = []
    for model in MODEL_IDS:
        start = time.perf_counter()
        adapter = Adapter(model, device)
        adapter.load()
        adapter.set_structure()
        batch = np.random.default_rng(42).random((1, adapter.context, 64, 64), dtype=np.float32)
        result = adapter.predict(batch)
        assert result.shape == (1, 1, 64, 64) and np.isfinite(result).all()
        repeated = adapter.predict(batch)
        np.testing.assert_allclose(result, repeated, rtol=1e-6, atol=1e-6)
        results.append(dict(model=model, shape=list(result.shape), seconds=time.perf_counter() - start,
            repeat_max_error=float(np.max(np.abs(result-repeated))),
            device=torch.cuda.get_device_name() if device == "cuda" else "cpu"))
        del adapter
        torch.cuda.empty_cache()
        print(json.dumps(results[-1]), flush=True)
    return results


if __name__ == "__main__":
    main()
