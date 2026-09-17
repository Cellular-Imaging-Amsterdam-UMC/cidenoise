"""Container and local command entrypoint."""
import argparse
import logging
from pathlib import Path
from datetime import datetime
import uuid
import json
from cidenoise.adapters import Adapter, MODEL_IDS
from cidenoise.engine import Settings, run_store


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--infolder", default="/data/in")
    p.add_argument("--outfolder", default="/data/out")
    p.add_argument("--model", choices=MODEL_IDS, default="fluoresfm")
    p.add_argument("--benchmark", action="store_true", help="One PNG gallery per image/timepoint: original and every model, all channels, centre crop of middle Z-plane")
    p.add_argument("--channels", default="all")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--tile-size", type=int, default=0, help="0: model-specific 12 GB preset")
    p.add_argument("--overlap", type=int, default=-1, help="-1: model-specific preset")
    p.add_argument("--batch-size", type=int, default=0, help="0: model-specific preset")
    p.add_argument("--output-dtype", choices=("source", "float32"), default="source")
    p.add_argument("--structures", default="{}")
    p.add_argument("--precision", choices=("auto", "float32", "float16"), default="auto")
    for channel in range(1,9):
        p.add_argument(f"--structure-{channel}", default="", help=f"FluoResFM structure for channel {channel}; empty/task-only means no description")
    p.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
    try:
        descriptions = settings.validate()
        for channel in range(1,9):
            value = getattr(args, f"structure_{channel}")
            if value:
                if value == "task-only":
                    descriptions.pop(str(channel), None)
                else:
                    descriptions[str(channel)] = value
        settings.structures = json.dumps(descriptions)
        settings.validate()
        inputs = sorted(Path(args.infolder).glob("*.ome.zarr"))
        if not inputs:
            raise ValueError(f"No top-level .ome.zarr stores in {args.infolder}")
        output = Path(args.outfolder).resolve()
        if any(output == source.resolve() or source.resolve() in output.parents for source in inputs):
            raise ValueError("Output folder must not be inside an input store")
        output.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(output / f"cidenoise_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.info("Settings: %s", settings)
        if args.benchmark:
            from cidenoise.visual_benchmark import run_benchmark
            run_benchmark(inputs, output, settings)
            return 0
        adapter = Adapter(settings.model, settings.device, precision=settings.precision)
        for source in inputs:
            run_store(source, args.outfolder, settings, adapter)
        return 0
    except Exception as exc:
        logging.exception("Denoising failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
