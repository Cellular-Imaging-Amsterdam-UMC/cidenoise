"""Pretrained-only model adapters. Runtime never downloads assets."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

MODEL_IDS = ("fluoresfm", "unifmir-planaria", "unifmir-tribolium", "noise2noise-fmd", "noise2noise-confocal", "cellpose-cyto3", "cellpose-nuclei")
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(device):
    import torch
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("Device must be auto, cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install the CUDA PyTorch build or select cpu")
    return "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device


class Adapter:
    def __init__(self, model_id, device="auto", models_dir=None, precision="float32"):
        if model_id not in MODEL_IDS:
            raise ValueError(f"Unknown model: {model_id}")
        self.model_id = model_id
        self.device = resolve_device(device)
        if precision not in ("auto", "float32", "float16"):
            raise ValueError("Precision must be auto, float32 or float16")
        self.precision = ("float16" if self.device == "cuda" and model_id == "fluoresfm" else "float32") if precision == "auto" else precision
        if self.precision == "float16" and self.device != "cuda":
            raise ValueError("Float16 inference requires CUDA; use auto or float32 on CPU")
        self.context = 5 if model_id == "unifmir-tribolium" else 1
        self.models_dir = Path(models_dir or os.environ.get("CIDENOISE_MODELS", ROOT / "models"))
        self.loaded = False
        self.embeddings = {}
        self.provenance = {"model": model_id, "device": self.device, "context_z": self.context}
        self.provenance["precision"] = self.precision

    def load(self):
        if self.loaded:
            return
        import torch
        start = time.perf_counter()
        if self.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            budget = min(9 * 1024**3, total * .8)
            torch.cuda.set_per_process_memory_fraction(budget / total)
            self.provenance["cuda_allocator_budget_gib"] = budget / 1024**3
        manifest = json.loads((ROOT / "model_manifest.json").read_text())
        entry = manifest["models"][self.model_id]
        checkpoint = self.models_dir / entry["checkpoint"]
        if not checkpoint.is_file():
            if entry.get("local_only"):
                raise FileNotFoundError(f"Missing local confocal checkpoint {checkpoint}. Copy the pinned best.pt from the training run; see training/README.md.")
            raise FileNotFoundError(f"Missing {checkpoint}. Run python tools/download_models.py first.")
        digest = sha256(checkpoint)
        if digest != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        if self.model_id == "fluoresfm":
            from .vendor.fluoresfm.unet_sd_c import UNetModel
            self.model = UNetModel(in_channels=1, out_channels=1, channels=320,
                n_res_blocks=1, attention_levels=[0, 1, 2, 3], channel_multipliers=[1, 2, 4, 4],
                n_heads=8, tf_layers=1, d_cond=768, pixel_shuffle=False, scale_factor=4)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"]
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        elif self.model_id.startswith("noise2noise-"):
            from .vendor.instant import Noise2Noise
            self.model = Noise2Noise()
            if self.model_id == "noise2noise-confocal":
                trained = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if trained["architecture"] != "instant-noise2noise-v1":
                    raise ValueError("Unexpected confocal checkpoint architecture")
                state = trained["model_state_dict"]
                self.provenance["training"] = entry["training"]
            else:
                self.model.load_published(checkpoint)
                state = self.model.state_dict()
        elif self.model_id.startswith("cellpose-"):
            from .vendor.cellpose.resnet_torch import CPnet
            self.model = CPnet([1,32,64,128,256], nout=1, sz=3)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.provenance["spatial_scale"] = 1.0
        else:
            from .vendor.unifmir.swinir import swinir
            self.model = swinir(upscale=1, in_chans=self.context)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(self.device).eval()
        if self.model_id == "fluoresfm":
            from .attention import install
            install(self.model)
            self.provenance["attention"] = "torch_scaled_dot_product_attention"
        self.provenance.update(checkpoint=entry["checkpoint"], checkpoint_sha256=digest,
            upstream_revision=entry["upstream_revision"], normalization=entry["normalization"],
            load_seconds=time.perf_counter() - start)
        self.loaded = True

    def set_structure(self, structure=""):
        self.load()
        if self.model_id != "fluoresfm":
            return
        import torch
        prompt = "Task: denoising" + (f"; structure: {structure}" if structure else "")
        if prompt not in self.embeddings:
            manifest = json.loads((ROOT / "model_manifest.json").read_text())
            assets = [a for a in manifest["assets"] if a["path"].startswith("biomedclip/")]
            for asset in assets:
                if sha256(self.models_dir / asset["path"]) != asset["sha256"]:
                    raise ValueError(f"Text encoder asset checksum mismatch: {asset['path']}")
            self.provenance["text_encoder_assets"] = {a["path"]:a["sha256"] for a in assets}
            from .vendor.fluoresfm.biomedclip_embedder import BiomedCLIPTextEmbedder
            # Instantiate from local configuration; no pretrained BERT fetch is needed.
            from open_clip.factory import _MODEL_CONFIGS
            config = json.loads((self.models_dir / "biomedclip/open_clip_config.json").read_text())
            cfg = config["model_cfg"]
            local_bert = str((self.models_dir / "biomedclip/bert").resolve())
            cfg["text_cfg"].update(hf_model_name=local_bert, hf_tokenizer_name=local_bert, hf_model_pretrained=False)
            _MODEL_CONFIGS["biomedclip_local"] = cfg
            embedder = BiomedCLIPTextEmbedder(
                path_json=str(self.models_dir / "biomedclip/open_clip_config.json"),
                path_bin=str(self.models_dir / "biomedclip/open_clip_pytorch_model.bin"),
                context_length=160, device=torch.device("cpu")).eval()
            with torch.inference_mode():
                self.embeddings[prompt] = embedder(prompt).to(self.device)
            del embedder
        self.embedding = self.embeddings[prompt]
        self.provenance["prompt"] = prompt

    def predict(self, batch):
        import torch
        self.load()
        if self.model_id == "fluoresfm" and not hasattr(self, "embedding"):
            self.set_structure()
        try:
            with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.precision == "float16"):
                x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
                result = self.model(x, None, self.embedding.expand(len(x), -1, -1)) if self.model_id == "fluoresfm" else self.model(x)
                if self.model_id.startswith("cellpose-"):
                    result = result[0]
                return result.float().cpu().numpy()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU memory exhausted. Reduce batch size or tile size; no alternate model was used.") from exc
