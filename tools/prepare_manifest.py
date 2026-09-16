"""Maintainer bootstrap: reproduce manifest from downloaded upstream release assets."""
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cidenoise.adapters import sha256
from tools.download_models import fetch


def main():
    modeldir = ROOT / "models"
    revision = "dc90f055db7434078ef68b8c1a1fba97af6dbbc0"
    models = {}
    for identifier, checkpoint, norm, rev in [
        ("fluoresfm", "fluoresfm/epoch_0_iter_700000.pt", "per-plane p3,p99.5; nonnegative input; no normalized clipping", "2fded7c31be8c52476de89934ced9093ebe2c307"),
        ("unifmir-planaria", "unifmir/planaria.pt", "per-channel stack p2,p99.8; no normalized clipping", revision),
        ("unifmir-tribolium", "unifmir/tribolium.pt", "per-channel stack p2,p99.8; five Z planes", revision),
    ]:
        models[identifier] = dict(checkpoint=checkpoint, sha256=sha256(modeldir / checkpoint), normalization=norm,
            upstream_revision=rev, license="MIT" if identifier == "fluoresfm" else "GPL-3.0")
    assets = []
    for path, member in [
        ("fluoresfm/epoch_0_iter_700000.pt", "example/checkpoints/fluoresfm/epoch_0_iter_700000.pt"),
        ("biomedclip/open_clip_config.json", "example/checkpoints/biomedclip/open_clip_config.json"),
        ("biomedclip/open_clip_pytorch_model.bin", "example/checkpoints/biomedclip/open_clip_pytorch_model.bin"),
        ("unifmir/planaria.pt", "experiment/SwinIRDenoising_Planaria/model_best15.pt"),
        ("unifmir/tribolium.pt", "experiment/SwinIRmto1Denoising_Tribolium/model_best.pt"),
    ]:
        assets.append(dict(path=path, sha256=sha256(modeldir / path), archive="unifmir" if path.startswith("unifmir") else "fluoresfm", member=member))
    repo = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    with urllib.request.urlopen("https://huggingface.co/api/models/" + repo) as response:
        info = json.load(response)
    bert_revision = info["sha"]
    names = {f["rfilename"] for f in info["siblings"]}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "special_tokens_map.json"):
        if name not in names:
            continue
        url = f"https://huggingface.co/{repo}/resolve/{bert_revision}/{name}"
        path = f"biomedclip/bert/{name}"
        fetch(url, modeldir / path)
        assets.append(dict(path=path, sha256=sha256(modeldir / path), url=url))
    manifest = dict(format_version=1, models=models, assets=assets,
        text_encoder=dict(model="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", license="MIT", tokenizer_revision=bert_revision),
        archives={
            "fluoresfm": dict(filename="fluoresfm-example.zip", url="https://zenodo.org/api/records/18382702/files/example.zip/content", checksum="98ca344be1834714ab72606d9228f9dd", algorithm="md5"),
            "unifmir": dict(filename="model.tgz", url="https://github.com/cxm12/UNiFMIR/releases/download/2023.10.05/model.tgz", checksum=sha256(ROOT / ".cache/upstream/model.tgz"), algorithm="sha256")})
    (ROOT / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Manifest prepared")


if __name__ == "__main__":
    main()
