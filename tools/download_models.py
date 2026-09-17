"""Download pinned assets, verify checksums, and extract only inference files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cidenoise.adapters import sha256


def fetch(url, path, expected=None, algorithm="sha256"):
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid():
        if not path.is_file():
            return False
        if not expected:
            return True
        h = hashlib.new(algorithm)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest() == expected
    if valid():
        return
    print(f"Downloading {url}", flush=True)
    partial = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "CIDenoise/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, 8 * 1024 * 1024)
    partial.replace(path)
    if not valid():
        raise ValueError(f"Checksum mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model_manifest.json").read_text())
    assets = manifest["assets"]
    missing = [a for a in assets if not (args.models_dir / a["path"]).is_file() or sha256(args.models_dir / a["path"]) != a["sha256"]]
    if args.verify_only and missing:
        raise RuntimeError("Missing or damaged model assets: " + ", ".join(a["path"] for a in missing))
    archives = {}
    try:
        for asset in missing:
            target = args.models_dir / asset["path"]
            if asset.get("local_only"):
                raise FileNotFoundError(f"Local trained checkpoint required at {target}; copy pinned best.pt as documented in training/README.md. No public download is available.")
            if "url" in asset:
                fetch(asset["url"], target, asset["sha256"])
            else:
                archive_id = asset["archive"]
                archive = manifest["archives"][archive_id]
                if archive_id not in archives:
                    cached = ROOT / ".cache/upstream" / archive["filename"]
                    fetch(archive["url"], cached, archive["checksum"], archive["algorithm"])
                    archives[archive_id] = zipfile.ZipFile(cached) if cached.suffix == ".zip" else tarfile.open(cached)
                opened = archives[archive_id]
                reader = opened.open(asset["member"]) if isinstance(opened, zipfile.ZipFile) else opened.extractfile(asset["member"])
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".download")
                with reader, partial.open("wb") as stream:
                    shutil.copyfileobj(reader, stream, 8 * 1024 * 1024)
                if sha256(partial) != asset["sha256"]:
                    raise ValueError(f"Extracted checksum mismatch: {asset['path']}")
                partial.replace(target)
        print(f"Verified {len(assets)} assets in {args.models_dir}")
        args.models_dir.mkdir(parents=True, exist_ok=True)
        (args.models_dir / ".complete.json").write_text(json.dumps({"manifest_sha256": sha256(ROOT / "model_manifest.json")}, indent=2))
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == "__main__":
    main()
