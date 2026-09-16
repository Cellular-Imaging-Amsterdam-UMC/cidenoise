"""Build content-addressed model/runtime layers and versioned application image."""
from pathlib import Path
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def exists(image):
    return subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    run(sys.executable, "tools/download_models.py", "--verify-only")
    model_hash = hashlib.sha256((ROOT / "model_manifest.json").read_bytes() + (ROOT / "Dockerfile.models").read_bytes()).hexdigest()[:20]
    model = f"w_cidenoise-model-cache:{model_hash}"
    if not exists(model):
        run("docker", "build", "-f", "Dockerfile.models", "-t", model, "-t", "w_cidenoise-model-cache:latest", ".")
    runtime_hash = hashlib.sha256(model.encode() + (ROOT / "requirements.txt").read_bytes() + (ROOT / "requirements-lock-linux.txt").read_bytes() + (ROOT / "Dockerfile.runtime").read_bytes()).hexdigest()[:20]
    runtime = f"w_cidenoise-runtime-cache:{runtime_hash}"
    if not exists(runtime):
        run("docker", "build", "-f", "Dockerfile.runtime", "--build-arg", f"MODEL_CACHE_IMAGE={model}", "-t", runtime, "-t", "w_cidenoise-runtime-cache:latest", ".")
    version = (ROOT / "version.txt").read_text().strip()
    run("docker", "build", "--build-arg", f"RUNTIME_CACHE_IMAGE={runtime}", "-t", f"w_cidenoise:{version}", "-t", "w_cidenoise:latest", ".")


if __name__ == "__main__":
    main()
