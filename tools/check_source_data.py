"""Record/verify checksums without modifying local input stores."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    target=ROOT / "outputs/validation/localdata-checksums.json"
    digests={}
    for path in sorted((ROOT / "localdata").rglob("*")):
        if path.is_file():
            h=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):
                    h.update(block)
            digests[str(path.relative_to(ROOT / "localdata"))]=h.hexdigest()
    if args.check:
        if json.loads(target.read_text()) != digests:
            raise RuntimeError("Localdata checksums differ from recorded snapshot")
        print(f"All {len(digests)} input files unchanged")
    else:
        if target.exists():
            raise FileExistsError("Input checksum snapshot already exists; use --check")
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(digests,indent=2))
        print(f"Recorded {len(digests)} input files")


if __name__ == "__main__":
    main()
