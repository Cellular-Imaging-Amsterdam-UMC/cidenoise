"""Download only the five published FMD confocal archives, with resume and checksums."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request
import uuid

SOURCES = Path(__file__).with_name("confocal_sources.json")
DEFAULT_ROOT = Path("F:/noise2noise_data")


def digest(path, algorithm="md5"):
    h=hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):
            h.update(block)
    return h.hexdigest()


def download(item, folder):
    target=folder/item["name"]
    expected=item["computed_md5"]
    if target.exists():
        if target.stat().st_size!=item["size"] or digest(target)!=expected:
            raise ValueError(f"Existing archive is damaged: {target}; move it aside before retrying")
        print(f"Verified existing {target.name}",flush=True)
        return target
    partial=target.with_suffix(".tar.part")
    for attempt in range(4):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>item["size"]:
            raise ValueError(f"Oversized partial download: {partial}")
        if offset==item["size"]:
            break
        request=urllib.request.Request(item["download_url"],headers={
            "User-Agent":"CIDenoise-confocal-training/1.0", "Accept-Encoding":"identity",
            "Range":f"bytes={offset}-"})
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status==206:
                    if not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected byte range")
                    mode="ab" if offset else "wb"
                elif response.status==200:
                    mode="wb";offset=0
                else:
                    raise ValueError(f"Unexpected HTTP response: {response.status}")
                print(f"Downloading {target.name}: resume at {offset/1e6:.1f} MB",flush=True)
                last=time.monotonic()
                with partial.open(mode) as stream:
                    while block:=response.read(4*1024**2):
                        stream.write(block);offset+=len(block)
                        if offset>item["size"]:
                            raise ValueError("Download exceeds published size")
                        if time.monotonic()-last>=15:
                            print(f"{target.name}: {offset/item['size']:.1%}",flush=True);last=time.monotonic()
            if partial.stat().st_size==item["size"]:
                break
        except (OSError,TimeoutError) as exc:
            if attempt==3:
                raise
            print(f"Retrying {target.name}: {exc}",flush=True)
            time.sleep(2*(attempt+1))
    if not partial.exists() or partial.stat().st_size!=item["size"] or digest(partial)!=expected:
        raise ValueError(f"Download checksum/size failed: {partial}")
    partial.replace(target)
    print(f"Checksum verified: {target.name}",flush=True)
    return target


def extract(archive, root, item):
    name=item["name"].removesuffix(".tar")
    destination=root/name
    marker=destination/".complete.json"
    if marker.exists():
        if json.loads(marker.read_text())["archive_md5"]!=item["computed_md5"]:
            raise ValueError(f"Extraction source differs: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Unverified existing extraction: {destination}")
    temp=root/("."+name+"."+uuid.uuid4().hex+".partial")
    temp.mkdir(parents=True)
    count=0
    try:
        with tarfile.open(archive,"r:*") as tar:
            for member in tar:
                path=PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError("Archive links are unsupported")
                if not member.isfile():
                    continue
                if len(path.parts)<3 or path.parts[0]!=name or path.parts[1] not in ("raw","gt"):
                    continue
                if path.suffix.lower()!=".png":
                    continue
                target=temp.joinpath(*path.parts[1:])
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open("xb") as output:
                    shutil.copyfileobj(source,output,1024**2)
                count+=1
        if count==0:
            raise ValueError(f"No raw/GT PNG files found in {archive}")
        (temp/".complete.json").write_text(json.dumps(dict(archive_md5=item["computed_md5"],files=count)))
        # Indexers/antivirus can briefly hold newly written folders on Windows.
        for attempt in range(8):
            if destination.exists():
                raise FileExistsError(destination)
            try:
                temp.rename(destination)
                break
            except PermissionError:
                if attempt==7:
                    raise
                time.sleep(.2*2**min(attempt,3))
    except BaseException:
        if temp.exists() and temp.resolve().parent==root.resolve():
            shutil.rmtree(temp)
        raise
    print(f"Extracted {name}: {count} raw/reference PNG files",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    if not 1<=args.workers<=4:
        raise ValueError("Download workers must be 1..4")
    args.data_root.mkdir(parents=True,exist_ok=True)
    archives=args.data_root/"archives";archives.mkdir(exist_ok=True)
    images=args.data_root/"FMD";images.mkdir(exist_ok=True)
    metadata=json.loads(SOURCES.read_text())
    (args.data_root/"sources.json").write_text(json.dumps(metadata,indent=2)+"\n")
    def prepare(item):
        if not item["name"].startswith("Confocal_"):
            raise ValueError("Non-confocal archive rejected")
        archive=download(item,archives)
        extract(archive,images,item)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(prepare,metadata["files"]))
    print(f"All five confocal archives verified and extracted to {images}",flush=True)


if __name__=="__main__":
    main()
