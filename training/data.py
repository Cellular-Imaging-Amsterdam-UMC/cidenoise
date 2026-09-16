"""FMD confocal repeats with field-level splits and independent noisy targets."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

CATEGORIES=("Confocal_BPAE_B","Confocal_BPAE_G","Confocal_BPAE_R","Confocal_FISH","Confocal_MICE")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            h.update(block)
    return h.hexdigest()


def read_frame(path):
    with Image.open(path) as image:
        if image.mode!="L":
            raise ValueError(f"Expected original 8-bit grayscale FMD PNG: {path}")
        return np.array(image)


def build_index(root):
    root=Path(root)
    records=[]
    for category in CATEGORIES:
        folder=root/"FMD"/category/"raw"
        fovs=sorted(p.name for p in folder.iterdir() if p.is_dir())
        if set(fovs)!={str(i) for i in range(1,21)}:
            raise ValueError(f"Expected all 20 fields in {folder}")
        for fov in range(1,21):
            files=sorted((folder/str(fov)).glob("*.png"))
            if len(files)!=50:
                raise ValueError(f"Expected 50 raw repeats: {category}/{fov}, found {len(files)}")
            frames=[]
            for file in files:
                array=read_frame(file)
                if array.shape!=(512,512):
                    raise ValueError(f"Unexpected raw dimensions: {file}")
                frames.append(dict(path=file.relative_to(root).as_posix(),sha256=sha256(file)))
            # Preserve the original benchmark's FOV 19 test set; use 20 for validation.
            split="test" if fov==19 else "validation" if fov==20 else "train"
            records.append(dict(category=category,fov=fov,split=split,frames=frames))
    result=dict(format_version=1,normalization="uint8 / 255 - 0.5",shape=[512,512],
        split_rule="Train FOV 1..18, validation 20, test 19, consistently across all categories/channels",
        training_targets="Distinct raw captures only; GT and pre-averaged exports are excluded",
        sources_sha256=sha256(root/"sources.json"),records=records)
    target=root/"index.json"
    text=json.dumps(result,indent=2)+"\n"
    if target.exists():
        if target.read_text()!=text:
            raise ValueError("Existing index differs: choose a new data directory or inspect the changed data")
    else:
        temporary=target.with_name(f".index.{uuid.uuid4().hex}.writing")
        temporary.write_text(text)
        temporary.replace(target)
    return result


def verify_index(root, index):
    root=Path(root).resolve()
    for record in index["records"]:
        if record["category"] not in CATEGORIES:
            raise ValueError("Non-confocal category rejected")
        expected="test" if record["fov"]==19 else "validation" if record["fov"]==20 else "train"
        if record["split"]!=expected:
            raise ValueError("Dataset split has changed")
        for frame in record["frames"]:
            path=(root/frame["path"]).resolve()
            relative=Path(frame["path"])
            if relative.parts[:4]!=("FMD",record["category"],"raw",str(record["fov"])):
                raise ValueError("Only raw captures from their declared field are allowed")
            if root not in path.parents or sha256(path)!=frame["sha256"]:
                raise ValueError(f"Missing, changed or invalid training image: {frame['path']}")


class NoisyPairs(Dataset):
    def __init__(self,root,index,split="train",patch=256,seed=42,averages=(1,2,4)):
        self.root=Path(root)
        self.patch=patch
        self.seed=seed
        self.split=split
        self.averages=tuple(averages)
        self.groups=defaultdict(list)
        for record in index["records"]:
            if record["split"]==split:
                self.groups[record["category"]].append(record)
        self.categories=sorted(self.groups)
        if not self.categories or patch<32 or patch>512 or patch%32:
            raise ValueError("Nonempty split and patch 32..512 divisible by 32 required")
        if not self.averages or any(k not in (1,2,4,8,16) for k in self.averages):
            raise ValueError("Averaging counts must be selected from 1,2,4,8,16")
        self._read=lru_cache(maxsize=64)(read_frame)

    def __getstate__(self):
        state=dict(self.__dict__);state.pop("_read",None)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state);self._read=lru_cache(maxsize=64)(read_frame)

    def __len__(self):
        return sum(len(g) for g in self.groups.values())*50

    def selection(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(index)]))
        category=self.categories[int(index)%len(self.categories)]
        records=self.groups[category]
        record=records[int(rng.integers(len(records)))]
        count=int(rng.choice(self.averages))
        chosen=rng.choice(len(record["frames"]),2*count,replace=False)
        y,x=rng.integers(0,513-self.patch,size=2)
        rotation=int(rng.integers(4)) if self.split=="train" else 0
        flip=bool(rng.integers(2)) if self.split=="train" else False
        return record,chosen[:count],chosen[count:],int(y),int(x),rotation,flip

    def __getitem__(self,index):
        record,a,b,y,x,rotation,flip=self.selection(index)
        def average(indices):
            result=np.zeros((self.patch,self.patch),np.float32)
            for i in indices:
                data=self._read(str(self.root/record["frames"][int(i)]["path"]))
                result+=data[y:y+self.patch,x:x+self.patch]
            return result/len(indices)/255.0-.5
        # Neither normalization nor augmentation is estimated from the noisy target.
        pair=np.stack([average(a),average(b)])
        pair=np.rot90(pair,rotation,axes=(-2,-1))
        if flip:
            pair=pair[:,:,::-1]
        pair=torch.from_numpy(pair.copy())
        return pair[:1],pair[1:]


class EpochSamples(Sampler):
    """Stateless sample seeds make worker counts and epoch-boundary resume predictable."""
    def __init__(self,size):
        self.size=size;self.epoch=0;self.skip=0

    def __iter__(self):
        return iter(range(self.epoch*self.size+self.skip,(self.epoch+1)*self.size))

    def __len__(self):
        return self.size-self.skip
