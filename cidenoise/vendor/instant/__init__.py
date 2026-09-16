"""PyTorch inference port of Mannam's pretrained Noise2Noise (GPL-3.0).

Architecture: ND-HowardGroup/Instant-Image-Denoising, revision
c885aee8adb62bcd2d5c3862273d4ea0f68b3068. See LICENSE and THIRD_PARTY.md.
No training code or parameter optimization is included.
"""
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ArrayUnpickler(pickle.Unpickler):
    """The published .h5 is actually a pickle of NumPy weights, not HDF5."""
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unexpected checkpoint object: {module}.{name}")
        return allowed[module, name]


class Noise2Noise(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [(1,48)] + [(48,48)] * 6 + [(96,96),(96,96)]
        channels += [(144,96),(96,96)] * 3 + [(97,64),(64,32),(32,1)]
        self.convs = nn.ModuleList([nn.Conv2d(i,o,3,padding=1,bias=False) for i,o in channels])

    def load_published(self, checkpoint):
        with open(checkpoint, "rb") as stream:
            layers = ArrayUnpickler(stream).load()
        weights = [(name, w) for name,w in layers if w]
        if len(weights) != len(self.convs):
            raise ValueError("Unexpected Noise2Noise layer count")
        with torch.no_grad():
            for index, (conv, (name, arrays)) in enumerate(zip(self.convs, weights)):
                expected = "conv2d" + (f"_{index}" if index else "")
                if name != expected or len(arrays) != 1:
                    raise ValueError(f"Unexpected Noise2Noise layer {name}")
                weight = torch.from_numpy(arrays[0].transpose(3,2,0,1).copy())
                if weight.shape != conv.weight.shape:
                    raise ValueError(f"Unexpected checkpoint dimensions: {name}")
                conv.weight.copy_(weight)

    def forward(self, x):
        def conv(i, a):
            return F.leaky_relu(self.convs[i](a), .1)
        h = conv(1, conv(0, x))
        pools = [F.max_pool2d(h, 2)]
        for i in range(2, 6):
            pools.append(F.max_pool2d(conv(i, pools[-1]), 2))
        h = conv(6, pools[-1])
        for j, skip in enumerate(reversed(pools[:-1])):
            h = torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), skip], dim=1)
            h = conv(8+2*j, conv(7+2*j, h))
        h = torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), x], dim=1)
        return torch.tanh(self.convs[17](conv(16, conv(15, h))))
