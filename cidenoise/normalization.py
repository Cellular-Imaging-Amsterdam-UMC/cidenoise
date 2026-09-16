"""Model-specific normalization, with bounded-memory stack percentiles."""
import numpy as np


def histogram_percentile(counts, percentile, offset=0):
    cumulative = np.cumsum(counts)
    n = int(cumulative[-1])
    rank = percentile / 100 * (n - 1)
    lo, hi = int(np.floor(rank)), int(np.ceil(rank))
    a, b = np.searchsorted(cumulative, [lo + 1, hi + 1]) + offset
    return float(a + (b - a) * (rank - lo))


def stack_bounds(image, t, c):
    dtype = image.array.dtype
    if dtype.kind in "ui" and dtype.itemsize <= 2:
        lim = np.iinfo(dtype)
        counts = np.zeros(lim.max - lim.min + 1, np.int64)
        for z in range(image.length("z")):
            plane = image.plane(t, c, z).astype(np.int64)
            counts += np.bincount((plane - lim.min).ravel(), minlength=len(counts))
        return histogram_percentile(counts, 2, lim.min), histogram_percentile(counts, 99.8, lim.min), "exact_stack_histogram"
    # Float/wide-integer stacks: deterministic, bounded sampling over every Z plane.
    per_plane = max(1, 1_000_000 // image.length("z"))
    samples = []
    for z in range(image.length("z")):
        plane = image.plane(t, c, z).ravel()
        if not np.isfinite(plane).all():
            raise ValueError("Input contains NaN or infinite intensities")
        samples.append(plane[np.linspace(0, len(plane) - 1, min(per_plane, len(plane)), dtype=int)])
    lo, hi = np.percentile(np.concatenate(samples), [2, 99.8])
    return float(lo), float(hi), "deterministic_stack_sample_max_1M"


def plane_bounds(plane):
    plane = np.maximum(plane, 0)
    lo, hi = np.percentile(plane, [3, 99.5])
    if hi == 0:
        hi = plane.max()
    return float(lo), float(hi), "exact_plane_percentile"
