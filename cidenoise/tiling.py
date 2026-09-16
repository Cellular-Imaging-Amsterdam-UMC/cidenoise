"""Overlapping bounded inference; each call receives B, context-Z, Y, X."""
import numpy as np


def reflected_index(index, length):
    if length == 1:
        return 0
    index %= 2 * length - 2
    return index if index < length else 2 * length - 2 - index


def starts(length, tile, overlap):
    positions = list(range(0, max(1, length - tile + 1), tile - overlap))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def predict_plane(read_patch, shape, predict, tile=64, overlap=16, batch_size=4):
    if tile < 8 or tile % 8 or not 0 <= overlap < tile or batch_size < 1:
        raise ValueError("Tile must be a positive multiple of 8; overlap < tile; batch size >= 1")
    h, w = shape
    output = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    ramp = np.ones(tile, np.float32)
    if overlap:
        fade = np.linspace(1 / (overlap + 1), 1, overlap, dtype=np.float32)
        ramp[:overlap] *= fade
        ramp[-overlap:] *= fade[::-1]
    mask = ramp[:, None] * ramp[None, :]
    positions = [(y, x) for y in starts(h, tile, overlap) for x in starts(w, tile, overlap)]
    for begin in range(0, len(positions), batch_size):
        batch_positions = positions[begin:begin + batch_size]
        patches = []
        for y, x in batch_positions:
            patch = np.asarray(read_patch(y, min(y + tile, h), x, min(x + tile, w)), dtype=np.float32)
            patch = np.pad(patch, ((0, 0), (0, tile - patch.shape[-2]), (0, tile - patch.shape[-1])), mode="reflect")
            patches.append(patch)
        predictions = np.asarray(predict(np.stack(patches)), dtype=np.float32)
        if predictions.shape != (len(patches), 1, tile, tile) or not np.isfinite(predictions).all():
            raise ValueError(f"Model returned invalid predictions: {predictions.shape}")
        for (y, x), prediction in zip(batch_positions, predictions):
            ph, pw = min(tile, h - y), min(tile, w - x)
            output[y:y + ph, x:x + pw] += prediction[0, :ph, :pw] * mask[:ph, :pw]
            weights[y:y + ph, x:x + pw] += mask[:ph, :pw]
    return output / weights
