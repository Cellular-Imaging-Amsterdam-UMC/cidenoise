"""Frozen two-pair benchmark with a static, local review report."""
import argparse
from dataclasses import asdict
import html
import json
import logging
from pathlib import Path
import time
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from .adapters import Adapter, MODEL_IDS, sha256, ROOT
from .engine import Settings, run_store, output_name
from .ome_zarr import open_images


def volume(image, channel):
    return np.stack([image.plane(0, channel, z) for z in range(image.length("z"))])


def paired(localdata, name):
    _, raw = open_images(localdata / f"{name}.ome.zarr")
    _, reference = open_images(localdata / f"{name}-dn.ome.zarr")
    if len(raw) != 1 or len(reference) != 1:
        raise ValueError("Local benchmark expects ordinary images")
    a, b = raw[0], reference[0]
    if a.axes != b.axes or a.array.shape != b.array.shape or a.datasets != b.datasets:
        raise ValueError(f"{name}: reference shape/coordinates differ")
    return a, b


def alignment(raw, reference):
    shifts = []
    for z in sorted(set([0, len(raw)//2, len(raw)-1])):
        if np.std(raw[z]) == 0 or np.std(reference[z]) == 0:
            continue
        shift, _, _ = phase_cross_correlation(raw[z], reference[z], upsample_factor=10, normalization=None)
        shifts.append([float(v) for v in shift])
    return dict(shifts_yx=shifts, accepted=bool(shifts) and max(abs(v) for s in shifts for v in s) <= 1)


def calibrate(localdata, settings, output):
    raw, ref = paired(localdata, "brain1")
    entries = []
    for c in range(raw.length("c")):
        x, y = volume(ref, c), volume(raw, c)
        align = alignment(y, x)
        if not align["accepted"]:
            raise ValueError(f"brain1 C{c+1}: reference alignment failed: {align}")
        xs, ys = x[:, ::8, ::8].ravel().astype(np.float64), y[:, ::8, ::8].ravel().astype(np.float64)
        slope, intercept = np.linalg.lstsq(np.column_stack([xs, np.ones_like(xs)]), ys, rcond=None)[0]
        lo, hi = np.percentile(y, [0, 99.9])
        entries.append(dict(channel=c + 1, slope=float(slope), intercept=float(intercept),
            data_range=max(float(hi-lo), 1), display_low=float(lo), display_high=float(hi), alignment=align))
    frozen = dict(calibration_source="brain1 only", settings=asdict(settings), channels=entries,
        manifest_sha256=sha256(ROOT / "model_manifest.json"),
        mapping="LAS-X in raw units = slope * LAS-X + intercept; fitted on brain1, fixed for brain2",
        caveat="LAS-X is a processed reference, not independent biological ground truth.")
    output.write_text(json.dumps(frozen, indent=2))
    return frozen


def metrics(data, raw, reference, calibration, aligned):
    dr = calibration["data_range"]
    reference_scaled = reference * calibration["slope"] + calibration["intercept"]
    error = data.astype(np.float64) - reference_scaled
    mse = float(np.mean(error ** 2))
    psnr = float(10 * np.log10(dr * dr / max(mse, 1e-30))) if aligned else None
    ssim = float(np.mean([structural_similarity(a, b, data_range=dr) for a,b in zip(data, reference_scaled)])) if aligned else None
    return dict(mean=float(data.mean()), minimum=float(data.min()), maximum=float(data.max()),
        percentiles=[float(v) for v in np.percentile(data, [1, 50, 99, 99.9])],
        mean_change_vs_raw=float(data.mean()-raw.mean()),
        raw_units_mae_vs_unscaled_lasx=float(np.mean(np.abs(data-reference))),
        mae_vs_scaled_lasx=float(np.mean(np.abs(error))) if aligned else None,
        psnr_vs_scaled_lasx=psnr, ssim_vs_scaled_lasx=ssim,
        z_difference_mean=float(np.mean(np.abs(np.diff(data, axis=0)))) if len(data)>1 else None)


def grey(data, lo, hi, size):
    data = (np.clip((data-lo)/max(hi-lo, 1e-10), 0, 1)*255).astype(np.uint8)
    return PILImage.fromarray(data).convert("RGB").resize(size, PILImage.Resampling.NEAREST)


def gallery(path, arrays, calibration):
    cell = 256
    labels = list(arrays)
    canvas = PILImage.new("RGB", (150 + cell * len(arrays), 48 + 5*(cell+24)), "#161b22")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    row_names = ["Central XY", "Central XY crop", "XZ section", "Z maximum", "Residual vs raw"]
    raw = arrays["Raw"]
    z, h, w = raw.shape
    low, high = calibration["display_low"], calibration["display_high"]
    for column, (label, data) in enumerate(arrays.items()):
        x = 150 + column*cell
        draw.text((x+8, 14), label, fill="white", font=font)
        crop = data[z//2, max(0,h//2-128):h//2+128, max(0,w//2-128):w//2+128]
        views = [data[z//2], crop, data[:, h//2, :], data.max(axis=0), data[z//2]-raw[z//2]]
        for row, view in enumerate(views):
            y = 48 + row*(cell+24)
            if column == 0:
                draw.text((8,y+8), row_names[row], fill="white", font=font)
            if row == 4:
                lim = max((high-low)*0.25, 1)
                pixels = grey(view, -lim, lim, (cell, cell))
            else:
                pixels = grey(view, low, high, (cell, cell))
            canvas.paste(pixels, (x,y))
    canvas.save(path)


def report(localdata, output, frozen, models):
    reportdir = output / "report"
    reportdir.mkdir(exist_ok=True)
    all_metrics = []
    sections = []
    performance = []
    for name in ("brain1", "brain2"):
        raw_image, ref_image = paired(localdata, name)
        model_images = {}
        for model in models:
            store = output / model / output_name(f"{name}.ome.zarr")
            if not store.exists():
                raise FileNotFoundError(f"Missing benchmark output {store}; run inference first")
            root, images = open_images(store)
            expected = dict(frozen["settings"], model=model)
            if root.attrs["cidenoise"]["settings"] != expected:
                raise ValueError(f"Output settings differ from frozen settings: {store}")
            model_images[model] = (images[0], root.attrs["cidenoise"])
            provenance = root.attrs["cidenoise"]
            performance.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                [name, model, f"{provenance['runtime_seconds'] / 60:.2f}",
                 f"{provenance['peak_gpu_bytes'] / 1024**3:.2f}"]) + "</tr>")
        for c, calibration in enumerate(frozen["channels"]):
            raw, reference = volume(raw_image,c), volume(ref_image,c)
            align = alignment(raw, reference)
            arrays = {"Raw": raw, "LAS-X scaled": reference*calibration["slope"]+calibration["intercept"]}
            rows = []
            for label, data, provenance in [("Raw",raw,{})] + [(m,volume(v[0],c),v[1]) for m,v in model_images.items()]:
                if label != "Raw":
                    arrays[label] = data
                values = metrics(data,raw,reference,calibration,align["accepted"])
                values.update(image=name, channel=c+1, model=label, alignment=align,
                    runtime_seconds=provenance.get("runtime_seconds"), peak_gpu_bytes=provenance.get("peak_gpu_bytes"),
                    clipped_pixels=sum(p["clipped_pixels"] for p in provenance.get("planes",[]) if p["channel"]==c+1))
                all_metrics.append(values)
                def fmt(v): return "not assessed" if v is None else f"{v:.4g}"
                rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                    [label, fmt(values["mean"]), fmt(values["mean_change_vs_raw"]),
                     fmt(values["raw_units_mae_vs_unscaled_lasx"]), fmt(values["mae_vs_scaled_lasx"]),
                     fmt(values["psnr_vs_scaled_lasx"]), fmt(values["ssim_vs_scaled_lasx"]),
                     fmt(values["z_difference_mean"]), values["clipped_pixels"]]) + "</tr>")
            filename = f"{name}_channel_{c+1}.png"
            gallery(reportdir / filename, arrays, calibration)
            sections.append(f'<section><h2>{name}, channel {c+1}</h2><p>Alignment: {html.escape(str(align))}</p><div class="table"><table><tr><th>Model</th><th>Mean</th><th>Mean change</th><th>MAE to original LAS-X</th><th>MAE to scaled LAS-X</th><th>PSNR to scaled LAS-X</th><th>SSIM to scaled LAS-X</th><th>Adjacent-Z MAE</th><th>Clipped pixels</th></tr>{"".join(rows)}</table></div><a href="{filename}"><img src="{filename}" alt="Matched image comparison"></a></section>')
    (reportdir / "metrics.json").write_text(json.dumps(all_metrics, indent=2, allow_nan=False))
    (reportdir / "index.html").write_text('''<!doctype html><meta charset="utf-8"><title>CIDenoise comparison</title>
<style>body{font:16px system-ui;background:#10151c;color:#e6edf3;margin:32px}a{color:#79c0ff}img{max-width:100%;height:auto}table{border-collapse:collapse;margin:16px 0}.table{overflow-x:auto}td,th{padding:8px 16px;border:1px solid #445}section{margin:48px 0}pre{white-space:pre-wrap}</style>
<h1>CIDenoise / Leica LAS-X comparison</h1><p>LAS-X is a processed reference, not ground truth. No automatic winner is assigned. Brightness mapping and display ranges were fitted on brain1 and frozen before brain2. Panels share those ranges; XZ views are vertically stretched for inspection, not shown at physical aspect ratio. Residual gray midpoint means zero, with limits ±25% of the display range.</p>
<p>Review weak puncta, fine processes, background texture, removed structures and introduced structures. Smoothness and similarity alone do not establish biological accuracy. Runtime and GPU memory in metrics.json are per complete store, not per channel.</p>
<p>Adjacent-Z MAE measures intensity differences between neighboring planes; true anatomical differences also contribute. It is not a standalone measure of denoising quality. The JSON includes intensity percentiles (1, 50, 99, 99.9) and metric ranges are fixed by the calibration below.</p>
<p><a href="metrics.json">All metrics and intensity distributions</a></p><h2>Performance</h2><p>Elapsed time includes image I/O and pyramid generation, excludes model loading. Peak memory is PyTorch-allocated GPU memory.</p>
<table><tr><th>Image</th><th>Model</th><th>Minutes</th><th>Peak GPU GiB</th></tr>''' + "".join(performance) + "</table><h2>Frozen calibration and settings</h2><pre>" + html.escape(json.dumps(frozen,indent=2)) + "</pre>" + "".join(sections), encoding="utf-8")
    return reportdir / "index.html"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--localdata", type=Path, default=ROOT / "localdata")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/benchmark")
    p.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    p.add_argument("--phase", choices=("all", "infer", "report"), default="all")
    p.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = Settings(device=args.device, tile_size=args.tile_size, overlap=args.overlap, batch_size=args.batch_size)
    settings.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "frozen.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else calibrate(args.localdata,settings,frozen_path)
    if frozen["settings"] != asdict(settings) or frozen["manifest_sha256"] != sha256(ROOT / "model_manifest.json"):
        raise ValueError("Frozen benchmark settings/assets differ; use a new output directory")
    if args.phase != "report":
        import torch
        for model in args.models:
            settings.model = model
            adapter = Adapter(model, args.device)
            for name in ("brain1", "brain2"):
                source = args.localdata / f"{name}.ome.zarr"
                destination = args.output / model / output_name(source)
                if destination.exists():
                    root = open_images(destination)[0]
                    if root.attrs["cidenoise"]["settings"] != asdict(settings):
                        raise ValueError(f"Existing benchmark settings mismatch: {destination}")
                    logging.info("Reusing completed output %s", destination)
                    continue
                run_store(source,args.output / model,settings,adapter)
            del adapter
            torch.cuda.empty_cache()
    if args.phase != "infer":
        print(report(args.localdata,args.output,frozen,args.models))


if __name__ == "__main__":
    main()
