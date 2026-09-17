"""PNG-only visual comparison of the original and all registered models."""
from dataclasses import replace
import gc
import json
import logging
import time
from pathlib import Path
import uuid
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont, PngImagePlugin
from .adapters import Adapter, MODEL_IDS
from .ome_zarr import open_images
from .normalization import plane_bounds, stack_bounds
from .tiling import predict_plane, reflected_index

LOG=logging.getLogger(__name__)
PALETTE=('FF0000','00FF00','0000FF','FF00FF','00FFFF','FFFF00','FF8000','FFFFFF')


class Crop:
    def __init__(self,image,cropped=True):
        self.image=image
        self.y0=max(0,(image.length('y')-512)//2)
        self.x0=max(0,(image.length('x')-512)//2)
        self.height=min(512,image.length('y'));self.width=min(512,image.length('x'))
        if not cropped:
            self.y0=self.x0=0
            self.height=image.length('y');self.width=image.length('x')
        self.array=image.array

    def length(self,axis):
        return self.height if axis=='y' else self.width if axis=='x' else self.image.length(axis)

    def plane(self,t,c,z):
        return self.image.plane(t,c,z,slice(self.y0,self.y0+self.height),slice(self.x0,self.x0+self.width))


def display(raw,image):
    """All channels; raw-derived display ranges shared unchanged by every panel."""
    specs=image.group.attrs.get('omero',{}).get('channels',[])
    result=[]
    for c,plane in enumerate(raw):
        spec=specs[c] if c<len(specs) else {}
        color=str(spec.get('color',PALETTE[c%len(PALETTE)])).lstrip('#')
        if len(color)!=6 or any(ch not in '0123456789abcdefABCDEF' for ch in color):
            color=PALETTE[c%len(PALETTE)]
        low,high=map(float,np.percentile(plane,[1,99.8]))
        if high<=low:
            low,high=0.,max(float(plane.max()),1.)
        result.append(dict(label=spec.get('label',f'Channel {c+1}'),color=color,low=low,high=high))
    return result


def overlay(planes,specs):
    rgb=np.zeros((*planes[0].shape,3),np.float32)
    for plane,spec in zip(planes,specs):
        if not np.isfinite(plane).all():
            raise ValueError('Nonfinite pixels in benchmark')
        intensity=np.clip((plane-spec['low'])/(spec['high']-spec['low']),0,1)
        color=np.array([int(spec['color'][i:i+2],16)/255 for i in (0,2,4)])
        rgb+=intensity[...,None]*color
    return PILImage.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))


def restored(crop,t,z,c,settings,adapter,bounds=None):
    center=crop.plane(t,c,z)
    if not np.isfinite(center).all():
        raise ValueError('Nonfinite input pixels')
    if center.min()==center.max() and adapter.context==1:
        return center
    low,high,_=bounds or (stack_bounds(crop,t,c) if settings.model.startswith('unifmir') else plane_bounds(center))
    if settings.model=='noise2noise-fmd':
        maximum=max(float(center.max()),0.)
        low,high=maximum/2,maximum*1.5
    elif settings.model.startswith('cellpose-'):
        low,high=map(float,np.percentile(center,[1,99]))
    scale=high-low if high>low else 1.
    radius=adapter.context//2
    context=np.stack([crop.plane(t,c,reflected_index(z+d,crop.length('z'))) for d in range(-radius,radius+1)])
    if not np.isfinite(context).all():
        raise ValueError('Nonfinite neighboring Z pixels')
    if settings.model=='fluoresfm':
        context=np.maximum(context,0)
    normalized=(context-low)/(scale+(1e-20 if settings.model.startswith('unifmir') else 0))
    return predict_plane(lambda y0,y1,x0,x1: normalized[:,y0:y1,x0:x1],center.shape,
                         adapter.predict,settings.tile_size,settings.overlap,settings.batch_size)*scale+low


def maximum_projection(planes):
    result=None
    for plane in planes:
        if not np.isfinite(plane).all():
            raise ValueError('Nonfinite pixels in benchmark')
        if result is None:
            result=plane.copy()
        else:
            np.maximum(result,plane,out=result)
    return result


def run_benchmark(sources,outfolder,settings,adapter_factory=Adapter,mode='2d-crop'):
    import torch
    if mode not in ('2d-full','2d-crop','3d-full','3d-crop'):
        raise ValueError('Expected an enabled benchmark mode')
    outfolder=Path(outfolder).resolve();outfolder.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for source in sources:
        source=Path(source).resolve()
        if outfolder==source or source in outfolder.parents:
            raise ValueError('Output must not be inside input')
        _,images=open_images(source)
        for field,image in enumerate(images):
            for t in range(image.length('t')):
                destination=outfolder/f'{source.name.removesuffix(".ome.zarr")}__field{field:04d}__t{t:04d}__{mode}__benchmark.png'
                if destination.exists():
                    raise FileExistsError(f'Benchmark already exists: {destination}')
                jobs.append((source,image,t,destination))
    failures=[]
    for source,image,t,destination in jobs:
        crop=Crop(image,cropped=mode.endswith('crop'));z=image.length('z')//2
        indices=range(image.length('z')) if mode.startswith('3d') else [z]
        raw=[maximum_projection(crop.plane(t,c,zi) for zi in indices) for c in range(image.length('c'))]
        specs=display(raw,image)
        panels=[('Original',overlay(raw,specs))];records=[]
        for identifier in MODEL_IDS:
            adapter=None
            try:
                chosen=replace(settings,model=identifier,channels='all').resolved()
                descriptions=chosen.validate()
                adapter=adapter_factory(identifier,chosen.device,precision=chosen.precision)
                adapter.load()
                for c in range(image.length('c')):
                    adapter.set_structure(descriptions.get(str(c+1),''))
                # Include crop reads, normalization, tiling and prediction;
                # exclude model/prompt loading and gallery rendering. Synchronize CUDA.
                if chosen.device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                started=time.perf_counter()
                predicted=[]
                for c in range(image.length('c')):
                    adapter.set_structure(descriptions.get(str(c+1),''))
                    bounds=stack_bounds(crop,t,c) if identifier.startswith('unifmir') else None
                    predicted.append(maximum_projection(restored(crop,t,zi,c,chosen,adapter,bounds) for zi in indices))
                if chosen.device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                panels.append((identifier,overlay(predicted,specs)))
                records.append(dict(model=identifier,processing_seconds=elapsed,settings=vars(chosen),provenance=adapter.provenance))
                LOG.info('Benchmark %s field=%s T=%d mode=%s Z planes=%d: %s complete',source.name,image.path,t,mode,len(indices),identifier)
            except Exception as exc:
                LOG.exception('Benchmark model %s failed',identifier)
                failures.append(f'{destination.name}: {identifier}: {exc}')
                panel=PILImage.new('RGB',(crop.width,crop.height),'#281818')
                ImageDraw.Draw(panel).text((10,10),'FAILED - see execution log',fill='white')
                panels.append((identifier,panel));records.append(dict(model=identifier,error=str(exc)))
            finally:
                del adapter
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        fastest=min((r['processing_seconds'] for r in records if 'processing_seconds' in r),default=None)
        for record in records:
            if 'processing_seconds' in record:
                record['relative_to_fastest']=record['processing_seconds']/fastest
        columns=4;cell_w=max(crop.width,256);cell_h=crop.height+56
        canvas=PILImage.new('RGB',(columns*cell_w,120+2*cell_h+24*len(specs)),'#151a22')
        draw=ImageDraw.Draw(canvas)
        draw.font=ImageFont.load_default(size=16)
        view=f'Z maximum projection of all {image.length("z")} planes' if mode.startswith('3d') else f'Z={z} (zero-based)'
        draw.text((12,10),f'{source.name} | field {image.path or "/"} | T={t} | {view}',fill='white')
        draw.text((12,32),f'{mode}: X={crop.x0}, Y={crop.y0}, {crop.width} x {crop.height}; all channels',fill='white')
        draw.text((12,54),'Same raw-derived display ranges in every panel; additive colour overlay; no intensity matching.',fill='white')
        draw.text((12,76),'Times: all requested Z/channels, reads + normalization + inference + projection; excludes model/prompt loading. Fastest = 1.0x.',fill='white')
        for i,(label,panel) in enumerate(panels):
            x=(i%columns)*cell_w;y=120+(i//columns)*cell_h
            draw.text((x+8,y+8),label,fill='white')
            if i and 'processing_seconds' in records[i-1]:
                record=records[i-1]
                draw.text((x+8,y+28),f'{record["processing_seconds"]:.3f} s | {record["relative_to_fastest"]:.2f}x fastest',fill='white')
            canvas.paste(panel,(x,y+56))
        for c,spec in enumerate(specs):
            draw.text((12,120+2*cell_h+24*c),f'{c+1}: {spec["label"]} | display {spec["low"]:.3g} .. {spec["high"]:.3g}',fill='#'+spec['color'])
        metadata=PngImagePlugin.PngInfo()
        metadata.add_text('cidenoise',json.dumps(dict(source=source.name,field=image.path,t=t,z=z if mode.startswith('2d') else None,
            benchmark_mode=mode,z_indices=list(indices),projection='maximum' if mode.startswith('3d') else 'none',
            crop=dict(x=crop.x0,y=crop.y0,width=crop.width,height=crop.height),display=specs,models=records)))
        temporary=destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            canvas.save(temporary,format='PNG',pnginfo=metadata)
            if destination.exists():
                raise FileExistsError(destination)
            temporary.rename(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if failures:
        raise RuntimeError('Some benchmark models failed; galleries mark failed panels. '+ '; '.join(failures))
