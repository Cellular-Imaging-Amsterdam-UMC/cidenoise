"""Train a confocal-only Noise2Noise CNN; no clean targets or synthetic noise required."""
import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from cidenoise.vendor.instant import Noise2Noise
from .data import NoisyPairs, EpochSamples, build_index, sha256, verify_index

LOG=logging.getLogger("confocal-training")
ARCHITECTURE="instant-noise2noise-v1"


def atomic_save(state,path):
    temporary=path.with_suffix(path.suffix+".writing")
    torch.save(state,temporary)
    temporary.replace(path)


def evaluate(model,loader,device,amp):
    model.eval();total=baseline=count=0
    with torch.inference_mode():
        for source,target in loader:
            source,target=source.to(device),target.to(device)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
                output=model(source)
            mse=(output.float()-target).square().mean()
            if not torch.isfinite(mse):
                raise RuntimeError("Nonfinite validation loss")
            total+=float(mse)*len(source)
            baseline+=float((source-target).square().mean())*len(source)
            count+=len(source)
    return dict(noisy_target_mse=total/count,raw_pair_mse=baseline/count,samples=count)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=Path("F:/noise2noise_data"))
    p.add_argument("--output",type=Path)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--steps-per-epoch",type=int,default=1000)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--patch-size",type=int,default=256)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--validation-batches",type=int,default=32)
    p.add_argument("--averages",type=int,nargs="+",default=[1,2,4])
    p.add_argument("--learning-rate",type=float,default=.001)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--precision",choices=["float16","float32"],default="float16")
    p.add_argument("--checkpoint-every",type=int,default=250)
    p.add_argument("--stop-after-steps",type=int,help="Save and stop after this global step (useful for a short trial)")
    p.add_argument("--resume",type=Path,help="Resume last.pt including optimizer, scaler and sample position")
    p.add_argument("--prepare-only",action="store_true",help="Verify/index downloaded data without training")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    root=args.data_root.resolve()
    if min(args.epochs,args.steps_per_epoch,args.batch_size,args.validation_batches,args.checkpoint_every)<1 or not 0<=args.workers<=4:
        raise ValueError("Positive training counts and 0..4 loader workers required")
    if args.patch_size<32 or args.patch_size>512 or args.patch_size%32 or not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        raise ValueError("Patch size must be 32..512 divisible by 32; learning rate must be positive")
    if args.stop_after_steps is not None and args.stop_after_steps<1:
        raise ValueError("Stop step must be positive")
    torch.set_num_threads(4)
    index_path=root/"index.json"
    index=json.loads(index_path.read_text()) if index_path.exists() else build_index(root)
    verify_index(root,index)
    fingerprint=sha256(index_path)
    counts={s:sum(r["split"]==s for r in index["records"]) for s in ("train","validation","test")}
    print(f"Verified data index {fingerprint}; category/FOV groups: {counts}",flush=True)
    if args.prepare_only:
        return 0
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu --precision float32 explicitly")
    if args.device=="cpu" and args.precision=="float16":
        raise ValueError("Use --precision float32 for CPU training")
    output=(args.output or (args.resume.parent if args.resume else root/"runs/noise2noise_confocal")).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run exists: {output}; resume its last.pt or choose a new --output")
    if args.resume and output.exists() and any(output.iterdir()) and output!=args.resume.resolve().parent:
        raise FileExistsError("A branched resume requires an empty output directory")
    output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[
        logging.StreamHandler(),logging.FileHandler(output/"training.log",encoding="utf-8")])
    config={k:v for k,v in vars(args).items() if k not in ("resume","prepare_only","output","data_root","stop_after_steps")}
    state=torch.load(args.resume,map_location="cpu",weights_only=True) if args.resume else None
    if state:
        if state["architecture"]!=ARCHITECTURE or state["dataset_sha256"]!=fingerprint:
            raise ValueError("Checkpoint architecture/data differs")
        for key in config:
            if key not in ("workers","checkpoint_every","device") and state["config"][key]!=config[key]:
                raise ValueError(f"Resume setting differs: {key}; reuse the original run settings")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    if args.device=="cuda":
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(9*1024**3,total*.8)/total)
        torch.cuda.reset_peak_memory_stats()
    model=Noise2Noise().to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs*args.steps_per_epoch,eta_min=args.learning_rate*.01)
    amp=args.device=="cuda" and args.precision=="float16"
    scaler=torch.amp.GradScaler("cuda",enabled=amp,init_scale=1024.)
    epoch_start=next_step=global_step=0;best=float("inf");loss_sum=0.;sample_count=0
    if state:
        model.load_state_dict(state["model_state_dict"],strict=True)
        optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        epoch_start=state["epoch"];next_step=state["next_step"];global_step=state["global_step"];best=state["best_validation"]
        loss_sum=state["loss_sum"];sample_count=state["sample_count"]
        LOG.info("Resuming epoch %d, step %d, global step %d",epoch_start+1,next_step,global_step)
        del state
    metadata=dict(architecture=ARCHITECTURE,dataset_sha256=fingerprint,config=config,
        initialization="random (no widefield/two-photon pretrained initialization)",
        normalization="uint8 / 255 - 0.5; inverse (prediction + 0.5)*255; validate any adaptation to uint16 inputs",
        sources=json.loads((root/"sources.json").read_text()),
        versions=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version),
        dataset_groups=counts,device=torch.cuda.get_device_name() if args.device=="cuda" else "cpu")
    (output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    training=NoisyPairs(root,index,"train",args.patch_size,args.seed,args.averages)
    validation=NoisyPairs(root,index,"validation",args.patch_size,args.seed+1,args.averages)
    sampler=EpochSamples(args.steps_per_epoch*args.batch_size)
    common=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=args.device=="cuda")
    if args.workers:
        common.update(prefetch_factor=2,persistent_workers=True)
    loader=DataLoader(training,sampler=sampler,**common)
    val_loader=DataLoader(validation,sampler=range(args.validation_batches*args.batch_size),**common)
    def save(epoch,step,filename="last.pt"):
        record=dict(metadata,format_version=1,model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,next_step=step,
            global_step=global_step,best_validation=best,loss_sum=loss_sum,sample_count=sample_count)
        atomic_save(record,output/filename)
    if not args.resume:
        save(0,0)
    LOG.info("%s; %d epochs x %d steps; no test or brain images used",
             "Resumed optimization" if args.resume else "Training starts from random weights",args.epochs,args.steps_per_epoch)
    started=time.perf_counter()
    try:
        for epoch in range(epoch_start,args.epochs):
            model.train();sampler.epoch=epoch;sampler.skip=next_step*args.batch_size
            for step,(source,target) in enumerate(loader,start=next_step):
                source,target=source.to(args.device,non_blocking=True),target.to(args.device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=amp):
                    prediction=model(source)
                loss=(prediction.float()-target).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; last saved checkpoint retained")
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();scheduler.step()
                global_step+=1;loss_sum+=float(loss.detach())*len(source);sample_count+=len(source)
                if global_step%25==0 or step==0:
                    LOG.info("epoch %d/%d step %d/%d loss %.6f lr %.6g",epoch+1,args.epochs,step+1,args.steps_per_epoch,float(loss.detach()),scheduler.get_last_lr()[0])
                if global_step%args.checkpoint_every==0:
                    save(epoch,step+1)
                if args.stop_after_steps is not None and global_step>=args.stop_after_steps:
                    save(epoch,step+1)
                    LOG.info("Trial stopped at step %d; resume %s",global_step,output/"last.pt")
                    return 0
            metrics=evaluate(model,val_loader,args.device,amp)
            row=dict(epoch=epoch+1,global_step=global_step,train_mse=loss_sum/sample_count,validation=metrics,
                elapsed_seconds=time.perf_counter()-started,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if args.device=="cuda" else 0,
                process_rss_gib=psutil.Process().memory_info().rss/1024**3)
            with (output/"metrics.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
            LOG.info("Epoch %d validation noisy-target MSE %.6f (raw pair %.6f)",epoch+1,metrics["noisy_target_mse"],metrics["raw_pair_mse"])
            improved=metrics["noisy_target_mse"]<best
            if improved:
                best=metrics["noisy_target_mse"]
            loss_sum=0.;sample_count=0;next_step=0
            save(epoch+1,0)
            if improved:
                save(epoch+1,0,"best.pt")
        LOG.info("Completed. Best validation MSE %.6f; checkpoint %s",best,output/"best.pt")
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Resume %s; work after its last complete step will be repeated.",output/"last.pt")
        return 130
    except torch.cuda.OutOfMemoryError:
        LOG.error("GPU memory exhausted. Use a new run with a smaller batch/patch, or resume the saved configuration on a larger GPU.")
        raise
    return 0


if __name__=="__main__":
    raise SystemExit(main())
