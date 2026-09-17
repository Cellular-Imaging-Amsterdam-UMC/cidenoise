"""Report convergence and evaluate the selected checkpoint on held-out FMD fields."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.data import NoisyPairs, verify_index
from training.train_noise2noise import evaluate
from cidenoise.adapters import Adapter, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('F:/noise2noise_data'))
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/confocal-training-review')
    args=parser.parse_args()
    run=args.data_root/'runs/noise2noise_confocal'
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    best=min(rows,key=lambda r:r['validation']['noisy_target_mse'])
    def mean(start,end):
        return float(np.mean([r['validation']['noisy_target_mse'] for r in rows if start<=r['epoch']<=end]))
    index=json.loads((args.data_root/'index.json').read_text());verify_index(args.data_root,index)
    adapter=Adapter('noise2noise-confocal','cuda');adapter.load()
    from cidenoise.vendor.instant import Noise2Noise
    original=Noise2Noise().cuda().eval()
    original.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['model_state_dict'])
    patch=np.random.default_rng(42).random((1,1,64,64),dtype=np.float32)-.5
    with torch.inference_mode():
        reference=original(torch.from_numpy(patch).cuda()).cpu().numpy()
    predicted=adapter.predict(patch)
    np.testing.assert_array_equal(predicted,reference)
    del original
    pairs=NoisyPairs(args.data_root,index,'test',256,44,(1,2,4))
    loader=DataLoader(pairs,batch_size=8,sampler=range(256),num_workers=0)
    metrics=evaluate(adapter.model,loader,'cuda',False)
    report=dict(best_epoch=best['epoch'],best_validation=best['validation'],
        mean_epochs_81_90=mean(81,90),mean_epochs_91_100=mean(91,100),
        final_window_improvement_percent=100*(mean(81,90)-mean(91,100))/mean(81,90),
        checkpoint_sha256=sha256(run/'best.pt'),held_out_test=metrics,
        adapter_maximum_absolute_error=float(np.max(np.abs(predicted-reference))),
        test_protocol='256 fixed patches, seed 44, FOV19 only; FP32; averages 1/2/4; noisy targets, not clean PSNR',
        interpretation='Validation has plateaued under the completed schedule; use epoch 75. This does not establish optimal training or generalization to brain images.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax in axes:
        ax.plot([r['epoch'] for r in rows],[r['validation']['noisy_target_mse'] for r in rows],label='Validation noisy-target MSE')
        ax.axvline(best['epoch'],color='green',linestyle='--',label='Selected epoch 75')
        ax.set_xlabel('Epoch');ax.set_ylabel('MSE');ax.grid(alpha=.3)
    axes[0].legend();axes[1].set_xlim(20,100);axes[1].set_ylim(.000925,.00094)
    axes[1].set_title('Late-training detail')
    fig.tight_layout();fig.savefig(args.output/'validation_curve.png',dpi=160)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
