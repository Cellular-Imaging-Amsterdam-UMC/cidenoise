# Denoising candidates

The first release implements pretrained FluoResFM and UniFMIR only.

| Method | Evidence/source | Future role |
|---|---|---|
| FluoResFM | [Nature Communications 2026](https://doi.org/10.1038/s41467-026-70307-4), [code](https://github.com/qiqi-lu/fluoresfm) | Primary pretrained, text-conditioned denoising |
| UniFMIR | [Nature Methods 2024](https://doi.org/10.1038/s41592-024-02244-3), [code](https://github.com/cxm12/UNiFMIR) | Pretrained single-plane and five-plane comparators |
| N2V2 / CAREamics | [Project](https://github.com/CAREamics/careamics), [N2V2 paper](https://arxiv.org/abs/2211.08512) | Dataset-adapted self-supervised learning; needs training or a suitable checkpoint |
| FM2S | [Project](https://github.com/Danielement321/FM2S) | Recent single-image fluorescence denoising; learns from each input |
| Noise2Fast | [Project](https://github.com/jason-lequyer/Noise2Fast) | Lightweight per-image self-supervised comparator |
| Cellpose3 denoising | [Nature Methods 2025](https://doi.org/10.1038/s41592-025-02595-5) | Segmentation-oriented pretrained restoration; not selected for v1 |
| FAST | [Project](https://github.com/FDU-donglab/FAST) | Time-series fluorescence denoising |
| DeepCAD-RT | [Project](https://cabooster.github.io/DeepCAD-RT/) | Temporal denoising for live imaging |

Published performance on other data does not establish performance on these brain
samples. Maintain raw images and assess biological structures alongside reference metrics.
