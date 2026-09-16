# Denoising candidates

Internet review updated 16 September 2026. Six checkpoint choices are now integrated:
FluoResFM, two UniFMIR models, Noise2Noise FMD, and two Cellpose denoising models.
No local training or fine-tuning is performed.

| Method | Evidence/source | Future role |
|---|---|---|
| FluoResFM | [Nature Communications 2026](https://doi.org/10.1038/s41467-026-70307-4), [code](https://github.com/qiqi-lu/fluoresfm) | Primary pretrained, text-conditioned denoising |
| UniFMIR | [Nature Methods 2024](https://doi.org/10.1038/s41592-024-02244-3), [code](https://github.com/cxm12/UNiFMIR) | Pretrained single-plane and five-plane comparators |
| N2V2 / CAREamics | [Project](https://github.com/CAREamics/careamics), [N2V2 paper](https://arxiv.org/abs/2211.08512) | Dataset-adapted self-supervised learning; needs training or a suitable checkpoint |
| FM2S | [Project](https://github.com/Danielement321/FM2S) | Recent single-image fluorescence denoising; learns from each input |
| Noise2Fast | [Project](https://github.com/jason-lequyer/Noise2Fast) | Lightweight per-image self-supervised comparator |
| Cellpose3 denoising | [Nature Methods 2025](https://doi.org/10.1038/s41592-025-02595-5), [official restoration documentation](https://cellpose.readthedocs.io/en/v3.1.1.1/restore.html) | Implemented: `denoise_cyto3` and `denoise_nuclei`. Fast, but segmentation/perceptual objectives require review of fine structures. No segmentation is run. |
| Noise2Noise FMD, Instant Image Denoising | [Official pretrained models](https://github.com/ND-HowardGroup/Instant-Image-Denoising), [Optica 2022 paper](https://authors.library.caltech.edu/records/1vkmt-27e03) | Implemented: published no-BatchNorm microscopy U-Net, ported to PyTorch. Strong speed candidate; includes diverse microscopy training domains. |
| FMD Noise2Noise / DnCNN | [CVPR 2019 official code](https://github.com/yinhaoz/denoising-fluorescence), [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Poisson-Gaussian_Denoising_Dataset_With_Real_Fluorescence_Microscopy_Images_CVPR_2019_paper.pdf) | Published pretrained microscopy checkpoints; additional DnCNN comparator worth testing after the current Noise2Noise integration. |
| FBI-Denoiser | [CVPR 2021 official code/checkpoints](https://github.com/csm9493/FBI-Denoiser) | Pretrained CF_MICE, CF_FISH and TP_MICE pairs available. Download inspected; not integrated. Requires correct Poisson-Gaussian noise estimation, variance stabilization and inverse transformation for 16-bit inputs. |
| SCUNet | [Official code/checkpoints](https://github.com/cszn/SCUNet) | Published blind/fixed-Gaussian pretrained models, primarily natural-image training. Lower priority than microscopy-trained CNNs; no local quality claim. |
| CNNT | [Scientific Reports 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11303381/), [official repository](https://github.com/AzR919/CNNT_Microscopy) | Interesting volumetric backbone, but inspected repository describes training/fine-tuning without a ready pinned checkpoint download. Defer pending usable weights and redistribution terms. |
| Self-supervised transfer denoising | [Bo Huang lab code and pretrained models](https://github.com/BoHuangLab/Transfer-Learning-Denoising) | Existing checkpoints are candidates, but proposed adaptation normally includes training. Evaluate only unchanged checkpoints if added. |
| Noise2Detail (MICCAI 2025) | [Official repository](https://github.com/ctom2/noise2detail) | Not training-free: optimizes/fine-tunes a network on noisy data. Excluded under the current pretrained-only constraint. |
| FAST | [Project](https://github.com/FDU-donglab/FAST) | Time-series fluorescence denoising |
| DeepCAD-RT | [Project](https://cabooster.github.io/DeepCAD-RT/) | Temporal denoising for live imaging |

Published performance on other data does not establish performance on these brain
samples. Maintain raw images and assess biological structures alongside reference metrics.
