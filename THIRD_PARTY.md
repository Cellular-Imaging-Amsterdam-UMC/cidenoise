# Third-party code and pretrained assets

CIDenoise's combined distribution is GPL-3.0. See LICENSE. Local data and downloaded
weights are not committed to the source repository.

| Component | Source and revision | Terms |
|---|---|---|
| Generic launcher and Bilayers utilities | Cellular-Imaging-Amsterdam-UMC/cisegmentation, adapted from the local checkout | MIT; original notice retained below |
| FluoResFM network and embedder | qiqi-lu/napari-fluoresfm, `2fded7c31be8c52476de89934ced9093ebe2c307` | MIT, see `cidenoise/vendor/fluoresfm/LICENSE` |
| UniFMIR SwinIR denoising network | cxm12/UNiFMIR, `dc90f055db7434078ef68b8c1a1fba97af6dbbc0` | GPL-3.0, see `cidenoise/vendor/unifmir/LICENSE` |
| FluoResFM release archive | https://doi.org/10.5281/zenodo.18382702 | Zenodo archive: CC-BY-4.0; this is separate from the code's MIT license |
| BiomedCLIP | microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, supplied with the FluoResFM archive | MIT model terms; retain Microsoft attribution |
| UniFMIR checkpoints | https://github.com/cxm12/UNiFMIR/releases/tag/2023.10.05 | Distributed with the GPL-3.0 project; retain upstream attribution |
| Noise2Noise FMD architecture and published weights | ND-HowardGroup/Instant-Image-Denoising, `c885aee8adb62bcd2d5c3862273d4ea0f68b3068`; Varun Mannam, University of Notre Dame | GPL-3.0; see `cidenoise/vendor/instant/LICENSE`. Inference architecture ported from TensorFlow to PyTorch; published NumPy kernels transposed without training. |
| Cellpose denoising CPnet and cyto3/nuclei weights | MouseLand/cellpose, `fb22843e70d03f7884c301b7b72bedd7d9c3d2d9` (3.1.1.1); Howard Hughes Medical Institute | BSD-3-Clause; see `cidenoise/vendor/cellpose/LICENSE`. Network source unchanged; only denoising checkpoints used. |

The model manifest pins checksums, source revisions, download URLs and normalization.
Its model `license` fields identify the model implementation's code license; archive
and text-encoder terms above also apply when distributing model-cache images.

Original FluoResFM/UniFMIR vendoring is reproducible with `tools/vendor_sources.py`:

- FluoResFM: only the internal import path is changed. CIDenoise installs equivalent
  PyTorch scaled-dot-product attention at runtime; original attention remains available
  for the upstream parity test. No compilation or external FlashAttention is required.
- UniFMIR: retain only denoising network classes and dependencies; remove diagnostic
  prints and unconditional CUDA synchronization. Architecture and state-dict keys are
  unchanged. Native CPU execution therefore works too.
- CIDenoise uses reflected Z boundaries, as specified for this workflow. UniFMIR's
  original evaluation script repeats boundary slices. Upstream parity is measured on
  identical input patches; it does not claim identical whole-volume boundary results.
- Spatial tiling, normalization reversal, metadata writing and evaluation are implemented
  by CIDenoise. FluoResFM's original demo writes normalized output; CIDenoise reverses
  the recorded affine input normalization into source intensity units.

The local reference checkout is not a runtime dependency. Optional BIOMERO import
validation uses a separately installed site metadata probe and the active deployment.

## Reference workflow MIT notice

The local `noise2noise-confocal` checkpoint was trained from random initialization
using the GPL-3.0 Noise2Noise architecture port and only the five confocal categories
of the FMD dataset (Zhang et al., DOI 10.7274/r0-ed2r-4052, CC BY-SA 4.0).
The data source and license are recorded in `training/confocal_sources.json`.
Local trained weights are not distributed in this repository and have no public
download URL; the model manifest pins their identity and training provenance.


MIT License

Copyright (c) 2026 Cellular Imaging, Amsterdam UMC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
