# DeepMine-Mamba

[![arXiv](https://img.shields.io/badge/arXiv-2606.08781-b31b1b.svg)](https://arxiv.org/abs/2606.08781)

**[DeepMine-Mamba: Mitigating Information Dilution in Mamba-Based State Space Models for Document Image Binarization](https://arxiv.org/abs/2606.08781)**

This repository is the official implementation of **DeepMine-Mamba**, a Mamba-based document image binarization framework designed to preserve weak foreground strokes, faint ink traces, fragmented characters, and boundary-sensitive text details in degraded document images.

---

## Overview

Document image binarization aims to separate foreground text from degraded backgrounds while preserving thin, broken, and low-contrast strokes. Although CNN-, Transformer-, GAN-, and diffusion-based methods have been widely explored, Mamba-based state space models remain under-explored for document image binarization.

DeepMine-Mamba investigates the information dilution problem in Mamba-based feature propagation. Direct long-range state-space propagation may weaken faint foreground cues, especially thin strokes, fragmented characters, and low-contrast ink traces.

To address this issue, we propose an **Anti-Dilution Gate**, which estimates propagation-induced feature changes and selectively restores stroke-sensitive local responses after Mamba-based propagation.

---

## Main Contributions

- We explore **Mamba-based state space modeling** for document image binarization.
- We identify a stroke-level **information dilution problem** caused by long-range Mamba feature propagation.
- We propose an **Anti-Dilution Gate** to restore weak foreground responses while preserving global contextual modeling.
- We evaluate DeepMine-Mamba on DIBCO/H-DIBCO benchmarks using a strict **leave-one-year-out protocol**.
- Experimental results show that DeepMine-Mamba achieves competitive overall performance, especially in foreground stroke preservation.

---

## Method

DeepMine-Mamba consists of the following components:

- ConvNeXt-based encoder for multi-scale feature extraction
- Sobel edge prior for detail-aware guidance
- Four-direction Mamba propagation for long-range structural modeling
- Anti-Dilution Gate for stroke-sensitive feature restoration
- Refinement heads for foreground, background, and adaptive threshold prediction

The overall goal is to combine the global modeling ability of Mamba with local stroke-preserving refinement.

---

## Installation

Requirements: Python 3.10+, a CUDA-capable GPU (required by the `mamba-ssm` selective-scan kernels), and a PyTorch build that matches your local CUDA toolkit.

```bash
pip install -r requirements.txt
```

> **Note:** `mamba-ssm` compiles custom CUDA kernels at install time, so the CUDA toolkit (`nvcc`) must be available and compatible with your installed PyTorch.

---

## Usage

Input images are normalized with ImageNet statistics. The model returns a dict of tensors; `soft_bin` is the binarization map with text (foreground) ≈ 1 and background ≈ 0.

The following inference example requires a compatible trained model checkpoint (`deepmine_mamba_best.pth`) and an input image (`page.png`).

```python
import torch
from PIL import Image
from torchvision import transforms

from deepmine_mamba import ConvNeXtDeepMineMamba, sliding_window_inference

device = "cuda"  # mamba-ssm requires a CUDA GPU

# Initialize the model and load trained weights
model = ConvNeXtDeepMineMamba(pretrained_backbone=False).to(device).eval()
state = torch.load("deepmine_mamba_best.pth", map_location=device)
model.load_state_dict(state)  # adjust the key (e.g. state["model"]) if your checkpoint wraps the state_dict

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = transform(Image.open("page.png").convert("RGB")).unsqueeze(0).to(device)

# Full-page inference with overlapping windows
prob = sliding_window_inference(model, img, crop_size=512, stride=256)  # (1, 1, H, W)
binary = (prob > 0.5).float()

# Save as a standard binarized image (black text on white background)
out = (255 * (1 - binary))[0, 0].byte().cpu().numpy()
Image.fromarray(out).save("page_bin.png")
```

---

## Results

### Leave-One-Year-Out Results on DIBCO/H-DIBCO

| Test Year | FM ↑ | Fps ↑ | PSNR ↑ | DRD ↓ |
|---|---:|---:|---:|---:|
| 2009 | 95.74 | 96.50 | 21.59 | 1.3157 |
| 2010 | 94.47 | 96.84 | 21.83 | 1.3455 |
| 2011 | 95.35 | 97.56 | 21.40 | 1.3588 |
| 2012 | 96.26 | 97.32 | 23.30 | 1.1860 |
| 2013 | 96.15 | 97.93 | 23.13 | 1.2347 |
| 2014 | 97.45 | 98.84 | 23.58 | 0.6938 |
| 2016 | 90.28 | 94.90 | 19.26 | 3.2500 |
| 2017 | 93.31 | 95.19 | 19.39 | 2.3480 |
| 2018 | 90.69 | 94.62 | 20.00 | 2.8745 |
| 2019 | 75.90 | 76.44 | 15.41 | 7.8167 |
| **Average** | **92.56** | **94.61** | **20.88** | **2.3424** |

---

### Ablation Study

| Model | FM ↑ | Fps ↑ | PSNR ↑ | DRD ↓ |
|---|---:|---:|---:|---:|
| Baseline U-Net | 74.84 | 75.55 | 16.74 | 8.13 |
| + ConvNeXt Encoder | 86.17 | 87.96 | 17.39 | 7.24 |
| + Sobel Edge Prior | 87.05 | 88.53 | 17.78 | 6.51 |
| + DeepMine-Mamba Block | 90.54 | 93.35 | 19.72 | 4.79 |
| + Anti-Dilution Gate | 92.43 | 94.27 | 20.16 | 2.59 |
| + Refinement Heads | **92.56** | **94.61** | **20.88** | **2.34** |

---

## Data

Experiments are conducted on the official DIBCO / H-DIBCO benchmark datasets. The datasets are not redistributed in this repository; please obtain them from the official competition pages.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{chan2026deepmine,
  title={DeepMine-Mamba: Mitigating Information Dilution in Mamba-Based State Space Models for Document Image Binarization},
  author={Chan, Sheng-Wei and Wang, Yung-Che and Pan, Hsin-Jui and Lin, Chia-Min and Chiang, Jen-Shiun},
  journal={arXiv preprint arXiv:2606.08781},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
