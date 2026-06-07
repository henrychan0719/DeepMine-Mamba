# DeepMine-Mamba

**DeepMine-Mamba: Mitigating Information Dilution in Mamba-Based State Space Models for Document Image Binarization**

This repository is the official implementation of **DeepMine-Mamba**, a Mamba-based document image binarization framework designed to preserve weak foreground strokes, faint ink traces, fragmented characters, and boundary-sensitive text details in degraded document images.

> **Code will be released soon.**

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
| 2019 | 75.90 | 76.40 | 15.31 | 7.8167 |
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

## Citation

Coming soon.

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
