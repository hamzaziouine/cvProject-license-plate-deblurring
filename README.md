# License Plate Deblurring

**Group Opus** -- Hamza Ziouine & Leonce Theureau
**Course:** Introduction to Computer Vision, S8, Universite Internationale de Rabat
**Date:** April 2026

---

## Overview

Restore sharp, readable license plates from motion-blurred images using classical computer vision techniques: FFT-based blind blur estimation, Wiener filtering, Richardson--Lucy deconvolution, constrained least squares, and Total Variation post-processing.

---

## Repository Structure

```
final_project/
├── code/
│   └── pipeline.ipynb              # Main notebook (all tasks)
├── src/
│   ├── blur_generator.py           # Gaussian/motion/defocus blur + degradation
│   ├── classical_deblur.py         # 5 deconvolution methods + TV post-processing
│   ├── evaluation.py               # PSNR, SSIM, FFT, edge detection metrics
│   └── utils.py                    # Image I/O helpers
├── scripts/
│   └── generate_dataset.py         # Generate synthetic blurry/sharp pairs
├── config/
│   └── default.yaml                # Blur, degradation, split parameters
├── data/
│   ├── synthetic/                  # Generated pairs (not in git)
│   ├── splits/                     # train.txt, val.txt, test.txt
│   └── samples/                    # Example pairs (in git)
├── outputs/
│   └── figures/                    # Plots referenced by report and slides
├── report/
│   ├── report.tex                  # Project report (LaTeX)
│   └── slides.tex                  # Presentation slides (Beamer)
├── report.pdf                      # Compiled report
└── requirements.txt
```

---

## Tasks Implemented

### Task 1: Synthetic Dataset Generation
3,003 paired blurry/sharp images from 1,001 clean plates with three blur types (Gaussian, motion, defocus), additive noise, and JPEG compression.

### Task 2: Blur Parameter Estimation via FFT
Blind estimation of blur angle and length from the Fourier magnitude spectrum. Dark bands perpendicular to blur direction reveal the angle; their spacing gives kernel length.

### Task 3: Classical Deconvolution (5 methods)
1. **Inverse filter** -- naive baseline, demonstrates noise amplification
2. **Wiener filter** -- optimal for known PSF and noise statistics
3. **Unsupervised Wiener** -- auto-estimates noise-to-signal ratio
4. **Richardson--Lucy** -- iterative Bayesian deconvolution (Poisson model)
5. **Constrained Least Squares** -- Laplacian-regularised frequency domain filter

### Task 4: Evaluation
PSNR, SSIM, and OCR readability metrics across the full 151-image test set.

### Post-Processing: Total Variation Denoising
Chambolle TV denoising applied after deconvolution to reduce ringing artifacts.

---

## Setup

```bash
pip install -r requirements.txt
```

### Dataset Generation

```bash
python scripts/generate_dataset.py --config config/default.yaml
```

### Running the Notebook

Open `code/pipeline.ipynb` in Jupyter and run all cells sequentially.

---

## Course Concept Connections

- **Lab 1 (Convolution):** Blur as convolution with PSF; deblurring as deconvolution
- **Lab 2 (Fourier/Frequency):** FFT-based blur estimation, Wiener and CLS in frequency domain
- **Lab 3 (Edge Detection):** Canny/Sobel comparison; CLS uses the Laplacian as regulariser

---

Academic project -- Universite Internationale de Rabat, 2026.
