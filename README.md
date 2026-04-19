# License Plate Deblurring

**Group Opus** -- Hamza Ziouine & Leonce Theureau
**Course:** Introduction to Computer Vision, S8, Universite Internationale de Rabat
**Date:** April 2026

---

## Overview

A classical, learning-free evaluation of blind deblurring for license plate images. Pipeline: FFT-based blind PSF estimation &rarr; five deconvolution methods (inverse, Wiener, unsupervised Wiener, Richardson--Lucy, CLS) &rarr; optional TV post-processing &rarr; image-fidelity metrics (PSNR, SSIM).

**Key finding.** Under blind PSF estimation with realistic degradation (additive noise + JPEG compression), all classical methods reduce image fidelity below the blurry baseline on the pipeline's target domain (motion blur). Wiener is the least-damaging method but still loses ~8 dB PSNR vs. doing nothing. With an oracle (known) PSF, the same Wiener code gains ~2 dB, confirming the deconvolution is correctly implemented and PSF estimation is the true bottleneck.

See [`report/report.pdf`](report/report.pdf) for the full evaluation and discussion.

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
│   ├── report.tex                  # Project report (LaTeX source)
│   ├── report.pdf                  # Compiled report
│   ├── slides.tex                  # Presentation slides (Beamer source)
│   └── slides.pdf                  # Compiled slides
└── requirements.txt
```

---

## Tasks Implemented

### Task 1: Synthetic Dataset Generation
3,003 paired blurry/sharp images from 1,001 clean plates. Three blur types (Gaussian, motion, defocus) applied with randomised parameters, followed by additive noise and JPEG compression. Split by source image (70/15/15) to prevent data leakage.

### Task 2: Blind Blur Parameter Estimation via FFT
Estimate blur angle and length from the Fourier magnitude spectrum. The estimator assumes motion blur; report &sect;4.2 quantifies the cost of applying this estimator to non-motion blur as a known methodological limitation.

### Task 3: Five Classical Deconvolution Methods
1. **Inverse filter** -- pedagogical baseline; demonstrates noise amplification
2. **Wiener filter** -- MMSE-optimal linear estimator, core Lab 2 method
3. **Unsupervised Wiener** -- auto noise-to-signal estimation via Gibbs sampling
4. **Richardson--Lucy** -- iterative Bayesian, Poisson noise model
5. **Constrained Least Squares** -- pedagogical; Laplacian-regularised frequency-domain filter

### Task 4: Evaluation
PSNR and SSIM on the 453-image test set (151 per blur type). Headline results reported on the **motion subset** (the estimator's target). Per-blur-type breakdown and single-image oracle validation in the report.

### Optional: Total Variation Denoising
Chambolle TV denoising is implemented as an available post-processing step. A quantitative TV on/off ablation is flagged as future work in the report and is **not included in the headline tables**.

### Optional: OCR Demonstration (standalone script)
`scripts/ocr_demo.py` runs a qualitative EasyOCR comparison (blurry vs.\ Wiener-deblurred) on the three sample plates in `data/samples/` -- no full dataset download needed. Requires `pip install easyocr`. A quantitative CER evaluation across the test set is future work.

---

## Dataset Access

The 1,001 clean source plates and the 3,003 generated blurry/sharp pairs are **not tracked in this repository** (too large for Git). They are hosted on Google Drive:

**Clean plates + generated dataset:** 
`https://drive.google.com/file/d/1dpV-lcp0q4gqLsrifik2zK-2F2wftVtc/view?usp=drive_link`

After downloading:
```
final_project/data/
├── clean_plates/      # Extract the 1001 clean plate images here
└── synthetic/         # (optional) pre-generated pairs, OR run generate_dataset.py
```

Regeneration is deterministic given `seed: 42` in `config/default.yaml`, so you can either download the pre-generated set or reproduce it locally from the clean plates.

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
