"""Regenerate all report/slides figures using classical methods only."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks

from src.classical_deblur import (
    make_motion_psf, inverse_filter, wiener_deblur,
    richardson_lucy_deblur, constrained_least_squares,
)
from src.evaluation import (
    compute_psnr, compute_ssim,
    compute_fft_magnitude, compute_canny_edges, compute_sobel_edges,
)

FIGURES = PROJECT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DATA = PROJECT / "data"
BLURRY = DATA / "synthetic" / "blurry"
SHARP = DATA / "synthetic" / "sharp"


def load(path):
    return np.array(Image.open(path).convert("RGB"))

def to_u8(x):
    return (x * 255).clip(0, 255).astype(np.uint8)

def estimate_blur_angle(gray):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) // 2
    if max_r <= 5:
        return 0.0
    angles = np.arange(0, 180, dtype=float)
    profile = np.zeros(len(angles))
    for i, a in enumerate(angles):
        theta = np.radians(a)
        vals = []
        for r in range(5, max_r):
            x = int(round(cx + r * np.cos(theta)))
            y = int(round(cy - r * np.sin(theta)))
            if 0 <= x < w and 0 <= y < h:
                vals.append(mag[y, x])
        profile[i] = np.mean(vals) if vals else 0
    perp = angles[np.argmin(profile)]
    return (perp + 90) % 180

def estimate_blur_length(gray, perp_deg):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) // 2
    theta = np.radians(perp_deg)
    prof = []
    for r in range(1, max_r):
        x = int(round(cx + r * np.cos(theta)))
        y = int(round(cy - r * np.sin(theta)))
        if 0 <= x < w and 0 <= y < h:
            prof.append(mag[y, x])
    if len(prof) < 2:
        return 10
    prof = np.array(prof)
    inv = prof.max() - prof
    peaks, _ = find_peaks(inv, distance=3, prominence=0.3)
    if len(peaks) >= 2:
        return max(3, int(round(len(prof) / np.mean(np.diff(peaks)))))
    return 10

def deblur(b_img):
    """Run blind estimation + all methods on one image."""
    b_f = b_img.astype(np.float64) / 255.0
    g = cv2.cvtColor(b_img, cv2.COLOR_RGB2GRAY)
    ang = estimate_blur_angle(g)
    perp = (ang + 90) % 180
    length = estimate_blur_length(g, perp)
    psf = make_motion_psf(length, ang)
    return {
        "inverse": to_u8(inverse_filter(b_f, psf)),
        "wiener":  to_u8(wiener_deblur(b_f, psf, balance=0.05)),
        "rl":      to_u8(richardson_lucy_deblur(b_f, psf, iterations=30)),
        "cls":     to_u8(constrained_least_squares(b_f, psf, gamma=0.01)),
    }


if __name__ != "__main__":
    raise SystemExit(0)

# Pick 5 evenly spaced test images
names = (DATA / "splits" / "test.txt").read_text().strip().splitlines()
indices = np.linspace(0, len(names) - 1, 5, dtype=int)
samples = [names[i] for i in indices]

print(f"Generating figures from {len(samples)} sample images...")

rows_data = []
for i, fname in enumerate(samples):
    print(f"  [{i+1}/5] Processing {fname}...")
    b = load(BLURRY / fname)
    s = load(SHARP / fname)
    results = deblur(b)
    rows_data.append((b, s, results))

# --- Figure 1: comparison_grid.png ---
print("Generating comparison_grid.png...")
fig, axes = plt.subplots(5, 6, figsize=(24, 20))
col_labels = ["Blurry", "Inverse Filter", "Wiener Filter", "Richardson-Lucy", "CLS Filter", "Ground Truth"]
method_keys = ["inverse", "wiener", "rl", "cls"]

for r, (b, s, res) in enumerate(rows_data):
    imgs = [b] + [res[k] for k in method_keys] + [s]
    for c, (img, label) in enumerate(zip(imgs, col_labels)):
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        psnr = compute_psnr(img, s)
        if label == "Ground Truth":
            axes[r, c].set_title(f"Reference" if r > 0 else "Ground Truth", fontsize=11)
        elif label == "Blurry":
            axes[r, c].set_title(f"PSNR: {psnr:.1f} dB" if r > 0 else f"Blurry\n{psnr:.1f} dB", fontsize=11)
        else:
            axes[r, c].set_title(f"{psnr:.1f} dB" if r > 0 else f"{label}\n{psnr:.1f} dB", fontsize=11)

fig.suptitle("License Plate Deblurring: Method Comparison", fontsize=16, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES / "comparison_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> comparison_grid.png saved")

# --- Figure 2: deblur_results.png ---
print("Generating deblur_results.png...")
fig, axes = plt.subplots(5, 3, figsize=(15, 25))
for r, (b, s, res) in enumerate(rows_data):
    best = res["wiener"]  # best classical method
    for c, (img, label) in enumerate(zip(
        [b, best, s],
        ["Blurry Input", "Wiener Deblurred", "Clean Ground Truth"]
    )):
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        if r == 0:
            psnr = compute_psnr(img, s)
            axes[r, c].set_title(f"{label}\nPSNR: {psnr:.1f} dB" if label != "Clean Ground Truth" else label, fontsize=12)
        else:
            psnr = compute_psnr(img, s)
            axes[r, c].set_title(f"PSNR: {psnr:.1f} dB" if label != "Clean Ground Truth" else "Reference", fontsize=11)

fig.suptitle("License Plate Deblurring Results", fontsize=16, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES / "deblur_results.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> deblur_results.png saved")

# --- Figure 3: fft_analysis.png ---
print("Generating fft_analysis.png...")
fig, axes = plt.subplots(5, 3, figsize=(18, 30))
for r, (b, s, res) in enumerate(rows_data):
    best = res["wiener"]
    fft_b = compute_fft_magnitude(b)
    fft_d = compute_fft_magnitude(best)
    fft_c = compute_fft_magnitude(s)
    for c, (spec, label) in enumerate(zip(
        [fft_b, fft_d, fft_c],
        ["Blurry (FFT)", "Wiener Deblurred (FFT)", "Clean Ground Truth (FFT)"]
    )):
        axes[r, c].imshow(spec, cmap="magma")
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(label, fontsize=12)

fig.suptitle("Frequency Domain Analysis (FFT Magnitude Spectrum)\nBlur attenuates high frequencies — classical deconvolution recovers them\n(Connects to Lab 2: Frequency Filtering)", fontsize=14, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES / "fft_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> fft_analysis.png saved")

# --- Figure 4: edge_comparison.png ---
print("Generating edge_comparison.png...")
fig, axes = plt.subplots(5, 6, figsize=(24, 20))
for r, (b, s, res) in enumerate(rows_data):
    best = res["wiener"]
    canny_b = compute_canny_edges(b)
    canny_d = compute_canny_edges(best)
    canny_c = compute_canny_edges(s)
    sobel_b = compute_sobel_edges(b)
    sobel_d = compute_sobel_edges(best)
    sobel_c = compute_sobel_edges(s)

    imgs = [canny_b, canny_d, canny_c, sobel_b, sobel_d, sobel_c]
    labels = [
        "Blurry (Canny)", "Deblurred (Canny)", "Clean (Canny)",
        "Blurry (Sobel)", "Deblurred (Sobel)", "Clean (Sobel)",
    ]
    for c, (img, label) in enumerate(zip(imgs, labels)):
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(label, fontsize=11)

fig.suptitle("Edge Detection Comparison: Canny & Sobel\nDeblurring restores edges lost to blur\n(Connects to Lab 3: Edge Detection)", fontsize=14, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES / "edge_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> edge_comparison.png saved")

print("\nAll 4 figures regenerated successfully.")
