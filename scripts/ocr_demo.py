"""Standalone qualitative OCR demo on the 3 sample plates in data/samples/.

Runs blind-PSF Wiener deblurring on each blurry sample, then compares
EasyOCR output against the clean ground-truth plate.

Prints per-sample results to stdout AND writes them to
outputs/ocr_demo_results.txt.

No dataset download required: data/samples/ ships with the repository.

Usage
-----
    pip install easyocr   # one-time; ~500 MB incl. PyTorch + OCR models
    python scripts/ocr_demo.py

Scope
-----
This is a qualitative demonstration, not a quantitative evaluation. A full
character-error-rate (CER) sweep over the 453-image test set is flagged as
future work in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks

from src.classical_deblur import make_motion_psf, wiener_deblur

SAMPLES_BLURRY = PROJECT / "data" / "samples" / "blurry"
SAMPLES_SHARP = PROJECT / "data" / "samples" / "sharp"
OUT_FILE = PROJECT / "outputs" / "ocr_demo_results.txt"


# -- blind PSF estimator (same as scripts/evaluate_test_set.py) ---------

def _estimate_angle(gray: np.ndarray) -> float:
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
        profile[i] = np.mean(vals) if vals else 0.0
    perp = angles[int(np.argmin(profile))]
    return float((perp + 90) % 180)


def _estimate_length(gray: np.ndarray, perp_deg: float) -> int:
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
    prof_arr = np.array(prof)
    inv = prof_arr.max() - prof_arr
    peaks, _ = find_peaks(inv, distance=3, prominence=0.3)
    if len(peaks) >= 2:
        return max(3, int(round(len(prof_arr) / np.mean(np.diff(peaks)))))
    return 10


def deblur_one(blurry_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(blurry_rgb, cv2.COLOR_RGB2GRAY)
    angle = _estimate_angle(g)
    perp = (angle + 90) % 180
    length = _estimate_length(g, perp)
    psf = make_motion_psf(length, angle)
    deblurred = wiener_deblur(blurry_rgb.astype(np.float64) / 255.0, psf, balance=0.05)
    return (deblurred * 255).clip(0, 255).astype(np.uint8)


# -- main ----------------------------------------------------------------

def main() -> int:
    try:
        import easyocr
    except ImportError:
        msg = (
            "EasyOCR is not installed. Install with `pip install easyocr`\n"
            "(adds PyTorch + OCR weights; ~500 MB first time). Re-run this\n"
            "script afterwards to get OCR output on the sample plates."
        )
        print(msg)
        return 1

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    samples = sorted(SAMPLES_BLURRY.glob("*.png"))
    if not samples:
        print(f"No samples found in {SAMPLES_BLURRY}")
        return 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "OCR demonstration (qualitative) -- Wiener blind deblurring",
        "=" * 60,
        f"Samples: {len(samples)} from data/samples/blurry/",
        "",
    ]

    for path in samples:
        fname = path.name
        blurry = np.array(Image.open(path).convert("RGB"))
        sharp_path = SAMPLES_SHARP / fname
        sharp = np.array(Image.open(sharp_path).convert("RGB")) if sharp_path.exists() else None
        deblurred = deblur_one(blurry)

        def read(img: np.ndarray) -> str:
            results = reader.readtext(img)
            if not results:
                return "(nothing detected)"
            return " ".join(r[1] for r in results).strip() or "(nothing detected)"

        txt_b = read(blurry)
        txt_d = read(deblurred)
        txt_s = read(sharp) if sharp is not None else "(no ground truth)"

        lines += [
            f"[{fname}]",
            f"  blurry    -> {txt_b}",
            f"  deblurred -> {txt_d}",
            f"  clean     -> {txt_s}",
            "",
        ]

    output = "\n".join(lines)
    print(output)
    OUT_FILE.write_text(output, encoding="utf-8")
    print(f"Results saved to {OUT_FILE.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
