"""Evaluate all deconvolution methods on the test set, with per-blur-type breakdown."""
import sys, json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks
from src.classical_deblur import (
    make_motion_psf, inverse_filter, wiener_deblur,
    unsupervised_wiener_deblur, richardson_lucy_deblur,
    constrained_least_squares, tv_denoise,
)
from src.evaluation import compute_psnr, compute_ssim


def load_image_pil(path):
    return np.array(Image.open(path).convert("RGB"))


def estimate_blur_angle(gray):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) // 2
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
    prof = np.array(prof)
    inv = prof.max() - prof
    peaks, _ = find_peaks(inv, distance=3, prominence=0.3)
    if len(peaks) >= 2:
        return max(3, int(round(len(prof) / np.mean(np.diff(peaks)))))
    return 10


def evaluate_subset(names, blurry_dir, sharp_dir, metadata, label=""):
    to_u8 = lambda x: (x * 255).clip(0, 255).astype(np.uint8)
    METHOD_KEYS = ["blurry", "inverse", "wiener", "unsup_wiener", "rl", "cls"]
    metrics = {k: {"psnr": [], "ssim": []} for k in METHOD_KEYS}

    for i, fname in enumerate(names):
        b_img = load_image_pil(blurry_dir / fname)
        s_img = load_image_pil(sharp_dir / fname)
        b_f = b_img.astype(np.float64) / 255.0

        g = cv2.cvtColor(b_img, cv2.COLOR_RGB2GRAY)
        ang = estimate_blur_angle(g)
        perp = (ang + 90) % 180
        length = estimate_blur_length(g, perp)
        psf = make_motion_psf(length, ang)

        outputs = {
            "blurry": b_img,
            "inverse": to_u8(inverse_filter(b_f, psf)),
            "wiener": to_u8(wiener_deblur(b_f, psf)),
            "unsup_wiener": to_u8(unsupervised_wiener_deblur(b_f, psf)),
            "rl": to_u8(richardson_lucy_deblur(b_f, psf, iterations=30)),
            "cls": to_u8(constrained_least_squares(b_f, psf)),
        }

        for key, img in outputs.items():
            metrics[key]["psnr"].append(compute_psnr(img, s_img))
            metrics[key]["ssim"].append(compute_ssim(img, s_img))

        if (i + 1) % 25 == 0 or (i + 1) == len(names):
            print(f"  {label} [{i+1}/{len(names)}]", flush=True)

    return metrics


LABELS = {
    "blurry": "Blurry (input)",
    "inverse": "Inverse filter",
    "wiener": "Wiener (est. PSF)",
    "unsup_wiener": "Unsupervised Wiener",
    "rl": "Richardson-Lucy (n=30)",
    "cls": "CLS filter (g=0.01)",
}
METHOD_KEYS = list(LABELS.keys())


def print_table(metrics, title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"{'Method':<28} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("-" * 48)
    for key in METHOD_KEYS:
        avg_p = np.mean(metrics[key]["psnr"])
        avg_s = np.mean(metrics[key]["ssim"])
        print(f"{LABELS[key]:<28} {avg_p:>10.2f} {avg_s:>8.4f}")


def main():
    data = PROJECT_ROOT / "data"
    split_file = data / "splits" / "test.txt"
    blurry_dir = data / "synthetic" / "blurry"
    sharp_dir = data / "synthetic" / "sharp"
    meta_file = data / "synthetic" / "metadata.json"

    names = split_file.read_text().strip().splitlines()
    metadata = json.load(open(meta_file))

    # Group by blur type
    by_type = defaultdict(list)
    for n in names:
        bt = metadata.get(n, {}).get("blur_type", "unknown")
        by_type[bt].append(n)

    print(f"Test set: {len(names)} images")
    for bt, ns in sorted(by_type.items()):
        print(f"  {bt}: {len(ns)}")

    # Evaluate motion-blurred subset first (pipeline is designed for this)
    if "motion" in by_type:
        print(f"\n--- Evaluating MOTION subset ({len(by_type['motion'])} images) ---")
        motion_metrics = evaluate_subset(
            by_type["motion"], blurry_dir, sharp_dir, metadata, "motion"
        )
        print_table(motion_metrics, "MOTION BLUR (pipeline target)")

    # Evaluate gaussian subset
    if "gaussian" in by_type:
        print(f"\n--- Evaluating GAUSSIAN subset ({len(by_type['gaussian'])} images) ---")
        gauss_metrics = evaluate_subset(
            by_type["gaussian"], blurry_dir, sharp_dir, metadata, "gaussian"
        )
        print_table(gauss_metrics, "GAUSSIAN BLUR (PSF mismatch expected)")

    # Evaluate defocus subset
    if "defocus" in by_type:
        print(f"\n--- Evaluating DEFOCUS subset ({len(by_type['defocus'])} images) ---")
        defocus_metrics = evaluate_subset(
            by_type["defocus"], blurry_dir, sharp_dir, metadata, "defocus"
        )
        print_table(defocus_metrics, "DEFOCUS BLUR (PSF mismatch expected)")

    # Compute overall
    all_metrics = {k: {"psnr": [], "ssim": []} for k in METHOD_KEYS}
    for bt_metrics in [motion_metrics, gauss_metrics, defocus_metrics]:
        for key in METHOD_KEYS:
            all_metrics[key]["psnr"].extend(bt_metrics[key]["psnr"])
            all_metrics[key]["ssim"].extend(bt_metrics[key]["ssim"])
    print_table(all_metrics, f"OVERALL ({len(names)} images, mixed blur)")

    # Save results
    out_path = PROJECT_ROOT / "outputs" / "test_set_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for title, metrics in [
            ("MOTION BLUR", motion_metrics),
            ("GAUSSIAN BLUR", gauss_metrics),
            ("DEFOCUS BLUR", defocus_metrics),
            ("OVERALL", all_metrics),
        ]:
            f.write(f"\n{title} ({len(by_type.get(title.split()[0].lower(), names))} images)\n")
            f.write(f"{'Method':<28} {'PSNR (dB)':>10} {'SSIM':>8}\n")
            f.write("-" * 48 + "\n")
            for key in METHOD_KEYS:
                avg_p = np.mean(metrics[key]["psnr"])
                avg_s = np.mean(metrics[key]["ssim"])
                f.write(f"{LABELS[key]:<28} {avg_p:>10.2f} {avg_s:>8.4f}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
