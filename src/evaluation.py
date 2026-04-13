import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two uint8 images."""
    return psnr_metric(img1, img2, data_range=255)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM between two uint8 RGB images."""
    return ssim_metric(img1, img2, data_range=255, channel_axis=2)


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """FFT magnitude spectrum, log scaled and shifted to center."""
    gray = _to_gray(image)
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    return np.log1p(np.abs(fshift))


def compute_canny_edges(image, low=50, high=150):
    return cv2.Canny(_to_gray(image), low, high)


def compute_sobel_edges(image):
    gray = _to_gray(image)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)
    return magnitude


def _to_gray(image):
    """Convert RGB to grayscale if needed."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
