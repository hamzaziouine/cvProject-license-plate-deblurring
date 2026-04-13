import cv2
import io
import numpy as np
from PIL import Image


def apply_gaussian_blur(
    image: np.ndarray, sigma_range: tuple = (1.0, 5.0)
) -> np.ndarray:
    """Apply Gaussian blur with random sigma from the given range."""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    # make sure kernel size is odd
    ksize = int(np.ceil(sigma * 6)) | 1
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return blurred.astype(np.uint8)


def apply_motion_blur(
    image: np.ndarray,
    length_range: tuple = (5, 25),
    angle_range: tuple = (0, 360),
) -> np.ndarray:
    """Apply motion blur with random length and angle."""
    length = np.random.randint(length_range[0], length_range[1] + 1)
    angle = np.random.uniform(angle_range[0], angle_range[1])

    kernel = _create_motion_kernel(length, angle)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred.astype(np.uint8)


def apply_defocus_blur(image: np.ndarray, radius_range: tuple = (3, 10)) -> np.ndarray:
    """Apply disk (defocus) blur with random radius."""
    radius = np.random.randint(radius_range[0], radius_range[1] + 1)

    kernel = _create_disk_kernel(radius)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred.astype(np.uint8)


def add_degradation(
    image: np.ndarray,
    noise_sigma_range: tuple = (5, 15),
    jpeg_quality_range: tuple = (50, 85),
) -> np.ndarray:
    """Add noise + JPEG compression artifacts."""
    noise_sigma = np.random.uniform(noise_sigma_range[0], noise_sigma_range[1])
    noise = np.random.randn(*image.shape) * noise_sigma
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    quality = np.random.randint(jpeg_quality_range[0], jpeg_quality_range[1] + 1)
    pil_img = Image.fromarray(noisy)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    degraded = np.array(Image.open(buffer))

    return degraded.astype(np.uint8)


def get_blur_kernel(blur_type: str, params: dict) -> np.ndarray:
    if blur_type == "gaussian":
        sigma = params["sigma"]
        ksize = int(np.ceil(sigma * 6)) | 1
        ksize = max(ksize, 3)
        ax = np.arange(ksize) - ksize // 2
        kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = np.outer(kernel_1d, kernel_1d)
        return kernel / kernel.sum()
    elif blur_type == "motion":
        return _create_motion_kernel(params["length"], params["angle"])
    elif blur_type == "defocus":
        return _create_disk_kernel(params["radius"])
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")


def _create_motion_kernel(length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    return kernel / kernel.sum()


def _create_disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    center = radius
    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
    kernel[mask] = 1.0
    return kernel / kernel.sum()
