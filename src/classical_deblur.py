import numpy as np
from skimage import restoration


def make_motion_psf(length, angle):
    """Create a motion blur PSF with given length and angle."""
    if length < 1:
        length = 1
    kernel = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    cos_val = np.cos(np.radians(angle))
    sin_val = np.sin(np.radians(angle))
    for i in range(length):
        offset = i - center
        xi = int(round(center + offset * cos_val))
        yi = int(round(center + offset * sin_val))
        if 0 <= xi < length and 0 <= yi < length:
            kernel[yi, xi] = 1.0
    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        kernel[center, center] = 1.0
    return kernel


def inverse_filter(blurry_image, psf, epsilon=1e-3):
    """Naive inverse filtering in frequency domain.

    Demonstrates noise amplification at frequencies where |K| -> 0.
    """
    return _apply_per_channel(blurry_image, psf, lambda ch, p: _inverse_2d(ch, p, epsilon))


def wiener_deblur(blurry_image, psf, balance=0.05):
    """Wiener deconvolution with known PSF.

    Optimal linear filter when noise-to-signal ratio (balance) is known.
    """
    return _apply_per_channel(blurry_image, psf, lambda ch, p: restoration.wiener(ch, p, balance))


def unsupervised_wiener_deblur(blurry_image, psf):
    """Unsupervised Wiener deconvolution.

    Auto-estimates the noise-to-signal ratio using an iterative Gibbs sampler,
    eliminating the need to manually tune the balance parameter.
    """
    return _apply_per_channel(
        blurry_image, psf, lambda ch, p: restoration.unsupervised_wiener(ch, p)[0],
    )


def richardson_lucy_deblur(blurry_image, psf, iterations=30):
    """Richardson-Lucy iterative deconvolution.

    Bayesian approach that maximises the likelihood of the observed image
    under a Poisson noise model.  More robust to PSF mismatch than Wiener.
    """
    blurry_image = np.asarray(blurry_image, dtype=np.float64)
    # RL requires strictly positive values
    blurry_image = np.clip(blurry_image, 1e-6, None)
    return _apply_per_channel(
        blurry_image, psf,
        lambda ch, p: restoration.richardson_lucy(ch, p, num_iter=iterations, clip=False),
        skip_cast=True,
    )


def constrained_least_squares(blurry_image, psf, gamma=0.01):
    """Constrained Least Squares (CLS) filtering.

    Minimises ||Lf||^2 subject to ||g - Hf||^2 = ||n||^2, where L is the
    Laplacian operator. Pedagogical frequency-domain baseline; empirically
    under-performs Wiener under our blind-estimation conditions.
    """
    return _apply_per_channel(blurry_image, psf, lambda ch, p: _cls_2d(ch, p, gamma))


def tv_denoise(image, weight=0.05):
    """Total Variation denoising (Chambolle algorithm).

    Reduces ringing artifacts from deconvolution while preserving edges.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        return np.clip(
            restoration.denoise_tv_chambolle(image, weight=weight, channel_axis=2),
            0.0, 1.0,
        )
    return np.clip(restoration.denoise_tv_chambolle(image, weight=weight), 0.0, 1.0)


# -- internal helpers --

def _apply_per_channel(image, psf, fn, skip_cast=False):
    """Apply a 2-D deconvolution function per channel, clipping to [0, 1]."""
    if not skip_cast:
        image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    if image.ndim == 3:
        return np.clip(
            np.stack([fn(image[:, :, c], psf)
                      for c in range(image.shape[2])], axis=2),
            0.0, 1.0,
        )
    return np.clip(fn(image, psf), 0.0, 1.0)


def _inverse_2d(image_2d, psf, epsilon):
    h, w = image_2d.shape
    kh, kw = psf.shape
    psf_pad = np.zeros((h, w), dtype=np.float64)
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, -(kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(kw // 2), axis=1)
    H = np.fft.fft2(psf_pad)
    H_inv = np.conj(H) / (np.abs(H) ** 2 + epsilon)
    return np.real(np.fft.ifft2(np.fft.fft2(image_2d) * H_inv))


def _cls_2d(image_2d, psf, gamma):
    h, w = image_2d.shape
    kh, kw = psf.shape
    psf_pad = np.zeros((h, w), dtype=np.float64)
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, -(kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(kw // 2), axis=1)
    lap = np.zeros((h, w), dtype=np.float64)
    lap[:3, :3] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    lap = np.roll(lap, -1, axis=0)
    lap = np.roll(lap, -1, axis=1)
    H = np.fft.fft2(psf_pad)
    P = np.fft.fft2(lap)
    H_cls = np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2 + 1e-10)
    return np.real(np.fft.ifft2(np.fft.fft2(image_2d) * H_cls))
