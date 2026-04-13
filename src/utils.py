import cv2
from pathlib import Path


def load_image(path):
    """Load image as RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image, path):
    """Save RGB numpy array as an image file."""
    from PIL import Image

    ensure_dir(Path(path).parent)
    Image.fromarray(image).save(str(path))


def resize_max(image, max_size=512):
    """Resize so the longest side is at most max_size pixels."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
