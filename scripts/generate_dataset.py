# Synthetic dataset generation: blurry/sharp plate pairs
import argparse
import json
import sys
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blur_generator import (
    apply_gaussian_blur,
    apply_motion_blur,
    apply_defocus_blur,
    add_degradation,
    get_blur_kernel,
)
from src.utils import load_image, save_image, ensure_dir, resize_max

BLUR_TYPES = ["gaussian", "motion", "defocus"]

BLUR_FUNCTIONS = {
    "gaussian": apply_gaussian_blur,
    "motion": apply_motion_blur,
    "defocus": apply_defocus_blur,
}


def generate_dataset(config):
    paths = config["paths"]
    clean_dir = Path(paths["clean_dir"])
    output_dir = Path(paths["output_dir"])
    splits_dir = Path(paths["splits_dir"])

    num_variants = config["dataset"]["num_variants_per_image"]
    seed = config["dataset"]["seed"]

    sharp_dir = output_dir / "sharp"
    blurry_dir = output_dir / "blurry"
    kernels_dir = output_dir / "kernels"
    ensure_dir(sharp_dir)
    ensure_dir(blurry_dir)
    ensure_dir(kernels_dir)
    ensure_dir(splits_dir)

    clean_images = sorted(
        [
            f
            for f in clean_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )
    if not clean_images:
        raise FileNotFoundError(f"No images found in {clean_dir}")

    np.random.seed(seed)
    metadata = {}

    max_size = config.get("dataset", {}).get("max_image_size", 512)

    for img_path in tqdm(clean_images, desc="Generating pairs"):
        image = resize_max(load_image(str(img_path)), max_size)
        source_stem = img_path.stem

        for v in range(num_variants):
            blur_type = BLUR_TYPES[v % len(BLUR_TYPES)]
            blur_fn = BLUR_FUNCTIONS[blur_type]

            blur_cfg = config["blur"][blur_type]
            blur_kwargs = {}
            for key, val in blur_cfg.items():
                blur_kwargs[key] = tuple(val)

            blur_params = _sample_blur_params(blur_type, blur_kwargs)

            blurred = blur_fn(image, **blur_kwargs)

            deg_cfg = config["degradation"]
            degraded = add_degradation(
                blurred,
                noise_sigma_range=tuple(deg_cfg["noise_sigma_range"]),
                jpeg_quality_range=tuple(deg_cfg["jpeg_quality_range"]),
            )

            out_name = f"{source_stem}_v{v}.png"

            save_image(image, str(sharp_dir / out_name))
            save_image(degraded, str(blurry_dir / out_name))

            try:
                kernel = get_blur_kernel(blur_type, blur_params)
                np.save(str(kernels_dir / f"{source_stem}_v{v}.npy"), kernel)
            except OSError:
                pass  # Skip kernel save on disk errors (OneDrive sync)

            metadata[out_name] = {
                "blur_type": blur_type,
                "source": source_stem,
                "variant": v,
                "blur_params": {k: float(val) for k, val in blur_params.items()},
            }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Split by source image (not variant) to prevent data leakage
    source_stems = sorted(set(name.rsplit("_v", 1)[0] for name in metadata.keys()))

    ratios = config["split_ratios"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]

    train_sources, valtest_sources = train_test_split(
        source_stems,
        test_size=val_ratio + test_ratio,
        random_state=seed,
    )

    relative_test = test_ratio / (val_ratio + test_ratio)
    val_sources, test_sources = train_test_split(
        valtest_sources,
        test_size=relative_test,
        random_state=seed,
    )

    train_sources_set = set(train_sources)
    val_sources_set = set(val_sources)
    test_sources_set = set(test_sources)

    train_files, val_files, test_files = [], [], []
    for fname in sorted(metadata.keys()):
        source = fname.rsplit("_v", 1)[0]
        if source in train_sources_set:
            train_files.append(fname)
        elif source in val_sources_set:
            val_files.append(fname)
        elif source in test_sources_set:
            test_files.append(fname)

    _write_split_file(splits_dir / "train.txt", train_files)
    _write_split_file(splits_dir / "val.txt", val_files)
    _write_split_file(splits_dir / "test.txt", test_files)

    print(f"Generated {len(metadata)} pairs from {len(clean_images)} source images")
    print(
        f"Splits: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )


def _sample_blur_params(blur_type, blur_kwargs):
    if blur_type == "gaussian":
        sigma_range = blur_kwargs.get("sigma_range", (1.0, 5.0))
        return {"sigma": np.random.uniform(sigma_range[0], sigma_range[1])}
    elif blur_type == "motion":
        length_range = blur_kwargs.get("length_range", (5, 25))
        angle_range = blur_kwargs.get("angle_range", (0, 360))
        return {
            "length": int(np.random.randint(length_range[0], length_range[1] + 1)),
            "angle": np.random.uniform(angle_range[0], angle_range[1]),
        }
    elif blur_type == "defocus":
        radius_range = blur_kwargs.get("radius_range", (3, 10))
        return {"radius": int(np.random.randint(radius_range[0], radius_range[1] + 1))}
    return {}


def _write_split_file(path, filenames):
    with open(path, "w") as f:
        f.write("\n".join(filenames))


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic blurred plate dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--clean-dir", type=str, default=None, help="Override clean plates directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--num-variants", type=int, default=None, help="Override variants per image"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.clean_dir:
        config["paths"]["clean_dir"] = args.clean_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.num_variants:
        config["dataset"]["num_variants_per_image"] = args.num_variants

    generate_dataset(config)


if __name__ == "__main__":
    main()
