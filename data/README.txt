License Plate Deblurring Dataset
=================================
Group Opus — Hamza Ziouine & Leonce Theureau
Introduction to Computer Vision, S8, UIR, Spring 2026

Overview
--------
This dataset contains synthetic blurry/sharp paired license plate images
used to develop and evaluate our deblurring pipeline.

Source Data
-----------
- 1,001 clean license plate images collected from public datasets
- Located in clean_plates/ (not tracked in git due to size)

Generated Data (synthetic/)
---------------------------
- blurry/   : Degraded images with synthetic blur + noise + JPEG artifacts
- sharp/    : Corresponding clean ground-truth images (resized to 512px max)
- kernels/  : Blur kernels saved as .npy files for classical baseline evaluation
- metadata.json : Per-image blur parameters (type, sigma/length/radius, angle)

Blur Types Applied
------------------
1. Gaussian blur   : sigma in [1.0, 5.0]
2. Motion blur     : length in [5, 25] px, angle in [0, 360] degrees
3. Defocus blur    : radius in [3, 10] px

Additional degradation: Gaussian noise (sigma 5-15) + JPEG compression (quality 50-85)

Data Splits (splits/)
---------------------
- train.txt : 2,100 images (70%)
- val.txt   :   450 images (15%)
- test.txt  :   453 images (15%)

Splits are done by source image stem to prevent data leakage between sets.

Generation
----------
Run: python scripts/generate_dataset.py --config config/default.yaml
Configuration: config/default.yaml

Third-Party Data
----------------
"3rd party data/" contains the original dataset (Ground Truth images)
from which clean plates were extracted. Not tracked in git.
