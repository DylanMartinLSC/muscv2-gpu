"""
Build small (~100 image) MVTec-3D-AD-formatted datasets for quick speed / exactness evals.

RGB images and GT masks come from MVTec AD; the organized point clouds (xyz/*.tiff) are
synthesized (dome + image-driven relief + noise, with missing-point holes) so the 3D branch
exercises the same code paths and roughly the same foreground point counts as MVTec 3D-AD.
The data is not meant for accuracy numbers, only for timing and output-equality checks.

    python benchmarks/make_mini_datasets.py --mvtec_ad C:/path/to/mvtec_anomaly_detection
"""
import argparse
import os
import random

import numpy as np
import tifffile as tiff
from PIL import Image, ImageFilter

SIZE = 448

# name -> {class: (n_good, n_per_defect)}
DATASETS = {
    # one class with a realistic per-class test-set size (MSM cost is O(N^2) per class)
    'mini3d_single': {'hazelnut': (40, 15)},
    # several small classes to cover the per-category loop
    'mini3d_multi': {'bottle': (10, 8), 'metal_nut': (10, 6), 'screw': (10, 5)},
}


def synth_xyz(rgb, rng):
    h = w = SIZE
    gray = np.asarray(rgb.convert('L').filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255.0
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x = u / w - 0.5
    y = v / h - 0.5
    r = np.sqrt(x ** 2 + y ** 2)
    # irregular foreground boundary
    theta = np.arctan2(y, x)
    phase = rng.uniform(0, 2 * np.pi, size=3)
    r0 = 0.40 + 0.03 * np.sin(3 * theta + phase[0]) + 0.02 * np.sin(5 * theta + phase[1])
    fg = r < r0
    # missing-point holes like a real structured-light scanner
    for _ in range(rng.integers(3, 8)):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        rad = rng.integers(3, 12)
        fg &= (v - cy) ** 2 + (u - cx) ** 2 > rad ** 2
    z = 0.35 * np.sqrt(np.clip(r0 ** 2 - r ** 2, 0, None)) + 0.03 * gray
    z += rng.normal(0, 1e-3, size=z.shape).astype(np.float32)
    xyz = np.stack([x, y, z + 0.5], axis=-1).astype(np.float32)
    xyz[~fg] = 0
    return xyz


def build(src_root, dst_root, spec, seed=0):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    total = 0
    for cls, (n_good, n_def) in spec.items():
        test_dir = os.path.join(src_root, cls, 'test')
        for defect in sorted(os.listdir(test_dir)):
            files = sorted(f for f in os.listdir(os.path.join(test_dir, defect)) if f.endswith('.png'))
            files = files[:n_good] if defect == 'good' else files[:n_def]
            out = os.path.join(dst_root, cls, 'test', defect)
            os.makedirs(os.path.join(out, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(out, 'xyz'), exist_ok=True)
            if defect != 'good':
                os.makedirs(os.path.join(out, 'gt'), exist_ok=True)
            for f in files:
                stem = os.path.splitext(f)[0]
                rgb = Image.open(os.path.join(test_dir, defect, f)).convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)
                rgb.save(os.path.join(out, 'rgb', stem + '.png'))
                tiff.imwrite(os.path.join(out, 'xyz', stem + '.tiff'), synth_xyz(rgb, rng))
                if defect != 'good':
                    mask = Image.open(os.path.join(src_root, cls, 'ground_truth', defect, stem + '_mask.png'))
                    mask.convert('L').resize((SIZE, SIZE), Image.NEAREST).save(os.path.join(out, 'gt', stem + '.png'))
                total += 1
        print(f'{dst_root}/{cls}: done')
    print(f'{dst_root}: {total} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvtec_ad', default=os.path.expanduser('~/mvtec_anomaly_detection'))
    parser.add_argument('--out', default='./data')
    args = parser.parse_args()
    for name, spec in DATASETS.items():
        build(args.mvtec_ad, os.path.join(args.out, name), spec)
