# MuSc-V2 inference speed / exactness benchmarks

Tools to check that inference optimizations keep the outputs **bit-identical** and to measure the speedup.

## Datasets

`make_mini_datasets.py` builds two small MVTec-3D-AD-formatted sets under `./data` from an MVTec AD copy:

| name | classes | images |
|---|---|---|
| `mini3d_single` | hazelnut | 100 (one realistic-size category; MSM is O(N²) per category) |
| `mini3d_multi` | bottle, metal_nut, screw | 103 (exercises the per-category loop) |

RGB and GT masks are real MVTec AD data; the `xyz/*.tiff` point clouds are synthetic (dome + image relief + noise,
with scanner-like holes). They are for timing and equality only, not accuracy numbers.

```bash
python benchmarks/make_mini_datasets.py --mvtec_ad ~/mvtec_anomaly_detection
```

## End-to-end comparison

`run_eval.py` runs `MuScV2.make_category_data` from any code tree and saves `pr_px` / `pr_sp` per category.
Without the DINO / Point-MAE checkpoints it uses seeded random weights; without `pointnet2_ops` it uses the
CPU FPS stand-in in `shims/`.

```bash
# reference tree = a clean export of the original commit
git archive b5e911d | tar -x -C ../musc_baseline
python benchmarks/run_eval.py --code ../musc_baseline --data data/mini3d_multi --modality 2d+3d --stages --out base.npz
python benchmarks/run_eval.py --data data/mini3d_multi --modality 2d+3d --stages --out opt.npz
python benchmarks/run_eval.py --compare base.npz opt.npz
```

`--device cpu` is the default (sets `CUDA_VISIBLE_DEVICES=-1`; an empty value would *unset* it on Windows).
Use `--device 0` for GPU runs. `--max_images N` subsamples each category; `--stages` prints per-stage
wall time; on GPU the peak memory per category is reported as well.

See [RESULTS.md](RESULTS.md) for the measured speedups and the exactness argument for each change.

## Function-level checks

```bash
python benchmarks/check_equivalence.py --ref ../musc_baseline [--device cuda:0]
```

Compares the changed functions (`patch2point` / `point2patch` / `cross_modal_align`, center-only KNN +
`iter_group`) against the original on edge-case inputs: crowded patches, values spread over many binades,
the multi-step grouping fallback, duplicate points.
