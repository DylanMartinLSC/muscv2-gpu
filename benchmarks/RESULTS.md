# Faster MuSc-V2 inference with bit-identical outputs

## Summary

Four changes to the inference path cut the wall time of a 100-image 2D+3D category from **123 s to 20 s (6.1×)**
on an RTX 3090. The anomaly maps (`pr_px`) and image scores (`pr_sp`) are **bit-for-bit identical** to the
original code in every configuration tested, on GPU and CPU. Since the outputs are identical, every reported
metric (AUROC, F1, AP, AUPRO) is identical by construction. Peak GPU memory goes down slightly.

| config (GPU, median of 3 runs) | original | optimized | speedup | outputs |
|---|---|---|---|---|
| 2D+3D, 1 category × 100 images | 123.2 s | 20.3 s | **6.07×** | identical |
| 2D+3D, 3 categories × ~34 images | 134.5 s | 27.0 s | **4.98×** | identical |
| 3D only, 3 categories × ~34 images | 92.6 s | 23.7 s | **3.90×** | identical |
| 2D only, 3 categories × ~34 images | 17.0 s | 14.9 s | 1.15× | identical |

Peak GPU memory, 2D+3D, 100 images: 3,294 MiB → 3,219 MiB.

The 2D-only path was already GPU-bound; there it only gains from loading the backbones once per run instead of
once per category. The large gains are in the 3D and cross-modal code, which was dominated by Python-level
loops, per-element GPU kernel launches and host–device syncs, and by a nearest-neighbour search that computed
about 25× more neighbourhoods than were used.

## Changes

### 1. kNN only for the FPS centers (`models/backbone/point_transformer.py`)

`Group.forward` ran sklearn's `kneighbors` (k = 256) for **every** foreground point (~25k per image), but
`iter_group` only ever reads the rows of the 1,024 FPS centers (`idx_all[center_idx]`). The kNN is now
queried with the centers alone (`center` is FPS's gather of those points, so the coordinates are the same
values), and `iter_group` receives those rows directly.

*Exactness:* sklearn's kd-tree answers each query point independently (single-tree, depth-first traversal
per point; `n_jobs=None`), so a query row does not depend on which other points are queried. The tree is
built on the same data as before.

*Effect:* kNN 59.5 s → 3.3 s per 100 images.

### 2. Grouping without per-center syncs (`models/modules/_IPG.py`)

For high-curvature centers, `closed_k` built its support set as
`unique(row[~isin(row, row[:80])])`. Each call needed a data-dependent-size tensor, so every center cost
several host–device syncs. A kNN row holds distinct point indices, so this set is exactly `sort(row[80:])`.
It is now computed for all expanding centers in one batched sort, after a batched check that the rows really
are distinct. The per-center `cdist` and `topk` calls are kept with the **same shapes and inputs** as before,
so they run the same kernels. The original `closed_k` loop is still used when the check fails, or when the
group size needs more than one expansion step (not the case for the shipped `group_size=128`,
`K_iter=80`).

The curvature threshold is still compared per scalar, as in the original. With NumPy 1.x, comparing a
`float32` scalar and a `float32` array against `0.01` round differently at exactly `float32(0.01)`.

*Effect:* `iter_group` 7.9 s → 1.6 s per 100 images. An ablation that keeps the original `closed_k` loop but
applies change 1 runs 23.0 s instead of 20.3 s, so this change saves a further 2.7 s (12%).

### 3. Vectorized cross-modal projection (`models/modules/_MSM.py`)

`patch2point` ran about 8 masked GPU ops for each of the 784 patches of every image, and `point2patch` called
`torch.unique` on a scalar for each of the 1,024 points of every image. `cross_modal_align` then looped over
all 784 + 1,024 entries per image, per layer, with several kernel launches per entry.

- **Membership:** patch membership is now computed with integer arithmetic. A point belongs to patch
  `(row // patch_size, col // patch_size)`, which is identical to the original's four range tests, including
  for non-square patch grids and image sizes that are not a multiple of the patch size. The point lists are
  stored per image, grouped by how many points `k` each patch holds, in ascending point order (stable sort),
  as `torch.where` returned them.
- **Averaging:** `cross_modal_align` averages each group with one `mean(0)` over a `(k, M, R)` tensor instead
  of `M` separate means over `(k, R)` tensors. In both layouts, CUDA's reduction kernel sums every output over
  `k` sequentially in the same order, as long as the reduced dimension is not split across warps. The split
  threshold is `min(16 · block_height, 256)`, with `block_height ≥ 4`, so it never applies for `k < 64`. The
  mean factor `R / (kR)` = `MR / (kMR)` = `fl(1/k)`. Buckets with `k ≥ 64` use the original per-entry mean.
  This only occurs with 14- or 16-pixel-patch backbones on very dense point clouds; it cannot occur with ViT/8
  at 224, where an 8×8 patch would need every pixel to be an FPS center. On CPU, both layouts accumulate in
  float32 row by row and divide once.
- **3D→2D direction:** every point projects to exactly one patch, so this direction is a pure gather.

*Effect:* `patch2point` + `point2patch` 23.5 s → 0.1 s; `cross_modal_align` 16.4 s → 2.2 s per 100 images
(3 layers).

### 4. Build the backbones once (`models/muscv2.py`)

`load_backbone` rebuilt and reloaded DINO and Point-MAE for every category. They are now built once. Point-MAE
is not switched to `eval()`, so DropPath is active at inference and its outputs depend on the RNG stream.
Building the models consumes random numbers right after `setup_seed`, so later categories restore the
torch / CUDA / NumPy / Python RNG states captured right after the first build. This is exactly the state a
rebuild would leave, since building happens on CPU and is a deterministic function of the seed. The cache is
keyed on the seed. Other model state doesn't change the outputs:

- Train-mode BatchNorm only writes its running statistics and never reads them.
- DINO is in `eval()` mode.

*Effect:* ~1 s per additional category with the test weights; more with real checkpoints loaded from disk.

## Verification

All runs used PyTorch 2.6.0+cu118, cuDNN 9.1, an RTX 3090 and Windows 11.

1. **End to end, GPU:** original, optimized and the ablation tree from change 2, each run 3 times on each of
   the 4 configurations in the table. Every run's `pr_px` and `pr_sp` is `np.array_equal` to the original's
   first run (36 runs).
2. **End to end, CPU:** the same 4 configurations, original vs optimized: identical. This also covers change 4
   with DropPath drawing from the CPU RNG across 3 categories.
3. **Function level, CPU and CUDA** (`check_equivalence.py`): the changed functions against the original on
   edge cases. All identical on both devices. The cases:
   - random and clustered projections (up to 112 points per patch, so the `k ≥ 64` path is exercised);
   - 8-, 14- and 16-pixel patch grids;
   - values spread over 5 orders of magnitude, so a changed summation order would show;
   - real-sized point clouds;
   - a cloud with duplicated points where every center expands;
   - `group_size=200`, which exercises the multi-step fallback.

## Limitations

- **Data:** the RGB images and masks are MVTec AD; the point clouds are synthetic, with realistic foreground
  sizes (~25k points) and scanner-like holes.
- **Weights:** seeded random weights stood in for the DINO and Point-MAE checkpoints. Equality does not depend
  on the weights, but absolute timings of the backbones would differ slightly.
- **FPS:** `pointnet2_ops` (CUDA FPS) was not installed, so both trees used the same CPU FPS stand-in
  (`shims/`), which takes ~2.9 s per 100 images. With the real CUDA op, both totals drop by about that much,
  and the speedups grow slightly.
- **Versions:** tested on PyTorch 2.6 only (the README pins 2.0.1). The kernel-level arguments above hold for
  the reduction and top-k code of both versions, but were not run on 2.0.1 or other GPUs.

## Reproduction

```bash
python benchmarks/make_mini_datasets.py --mvtec_ad <MVTec AD root>
git archive b5e911d | tar -x -C ../musc_baseline
python benchmarks/check_equivalence.py --ref ../musc_baseline --device cuda:0
python benchmarks/run_eval.py --device 0 --code ../musc_baseline --data data/mini3d_single --modality 2d+3d --stages --out base.npz
python benchmarks/run_eval.py --device 0 --data data/mini3d_single --modality 2d+3d --stages --out opt.npz
python benchmarks/run_eval.py --compare base.npz opt.npz
```
