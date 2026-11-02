"""
Function-level bit-exactness checks of the optimized MuSc-V2 pieces against a reference (original) tree,
on synthetic inputs chosen to hit edge cases (crowded patches, wide value ranges, multi-step grouping,
duplicate points). Complements run_eval.py, which compares the end-to-end outputs.

    python benchmarks/check_equivalence.py --ref <original tree> [--device cuda:0]
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())


def check_cross_modal(ref, new, device, rng):
    ok = True
    for trial in range(10):
        image_size, L, Np, R = 224, 784, 1024, int(rng.integers(5, 120))
        if trial >= 6:
            # 16 / 14 px patches with dense clusters: buckets with >= 64 points per patch, R a multiple of 4
            L, R = (196, 256)[trial % 2], 4 * int(rng.integers(1, 32))
        if trial % 2 == 0 and trial < 6:
            pix = rng.choice(image_size * image_size, Np, replace=False)
        else:   # clustered points: some patches get many points
            n_clusters, spread = (20, 6) if trial < 6 else (4, 14)
            centers = rng.integers(0, image_size, size=(n_clusters, 2))
            pts = centers[rng.integers(0, n_clusters, Np)] + rng.integers(-spread, spread + 1, size=(Np, 2))
            pts = np.clip(pts, 0, image_size - 1)
            pix = np.unique(pts[:, 0] * image_size + pts[:, 1])
            pix = np.concatenate([pix, rng.choice(np.setdiff1d(np.arange(image_size ** 2), pix), Np - len(pix), replace=False)])
            rng.shuffle(pix)
        idx = torch.tensor(pix, device=device).unsqueeze(0).repeat(2, 1)
        p2p_ref, p2p_new = ref.patch2point(L, idx, image_size), new.patch2point(L, idx, image_size)
        q2p_ref, q2p_new = ref.point2patch(L, idx, image_size), new.point2patch(L, idx, image_size)
        for img in range(2):
            # trials >= 4 spread values over many binades so that summation order would show
            wide = (lambda n: 10.0 ** rng.uniform(-5, 0.3, (n, R))) if trial >= 4 else (lambda n: 1.0)
            z2d = torch.tensor(rng.uniform(0, 1.5, (L, R)) * wide(L), device=device).half()
            z3d = torch.tensor(rng.uniform(0, 1.5, (Np, R)) * wide(Np), device=device).half()
            for name, a, b in [
                ('2d<-3d', ref.cross_modal_align(z2d[None], z3d[None], p2p_ref[img:img + 1]),
                 new.cross_modal_align(z2d[None], z3d[None], p2p_new[img:img + 1])),
                ('3d<-2d', ref.cross_modal_align(z3d[None], z2d[None], q2p_ref[img:img + 1]),
                 new.cross_modal_align(z3d[None], z2d[None], q2p_new[img:img + 1])),
            ]:
                if not same(a, b):
                    ok = False
                    print(f'  cross_modal_align {name} trial {trial}: MISMATCH max|diff|={(a - b).abs().max().item():.3e}')
        kmax = max(int(p.shape[1]) for _, p in p2p_new[0])
        print(f'  cross_modal trial {trial}: R={R} max points/patch={kmax} {"ok" if ok else "FAIL"}')
    return ok


def check_iter_group(ref_ipg, new_ipg, knn_cls, device, rng, group_sizes=(128, 200)):
    import tifffile
    ok = True
    files = sorted(glob.glob(os.path.join(os.path.dirname(HERE), 'data', 'mini3d_*', '*', 'test', '*', 'xyz', '*.tiff')))
    picks = [files[i] for i in rng.choice(len(files), 3, replace=False)] if files else []
    clouds = []
    for f in picks:
        pc = torch.tensor(tifffile.imread(f)).permute(2, 0, 1)[None]
        pc = torch.nn.functional.interpolate(pc, size=(224, 224), mode='nearest')[0].permute(1, 2, 0).reshape(-1, 3)
        clouds.append(pc[torch.all(pc != 0, dim=1)])
    # rough synthetic surface with duplicated points
    pts = torch.tensor(rng.normal(size=(6000, 3)), dtype=torch.float32)
    clouds.append(torch.cat([pts, pts[:500]]))
    for ci, xyz in enumerate(clouds):
        xyz = xyz[None].to(device)
        center_idx = torch.tensor(np.sort(rng.choice(xyz.shape[1], 1024, replace=False)), dtype=torch.int32, device=device)[None]
        for gs in group_sizes:
            knn = knn_cls(k=gs * 2)
            _, idx_all = knn(xyz.cpu(), xyz.cpu())
            xyz_cpu = xyz.cpu()
            _, idx_center = knn(xyz_cpu, xyz_cpu[torch.arange(1).unsqueeze(-1), center_idx.long().cpu()])
            if not same(idx_all[0][center_idx[0].long().cpu()], idx_center[0]):
                ok = False
                print(f'  knn cloud {ci} gs {gs}: center rows differ from full query')
            a = ref_ipg.iter_group(xyz, center_idx, idx_all.to(device), gs)
            b = new_ipg.iter_group(xyz, center_idx, idx_center.to(device), gs)
            res = [same(a[0], b[0]), same(a[1], b[1]), same(a[2], b[2])]
            n_exp = int((~(np.asarray(a[1]) < 0.01)).sum())
            print(f'  iter_group cloud {ci} ({xyz.shape[1]} pts) group_size {gs}: expanded {n_exp}/1024  '
                  f'idx/curv/neigh identical: {res}')
            ok &= all(res)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='original code tree')
    p.add_argument('--new', default=os.path.dirname(HERE))
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # '' deletes the variable on Windows
    rng = np.random.default_rng(0)
    ref_msm = load(os.path.join(args.ref, 'models', 'modules', '_MSM.py'), 'ref_msm')
    new_msm = load(os.path.join(args.new, 'models', 'modules', '_MSM.py'), 'new_msm')
    ref_ipg = load(os.path.join(args.ref, 'models', 'modules', '_IPG.py'), 'ref_ipg')
    new_ipg = load(os.path.join(args.new, 'models', 'modules', '_IPG.py'), 'new_ipg')
    sys.path.insert(0, args.new)
    from utils.pointnet2_utils import KNN_cpu
    with torch.no_grad():
        ok = check_cross_modal(ref_msm, new_msm, args.device, rng)
        ok &= check_iter_group(ref_ipg, new_ipg, KNN_cpu, args.device, rng)
    print('ALL IDENTICAL' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
