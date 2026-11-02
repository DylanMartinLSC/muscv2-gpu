"""
Speed / exactness harness for MuSc-V2 inference.

Runs MuScV2.make_category_data from a given code tree on a (small) dataset, records wall time per
category and saves the raw outputs (pixel anomaly maps + image scores) so two code trees can be
compared bit-for-bit.

    # run (CPU by default so it can share the machine with a GPU training job)
    python benchmarks/run_eval.py --code <tree> --data ./data/mini3d_multi --modality 2d+3d --out base.npz
    # compare
    python benchmarks/run_eval.py --compare base.npz opt.npz

If the DINO / Point-MAE checkpoints are not available locally, seeded random weights are used
(the outputs are then meaningless as anomaly scores but still valid for timing and equality).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--code', default=os.path.dirname(HERE), help='root of the MuSc-V2 code tree to run')
    p.add_argument('--data', default=os.path.join(os.path.dirname(HERE), 'data', 'mini3d_multi'))
    p.add_argument('--dataset_name', default='mvtec_3d')
    p.add_argument('--classes', nargs='+', default=None)
    p.add_argument('--modality', default='2d+3d', choices=['2d', '3d', '2d+3d'])
    p.add_argument('--img_resize', type=int, default=224)
    p.add_argument('--device', default='cpu', help="'cpu' or a cuda index such as '0'")
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--out', default=None, help='npz file to save outputs to')
    p.add_argument('--profile', default=None, help='write cProfile stats to this file')
    p.add_argument('--metrics', action='store_true', help='also compute image AUROC / pixel AUROC')
    p.add_argument('--stages', action='store_true', help='print per-stage wall time')
    p.add_argument('--max_images', type=int, default=None, help='evenly subsample each category to this many images')
    p.add_argument('--compare', nargs=2, default=None, metavar=('A', 'B'))
    return p.parse_args()


def compare(a_path, b_path):
    import numpy as np
    a, b = np.load(a_path), np.load(b_path)
    keys = sorted(k for k in a.files if not k.startswith(('time/', 'mem/')))
    ok = set(keys) == set(k for k in b.files if not k.startswith(('time/', 'mem/')))
    if not ok:
        print('key mismatch:', set(keys) ^ set(k for k in b.files if not k.startswith(('time/', 'mem/'))))
    for k in keys:
        if k not in b.files:
            continue
        x, y = a[k], b[k]
        same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
        line = f'{k:40s} {"IDENTICAL" if same else "DIFFERENT"}'
        if not same:
            ok = False
            if x.shape == y.shape:
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                line += f'  max|diff|={d.max():.3e}  n_diff={(d > 0).sum()}'
            else:
                line += f'  shapes {x.shape} vs {y.shape}'
        print(line)
    for k in sorted(k for k in a.files if k.startswith('time/')):
        if k in b.files:
            ta, tb = float(a[k]), float(b[k])
            print(f'{k:40s} {ta:8.2f}s -> {tb:8.2f}s  ({ta / max(tb, 1e-9):.2f}x)')
    print('ALL IDENTICAL' if ok else 'MISMATCH')
    return ok


def main():
    args = parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # '' deletes the variable on Windows
    prof_path = os.path.abspath(args.profile) if args.profile else None
    code = os.path.abspath(args.code)
    data = os.path.abspath(args.data) + '/'
    out = os.path.abspath(args.out) if args.out else None
    os.chdir(code)
    sys.path.insert(0, code)

    import numpy as np
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        import pointnet2_ops  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(HERE, 'shims'))

    import models.muscv2 as muscv2_mod
    from models.backbone.point_transformer import PointTransformer
    import models.backbone.vision_transformer as vits

    # ---- weights: real if available, otherwise seeded random ----
    real_hub = torch.hub.load_state_dict_from_url
    hub_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')

    def load_state_dict_from_url(url, *a, **k):
        if os.path.exists(os.path.join(hub_dir, os.path.basename(url))):
            return real_hub(url, *a, **k)
        g = torch.random.get_rng_state()
        torch.manual_seed(1234)
        sd = vits.__dict__['vit_base'](patch_size=8, num_classes=0).state_dict()
        torch.random.set_rng_state(g)
        return sd
    torch.hub.load_state_dict_from_url = load_state_dict_from_url

    real_ckpt = PointTransformer.load_model_from_ckpt

    def load_model_from_ckpt(self, path):
        if os.path.exists(path):
            return real_ckpt(self, path)
        g = torch.random.get_rng_state()
        torch.manual_seed(4321)
        sd = PointTransformer(group_size=self.group_size, num_group=self.num_group).state_dict()
        torch.random.set_rng_state(g)
        self.load_state_dict(sd, strict=False)
    PointTransformer.load_model_from_ckpt = load_model_from_ckpt

    # ---- capture outputs instead of computing the (slow) metrics ----
    captured = {}

    def capture_metrics(gt_sp, pr_sp, gt_px, pr_px):
        captured.update(gt_sp=gt_sp, pr_sp=pr_sp, gt_px=gt_px, pr_px=pr_px)
        return [0, 0, 0], [0, 0, 0, 0]
    muscv2_mod.compute_metrics = capture_metrics

    if args.max_images:
        load_datasets = muscv2_mod.MuScV2.load_datasets

        def load_datasets_limited(self, *a, **k):
            ds = load_datasets(self, *a, **k)
            if not isinstance(ds, list):
                # keep both normal and anomalous samples
                step = max(1, len(ds.data_to_iterate) // args.max_images)
                ds.data_to_iterate = ds.data_to_iterate[::step][:args.max_images]
            return ds
        muscv2_mod.MuScV2.load_datasets = load_datasets_limited

    classes = args.classes or sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)))
    backbones = {'2d': ['dino_vitbase8'], '3d': ['point-mae'], '2d+3d': ['dino_vitbase8', 'point-mae']}[args.modality]
    cfg = {
        'datasets': {'dataset_name': args.dataset_name, 'data_path': data, 'class_name': classes,
                     'img_resize': args.img_resize},
        'models': {'backbone_name': backbones, 'feature_layers': [3, 7, 11], 'r_list': [1, 3, 5]},
        'device': '0' if args.device == 'cpu' else args.device,
        'testing': {'output_dir': os.path.join(HERE, '.out'), 'vis': False, 'vis_type': 'whole_norm',
                    'save_excel': False},
    }
    model = muscv2_mod.MuScV2(cfg, seed=42)

    # ---- per-stage wall time (inclusive; nested stages overlap) ----
    stage_time = {}

    def timed(owner, attr, label):
        fn = getattr(owner, attr, None)
        if fn is None:
            return

        def wrapper(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_time[label] = stage_time.get(label, 0.0) + time.perf_counter() - t
            return r
        setattr(owner, attr, wrapper)

    if args.stages:
        import models.backbone.point_transformer as pt_mod
        import models.modules._IPG as ipg_mod
        import utils.pointnet2_utils as pn2_mod
        timed(muscv2_mod.MuScV2, 'load_backbone', 'load_backbone')
        timed(vits.VisionTransformer, 'get_intermediate_layers', 'dino_forward')
        timed(pt_mod.PointTransformer, 'forward', 'pointmae_forward(total)')
        timed(pt_mod, 'fps', '  fps')
        timed(pn2_mod.KNN_cpu, 'forward', '  knn')
        timed(pt_mod, 'iter_group', '  iter_group')
        timed(ipg_mod, 'closed_k', '    closed_k')
        timed(muscv2_mod.SNAMD, '_embed_2d', 'snamd_2d')
        timed(muscv2_mod.SNAMD, '_embed_3d', 'snamd_3d')
        for name in ['patch2point', 'point2patch', 'MSM_single', 'MSM_multiple', 'interpolating_points', 'RsCon']:
            timed(muscv2_mod, name, name)
        import models.modules._MSM as msm_mod
        timed(msm_mod, 'cross_modal_align', '  cross_modal_align')
        timed(msm_mod, 'compute_scores_multiple', '  compute_scores_multiple')
        timed(msm_mod, 'compute_scores_single', '  compute_scores_single')

    prof = None
    if args.profile:
        import cProfile
        prof = cProfile.Profile()

    results = {}
    t_all = time.perf_counter()
    for cls in classes:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if prof:
            prof.enable()
        model.make_category_data(category=cls)
        if prof:
            prof.disable()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        results[f'time/{cls}'] = dt
        results[f'{cls}/pr_px'] = np.asarray(captured['pr_px'])
        results[f'{cls}/pr_sp'] = np.asarray(captured['pr_sp'])
        line = f'[{cls}] {dt:.2f}s  n={len(captured["gt_sp"])}'
        if torch.cuda.is_available():
            results[f'mem/{cls}'] = torch.cuda.max_memory_allocated() / 2 ** 20
            line += f'  peak_mem={results[f"mem/{cls}"]:.0f}MiB'
        if args.metrics:
            from sklearn.metrics import roc_auc_score
            line += f'  auroc_sp={roc_auc_score(captured["gt_sp"], captured["pr_sp"]):.4f}'
            line += f'  auroc_px={roc_auc_score(captured["gt_px"].ravel(), captured["pr_px"].ravel()):.4f}'
        print(line, flush=True)
    results['time/total'] = time.perf_counter() - t_all
    print(f'total {results["time/total"]:.2f}s')
    for k, v in stage_time.items():
        print(f'  stage {k:30s} {v:8.2f}s')

    if out:
        np.savez(out, **results)
        print('saved', out)
    if prof:
        prof.dump_stats(prof_path)


if __name__ == '__main__':
    main()
