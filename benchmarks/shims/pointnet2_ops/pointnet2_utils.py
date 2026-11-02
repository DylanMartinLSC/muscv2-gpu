"""
CPU stand-in for the pointnet2_ops CUDA extension, used only by the benchmark harness when the real
extension is not installed (e.g. CPU-only eval runs). Mirrors the CUDA kernels' semantics: FPS starts
at index 0, skips near-origin points (mag <= 1e-3) and keeps the running min-distance in float32.
"""
import numpy as np
import torch


def furthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    out = np.zeros((B, npoint), dtype=np.int32)
    for b in range(B):
        pts = xyz[b].detach().float().cpu().numpy()
        x, y, z = (np.ascontiguousarray(pts[:, i]) for i in range(3))
        valid = (x * x + y * y + z * z) > 1e-3
        temp = np.full(N, 1e10, dtype=np.float32)
        temp[~valid] = -1.0
        d = np.empty(N, dtype=np.float32)
        old = 0
        for j in range(1, npoint):
            np.subtract(x, x[old], out=d)
            d *= d
            t = y - y[old]
            d += t * t
            t = z - z[old]
            d += t * t
            np.minimum(temp, d, out=d, where=valid)
            np.copyto(temp, d, where=valid)
            old = int(temp.argmax())
            out[b, j] = old
    return torch.from_numpy(out).to(xyz.device)


def gather_operation(features, idx):
    # features (B, C, N), idx (B, npoint) -> (B, C, npoint)
    idx = idx.long()
    return torch.gather(features, 2, idx.unsqueeze(1).expand(-1, features.shape[1], -1))
