import torch
import numpy as np

def closed_k(xyz, support_idx, query_idx, k):
    mask = ~torch.isin(support_idx, query_idx.view(-1))
    diff_tensor = support_idx[mask]
    support_idx = torch.unique(diff_tensor)
    support_xyz = xyz[support_idx]
    query_xyz = xyz[query_idx]
    distance = torch.cdist(query_xyz, support_xyz)
    _, indices = torch.topk(distance.reshape(-1), k, largest=False)
    indices = indices % support_idx.shape[0]
    indices = support_idx[indices]
    return indices


def iter_group(xyz, center_idx, idx_center, group_size):
    # idx_center: (B, Nc, 2*group_size) kNN indices of each center point, nearest first
    K_iter = 80
    Cthr = 0.01
    Nc = center_idx.shape[1]
    xyz, center_idx, idx_center = xyz.squeeze(), center_idx.squeeze(), idx_center.squeeze()

    # compute curvatures
    center_xyz = xyz[center_idx, :]
    neighbors_xyz = xyz[idx_center[:, :group_size]]
    neighbors_xyz = neighbors_xyz[:, 1:, :] - center_xyz.unsqueeze(1)
    neighbors_xyz /= neighbors_xyz.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0]
    neighbors_xyz = neighbors_xyz.permute(0, 2, 1)
    neighbors_xyz_mean = neighbors_xyz.mean(-1, keepdims=True)
    tmpa = neighbors_xyz - neighbors_xyz_mean
    covariance_matrix = torch.bmm(tmpa, tmpa.permute(0, 2, 1))/(group_size-2)   # (Nc, 3, 3)
    eigenvalues = np.linalg.eig(covariance_matrix.cpu().float())[0]
    curvatures = eigenvalues.min(-1) / eigenvalues.sum(-1)

    # group initialization: flat centers take their group_size nearest neighbors, the others start from K_iter
    # (compared per scalar on purpose: numpy scalar and array comparisons can round Cthr differently)
    iter_idx = idx_center[:, :group_size].clone(memory_format=torch.contiguous_format)
    expand = [i for i in range(Nc) if not curvatures[i] < Cthr]
    # group expansion
    if len(expand) > 0:
        rows = idx_center[expand]
        sorted_rows = rows.sort(dim=1)[0]
        if K_iter < group_size < 2 * K_iter and bool((sorted_rows[:, 1:] != sorted_rows[:, :-1]).all()):
            # One expansion step. kNN rows hold distinct points, so the support set of closed_k
            # (search space minus current group, deduplicated and sorted) is the sorted row tail.
            support = rows[:, K_iter:].sort(dim=1)[0]
            last_num = group_size - K_iter
            for j, i in enumerate(expand):
                query_idx = rows[j, :K_iter]
                support_idx = support[j]
                support_xyz = xyz[support_idx]
                query_xyz = xyz[query_idx]
                distance = torch.cdist(query_xyz, support_xyz)
                _, indices = torch.topk(distance.reshape(-1), last_num, largest=False)
                indices = indices % support_idx.shape[0]
                indices = support_idx[indices]
                iter_idx[i] = torch.cat([indices, query_idx], dim=0)
        else:
            for i in expand:
                group = idx_center[i, :K_iter]
                while group.shape[0] != group_size:
                    if group.shape[0] + K_iter <= group_size:
                        new_idx = closed_k(xyz, idx_center[i], group, K_iter)
                    else:
                        last_num = group_size - group.shape[0]
                        new_idx = closed_k(xyz, idx_center[i], group, last_num)
                    group = torch.cat([new_idx, group], dim=0)
                iter_idx[i] = group

    iter_idx = iter_idx.unsqueeze(0)
    iter_idx_ = iter_idx.view(-1)
    center = xyz[center_idx, :]
    neighborhood = xyz.squeeze()[iter_idx_, :]
    neighborhood = neighborhood.reshape(1, Nc, group_size, 3).contiguous()
    neighborhood = neighborhood - center.unsqueeze(0).unsqueeze(2)  # (1, 1024, 128, 3)

    return iter_idx, curvatures, neighborhood
