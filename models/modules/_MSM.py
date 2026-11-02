import torch
from tqdm import tqdm


# The cross-modal project lists hold, per image, groups (org_idx (M,), cross_idx (M, k)): the M entries
# org_idx of one modality each project to the k entries in the matching row of cross_idx of the other modality.
# Entries are grouped by k so they can be averaged together; entries projecting to nothing are left out.

def patch2point(patch_num, nonzero_indices_center_list, image_size):
    # (B, L, C) (B, N) (B, N, group_size)
    patch_num_ = int(patch_num**0.5)
    patch_size = int(image_size // patch_num_)
    patch_coor = (nonzero_indices_center_list//image_size//patch_size, nonzero_indices_center_list%image_size//patch_size)
    patch_idx = patch_coor[0]*patch_num_+patch_coor[1]
    in_patch = (patch_coor[1] < patch_num_) & (patch_idx < patch_num)
    patch_idx = torch.where(in_patch, patch_idx, patch_num)   # patch_num: in no patch
    patch2point_list = []
    for img_idx in range(nonzero_indices_center_list.shape[0]):
        patch2point_img_list = []
        # points sorted by patch, keeping ascending point order inside a patch
        point_order = torch.sort(patch_idx[img_idx], stable=True)[1]
        point_num = torch.bincount(patch_idx[img_idx], minlength=patch_num+1)[:patch_num]
        point_start = torch.cumsum(point_num, 0) - point_num
        for k in torch.unique(point_num).tolist():
            if k == 0:
                continue
            patches = torch.nonzero(point_num == k).squeeze(1)
            points = point_order[point_start[patches].unsqueeze(1) + torch.arange(k, device=patches.device)]
            patch2point_img_list.append((patches, points))
        patch2point_list.append(patch2point_img_list)
    return patch2point_list


def point2patch(patch_num, nonzero_indices_center_list, image_size):
    # (B, L, C) (B, N)
    patch_num_ = int(patch_num**0.5)
    patch_size = int(image_size // patch_num_)
    point_coor = (nonzero_indices_center_list//image_size, nonzero_indices_center_list%image_size)
    patch_coor = (point_coor[0]//patch_size, point_coor[1]//patch_size)
    patch_idx = patch_coor[0]*patch_num_+patch_coor[1]
    point_idx = torch.arange(patch_idx.shape[1], device=patch_idx.device)
    # every point projects to exactly one patch
    return [[(point_idx, patch_idx[img_idx].unsqueeze(1))] for img_idx in range(patch_idx.shape[0])]


def cross_modal_align(Z_org, Z_cross, proj_index):
    # cross-modal alignment
    image_num, patch_num, _ = Z_org.shape
    _, _, c_cross = Z_cross.shape
    Z_org_cross = torch.zeros((image_num, patch_num, c_cross), device=Z_org.device)
    for img_idx in range(image_num):
        for org_idx, cross_idx in proj_index[img_idx]:
            if cross_idx.shape[1] < 64:
                # (k, M, C) -> mean over the k projected entries -> (M, C). For k < 64 the CUDA reduction sums
                # each output sequentially, exactly as the per-entry (k, C) mean does, whatever M is.
                patch_proj_cross_features = Z_cross[img_idx][cross_idx.T]
                Z_org_cross[img_idx, org_idx] = patch_proj_cross_features.mean(0).to(Z_org_cross.dtype)
            else:
                # larger k may be split across warps depending on the output count, so keep one mean per entry
                for patch_idx, patch_proj_cross_idx in zip(org_idx.tolist(), cross_idx):
                    Z_org_cross[img_idx, patch_idx] = Z_cross[img_idx][patch_proj_cross_idx].mean(0)
    return Z_org_cross


def replace_sorted(source, target):
    _, indices_sourse = torch.sort(source, dim=-1)
    _, indices_target = torch.sort(target, dim=-1)
    indices_target_inv = torch.argsort(indices_target, dim=-1)
    indices_to_replace = torch.gather(indices_sourse, dim=-1, index=indices_target_inv)
    replaced_target = torch.gather(source, dim=-1, index=indices_to_replace)
    return replaced_target


def compute_scores_single(Z, i, topmin_max=0.3):
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    ref_image_num = Z_ref.shape[0]
    # L2 distance
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, ref_image_num, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_max = int(patch2image.shape[1]*topmin_max)
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)


def compute_scores_multiple(Z_2d, Z_3d, i, topmin_max=0.3, patch2point_list=None, point2patch_list=None):
    # compute anomaly scores
    image_num, patch_num_2d, c_2d = Z_2d.shape
    _, patch_num_3d, c_3d = Z_3d.shape
    Z_2d_ref = torch.cat((Z_2d[:i], Z_2d[i+1:]), dim=0)
    Z_3d_ref = torch.cat((Z_3d[:i], Z_3d[i+1:]), dim=0)
    ref_image_num = Z_2d_ref.shape[0]
    # L2 distance
    patch2image_2d_org = torch.cdist(Z_2d[i:i+1], Z_2d_ref.reshape(-1, c_2d)).reshape(patch_num_2d, ref_image_num, patch_num_2d)
    patch2image_3d_org = torch.cdist(Z_3d[i:i+1], Z_3d_ref.reshape(-1, c_3d)).reshape(patch_num_3d, ref_image_num, patch_num_3d)
    patch2image_2d_org, _ = torch.min(patch2image_2d_org, -1)
    patch2image_3d_org, _ = torch.min(patch2image_3d_org, -1)
    patch2image_2d_cross = cross_modal_align(patch2image_2d_org.unsqueeze(0), patch2image_3d_org.unsqueeze(0), patch2point_list[i:i+1])[0]
    patch2image_3d_cross = cross_modal_align(patch2image_3d_org.unsqueeze(0), patch2image_2d_org.unsqueeze(0), point2patch_list[i:i+1])[0]
    # calculate the confidence weight
    confidence_2d = 1 - replace_sorted(patch2image_2d_org.std(-1), patch2image_2d_cross.std(-1))
    confidence_3d = 1 - replace_sorted(patch2image_3d_org.std(-1), patch2image_3d_cross.std(-1))
    # rescale into value range of another modal
    patch2image_2d_cross_zero = patch2image_2d_cross.sum(-1) != 0
    patch2image_3d_cross = replace_sorted(patch2image_3d_org.reshape(-1), patch2image_3d_cross.reshape(-1)).reshape(patch_num_3d, ref_image_num)
    patch_no0_num = patch2image_2d_org[patch2image_2d_cross_zero].shape[0]
    patch2image_2d_org = patch2image_2d_org.float()
    patch2image_2d_cross = patch2image_2d_cross.float()
    tmp = replace_sorted(patch2image_2d_org[patch2image_2d_cross_zero].reshape(-1), patch2image_2d_cross[patch2image_2d_cross_zero].reshape(-1))
    patch2image_2d_cross[patch2image_2d_cross_zero] = tmp.reshape(patch_no0_num, ref_image_num)
    patch2image_2d_cross, _ = torch.cat((patch2image_2d_org.unsqueeze(-1), patch2image_2d_cross.unsqueeze(-1)), dim=-1).max(-1)
    patch2image_3d_cross, _ = torch.cat((patch2image_3d_org.unsqueeze(-1), patch2image_3d_cross.unsqueeze(-1)), dim=-1).max(-1)
    # anomaly enhancement
    patch2image_2d_cross *= confidence_2d.unsqueeze(-1)
    patch2image_3d_cross *= confidence_3d.unsqueeze(-1)
    patch2image_2d_cross = replace_sorted(patch2image_2d_org.reshape(-1), patch2image_2d_cross.reshape(-1))
    patch2image_2d_cross = patch2image_2d_cross.reshape(patch_num_2d, ref_image_num)
    patch2image_3d_cross = replace_sorted(patch2image_3d_org.reshape(-1), patch2image_3d_cross.reshape(-1))
    patch2image_3d_cross = patch2image_3d_cross.reshape(patch_num_3d, ref_image_num)
    patch2image_2d = (patch2image_2d_org + patch2image_2d_cross) / 2
    patch2image_3d = (patch2image_3d_org + patch2image_3d_cross) / 2
    # interval average
    k_max = max(1, int(ref_image_num*topmin_max))
    vals, _ = torch.topk(patch2image_2d.float(), k_max, largest=False, sorted=True)
    patch2image_2d = torch.mean(vals, dim=1)
    vals, _ = torch.topk(patch2image_3d.float(), k_max, largest=False, sorted=True)
    patch2image_3d = torch.mean(vals, dim=1)
    return patch2image_2d, patch2image_3d


def MSM_single(Z, device='cuda:0'):
    # Mutual scoring for only single modal
    anomaly_scores_matrix = torch.tensor([]).float().to(device)
    for i in tqdm(range(Z.shape[0])):
        anomaly_scores_i = compute_scores_single(Z, i).unsqueeze(0)   # (1, L)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.float()), dim=0)    # (N, L)
    return anomaly_scores_matrix


def MSM_multiple(Z_2d, Z_3d, device='cuda:0', patch2point_list=None, point2patch_list=None):
    # Mutual scoring for 2D+3D multimodal
    # cross-modal alignment
    anomaly_scores_2d_matrix = torch.tensor([]).float().to(device)
    anomaly_scores_3d_matrix = torch.tensor([]).float().to(device)
    for i in tqdm(range(Z_2d.shape[0])):
        anomaly_scores_i_2d_cross, anomaly_scores_i_3d_cross = compute_scores_multiple(Z_2d, Z_3d, i, patch2point_list=patch2point_list, point2patch_list=point2patch_list)   # (1, L)
        anomaly_scores_i_2d = anomaly_scores_i_2d_cross.unsqueeze(0)
        anomaly_scores_i_3d = anomaly_scores_i_3d_cross.unsqueeze(0)
        anomaly_scores_2d_matrix = torch.cat((anomaly_scores_2d_matrix, anomaly_scores_i_2d.float()), dim=0)    # (N, L)
        anomaly_scores_3d_matrix = torch.cat((anomaly_scores_3d_matrix, anomaly_scores_i_3d.float()), dim=0)    # (N, L)

    return anomaly_scores_2d_matrix, anomaly_scores_3d_matrix
