import torch

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    """
    Calculate per-channel mean and std of a feature map.
    Returns (mean, std), each shape [B, C, 1, 1].
    """
    B, C, H, W = feat.size()
    # reshape to [B, C, H*W]
    feat_var = feat.view(B, C, -1)
    mean = feat_var.mean(dim=2).view(B, C, 1, 1)
    std  = feat_var.std(dim=2).view(B, C, 1, 1) + eps
    return mean, std

def adaptive_instance_normalization(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    AdaIN: adjust content features to have the same channel-wise
    mean and std as style features.
    """
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat,   eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean
