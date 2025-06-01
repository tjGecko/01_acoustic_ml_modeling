import torch


def geodesic_loss(pred_vec, target_vec, eps=1e-7):
    pred_unit   = pred_vec / (pred_vec.norm(dim=1, keepdim=True) + eps)
    target_unit = target_vec / (target_vec.norm(dim=1, keepdim=True) + eps)
    cos_sim = (pred_unit * target_unit).sum(dim=1)           # ∈ [-1,1]
    # 0 when vectors align, π when opposite
    return torch.acos(torch.clamp(cos_sim,-1+eps,1-eps)).mean()
