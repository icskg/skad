import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel
from torch_geometric.graphgym.register import loss_dict

def get_best_device(device_list: List[str])-> str:

    if not device_list or not torch.cuda.is_available():
        return 'cpu'

    available_devices = torch.cuda.device_count()
    if len(set(device_list)) > available_devices:
        print(f"[WARNING] number of available devices is {available_devices} less than the number of settings: {len(device_list)}.")
        device_list = [f"cuda:{idx}" for idx in range(available_devices)]
        print(f"[WARNING] Using {device_list}")

    if len(device_list) == 1:
        return device_list[0]
    free_memory = []
    for device in device_list:
        if 'cuda' in device:
            # torch.cuda.set_device(device)
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            print(f"Device: {device}, Free memory: {free_mem / 1024 ** 3:.2f} GB")
            free_memory.append(free_mem)
        else:
            free_memory.append(0)
    best_idx = free_memory.index(max(free_memory))

    return device_list[best_idx]

def remove_constant_columns(df, drop_columns:list=None, tolerance=1e-8):
    """
    删除方差过小（常量或近似常量）的列
    """
    if drop_columns is None:
        stds = df.std()
        reserved_cols = list(stds[stds > tolerance].index)
        drop_columns = list(set(df.columns) - set(reserved_cols))
        if drop_columns:
            print(f"[INFO] Removed constant or near-constant columns: {drop_columns}")
    else:
        reserved_cols = [c for c in df.columns if c not in drop_columns]

    return df[reserved_cols], reserved_cols, drop_columns


def pairwise_cosine(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute cosine similarity across the last dimension of x.
    Args:
        x: (..., M, d)
        eps: float
    Returns:
        (..., M, M) cosine similarities in [-1, 1]
    """
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    return torch.matmul(x_norm, x_norm.transpose(-1, -2))

def huber_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    diff = input - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff <= delta,
                      0.5 * diff * diff,
                      delta * (abs_diff - 0.5 * delta))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 1.0,
        reduction: str = "mean"
) -> torch.Tensor:
    """Focal loss for multi-class classification."""

    log_probs = torch.log_softmax(logits, dim=-1)
    logpt = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    pt = logpt.exp()
    focal_term = (1 - pt) ** gamma
    loss = -alpha * focal_term * logpt

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def segment_stride_no_overlap(x: torch.Tensor, seg_len: int):
    """Split the time dimension into non-overlapping segments.
    Args:
        x: [batch, win_len, n_vars]
        seg_len: L
    Returns: x_seg [batch, n_seg, L, n_vars], n_seg
    """
    batch, win_len, n_vars = x.shape
    assert win_len % seg_len == 0, "win_len must be divisible by seg_len for non-overlapping segmentation."
    n_seg = win_len // seg_len
    return x.view(batch, n_seg, seg_len, n_vars), n_seg

def save_checkpoint(path: str, model: torch.nn.Module, cfg: object, epoch: int, monitor_value: float) -> None:
    """Save best checkpoint: model weights + a bit of metadata."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        cfg_dict = dict(cfg.__dict__) if hasattr(cfg, "__dict__") else cfg
    except Exception:
        cfg_dict = None
    torch.save({
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "monitor_value": float(monitor_value),
        "config": cfg_dict,
    }, path)

def load_model_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device, strict: bool = True) -> None:
    """Load weights into an existing model. Accepts common checkpoint formats."""
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            state = obj["model_state"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        else:
            state = obj  # assume it's already a state_dict
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")
    model.load_state_dict(state, strict=strict)

def set_env_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def soft_clip_contrib(l_rec_w: torch.Tensor,
                      l_cls_w: torch.Tensor,
                      l_edge_w: torch.Tensor,
                      max_frac_cls: float = None,
                      max_frac_edge: float = None,
                      eps: float = 1e-8):
    """
    以 (l_rec_w + l_cls_w + l_edge_w) 的初始和为基准，
    若分类/边的占比超过上限，则将该项按比例缩放到上限对应值；其它项不动。
    """
    device = l_rec_w.device
    one = torch.tensor(1.0, device=device, dtype=l_rec_w.dtype)

    frac_cls = (l_cls_w.detach() / l_rec_w) # negative is possible
    frac_edge = (l_edge_w.detach() / l_rec_w) # # negative is possible

    scale_cls = one
    scale_edge = one

    if max_frac_cls is not None:
        max_frac_cls_t = torch.tensor(float(max_frac_cls), device=device, dtype=l_rec_w.dtype)
        # 若初始占比 > 上限，缩放比例 = 上限 / 初始占比；否则比例=1
        scale_cls = torch.where(abs(frac_cls) > max_frac_cls_t,
                                (max_frac_cls_t / abs(frac_cls)).clamp(max=1.0),
                                one)

    if max_frac_edge is not None:
        max_frac_edge_t = torch.tensor(float(max_frac_edge), device=device, dtype=l_rec_w.dtype)
        scale_edge = torch.where(abs(frac_edge) > max_frac_edge_t,
                                 (max_frac_edge_t / abs(frac_edge)).clamp(max=1.0),
                                 one)

    loss_cls_scaled = l_cls_w * scale_cls.detach()
    loss_edge_scaled = l_edge_w * scale_edge.detach()

    return loss_cls_scaled, loss_edge_scaled