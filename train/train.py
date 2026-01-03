# train.py
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs import KMADConfig
from kmad.model import KMADModel
from data.data import KMADDataModule
from eval.evaluation import calculate_reconstruction_errors, calculate_anomaly_metrics
from comm.utils import set_env_seed, get_best_device


def setup_training_environment(args=None, cfg=None, fix_seed=True, devices=None):
    if devices is None:
        devices = ['cuda:0', 'cuda:1']
    cfg = KMADConfig() if cfg is None else cfg
    if args is not None:
        valid_args = {k: v for k, v in vars(args).items() if v is not None}
        for key, value in valid_args.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    cfg.device = get_best_device(devices)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if fix_seed:
        set_env_seed(cfg.seed)
    else:
        print("[Warning] Seed is not fixed. This may lead to different results across runs.")

    return cfg, device

def create_data_module(cfg):
    dm = KMADDataModule(
        dataset=cfg.dataset,
        window=cfg.win_len,
        stride=cfg.stride,
        val_ratio=cfg.val_ratio,
        batch_size=cfg.batch_size,
        scaling=cfg.scaling if hasattr(cfg, "scaling") else "minmax",
        limit_train_ratio=cfg.limit_train_ratio
    )
    train_loader, val_loader, test_loader = dm.train_val_test_loaders()
    return dm, train_loader, val_loader, test_loader

def create_model_and_optimizer(cfg, device):
    model = KMADModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=1e-6)
    return model, optimizer, scheduler

def train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    n_steps = 0
    current_lr = optimizer.param_groups[0]["lr"]
    show_progress = (epoch + 1) % 10 == 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} (lr={current_lr:.6f})"):
        x, y_types, adj_prior, _, _ = batch
        x, y_types, adj_prior = x.to(device), y_types.to(device), adj_prior.to(device)

        out = model(x, y_types, adj_prior=adj_prior, show_progress=show_progress)
        losses = model.compute_losses(x, out, y_types)

        optimizer.zero_grad()
        losses["l_total"].backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += float(losses["l_total"].detach().cpu())
        n_steps += 1
        show_progress = False

    if scheduler is not None:
        scheduler.step()

    return running_loss / max(1, n_steps)

def validate_model(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device,
                   epoch:int=None, train_loss:list=None, compute_metrics:bool=False):
    if val_loader is None:
        print("[Warning] Validation set is empty. Skipping validation.")
        return None

    total_loss: Dict[str, float] = {}
    batch_count = 0
    all_errors = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validate"):
            x, y_types, adj_prior, y_labels, _ = batch
            x, y_types, adj_prior, y_labels = x.to(device), y_types.to(device), adj_prior.to(device), y_labels.to(device)
            out = model(x, y_types, adj_prior=adj_prior, show_progress=False)
            losses = model.compute_losses(x, out, y_types)
            for k,v in losses.items():
                total_loss[k] = total_loss.get(k, 0.0) + v.item()

            if compute_metrics:
                reconstruction_errors = calculate_reconstruction_errors(x, out["x_hat"])
                all_errors.append(reconstruction_errors.cpu().numpy())
                if y_labels is not None:
                    all_labels.append(y_labels.cpu().numpy())
            batch_count += 1

    results = {k: v / max(batch_count, 1) for k, v in total_loss.items()}

    if compute_metrics and len(all_errors) > 0:
        all_errors = np.concatenate(all_errors)

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            metrics = calculate_anomaly_metrics(all_errors, all_labels)
            results.update(metrics)
    if epoch is not None and train_loss is not None:
        print(f"Epoch {epoch}: Train Loss = {train_loss if train_loss else 0:.6f},", end=" ")
    print(f"| Val Loss " + ", ".join(f"{k}={v:.6f}" for k,v in results.items()))

    return results


def handle_early_stopping(early_stopping, val_values, model, epoch):
    if early_stopping.step(val_values, model=model, epoch=epoch):
        print(
            f"Early stopping at epoch {epoch}. Best val={early_stopping.best_metric:.6f} at epoch {early_stopping.best_epoch}."
        )
        return True
    return False

def model_fit(model, train_loader, val_loader, optimizer, device, cfg, early_stopping, scheduler):

    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler)
        val_scores = validate_model(model, val_loader, device, epoch, train_loss)

        if handle_early_stopping(early_stopping, val_scores, model, epoch):
            break

    return early_stopping.best_epoch
