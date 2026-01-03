from __future__ import annotations
import copy
import os
from typing import Optional

import torch

from comm.utils import save_checkpoint
from configs import KMADConfig


class EarlyStopping:
    """
    Generic early-stopping utility for PyTorch training loops.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    mode : str
        'min' (lower is better) or 'max' (higher is better).
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
        Interpreted as absolute change unless `percentage=True`.
    percentage : bool
        If True, `min_delta` is interpreted as a relative percentage (e.g., 0.01 = 1%).
    restore_best_weights : bool
        If True, restores model weights from the epoch with the best value of the monitored quantity.
    verbose : bool
        If True, prints messages when improvements happen and when early stopping triggers.
    """
    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-3,
        percentage: bool = False,
        restore_best_weights: bool = True,
        verbose: bool = True,
        save_model: bool = True,
        cfg: KMADConfig = None,
        model_name: str = "best_model.pt",
    ) -> None:
        assert mode in ("min", "max")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.percentage = bool(percentage)
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = bool(verbose)

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs: int = 0
        self._best_state: Optional[dict] = None
        self._stopped: bool = False
        self.save_model = save_model
        self.model_name = model_name
        self.save_path = os.path.join(os.path.dirname(__file__), "../checkpoints/", self.model_name)
        self.cfg = cfg

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            if self.percentage:
                # smaller by at least min_delta percent
                return current < best * (1.0 - self.min_delta)
            else:
                return current < best - self.min_delta
        else:  # mode == 'max'
            if self.percentage:
                return current > best * (1.0 + self.min_delta)
            else:
                return current > best + self.min_delta

    def step(self, scores: dict, model: Optional[torch.nn.Module] = None, epoch:int=None) -> bool:
        """
        Update with the latest metric value.

        Parameters
        ----------
        scores : dict
            Current value of the monitored metrics (e.g., validation loss).
        model : torch.nn.Module, optional
            Model whose best weights should be remembered / restored.
        epoch: int
            current epoch

        Returns
        -------
        stop : bool
            True if training should stop now.
        """
        if self.cfg.monitor in scores:
            current = scores[self.cfg.monitor]
        else:
            raise ValueError(f"[EarlyStopping] Monitor metric '{self.cfg.monitor}' not found in scores.")

        if self.best_score is None:
            self.best_score = float(current)
            self.best_epoch = epoch
            if self.restore_best_weights and model is not None:
                self._best_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"[EarlyStopping] Init best {self.cfg.monitor}={self.best_score:.6f} @ epoch {epoch}")
            return False

        if self._is_improvement(float(current), float(self.best_score)):
            self.best_score = float(current)
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            if self.restore_best_weights and model is not None:
                self._best_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"[EarlyStopping] ↑ Improvement : best={self.best_score:.6f} @ epoch {epoch}")
            save_checkpoint(self.save_path, model, self.cfg, epoch, self.best_score)
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"[EarlyStopping] → No improvement  ({self.num_bad_epochs}/{self.patience})")

            if self.num_bad_epochs >= self.patience:
                self._stopped = True
                if self.restore_best_weights and model is not None and self._best_state is not None:
                    model.load_state_dict(self._best_state)
                    if self.verbose:
                        print(f"[EarlyStopping] Restored best weights from epoch {self.best_epoch}")
                if self.verbose:
                    print("[EarlyStopping] Stop training.")
                return True

        return False

    def restore(self, model: torch.nn.Module) -> None:
        """Manually restore best weights (if `restore_best_weights=False`, call this explicitly)."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def best_metric(self) -> Optional[float]:
        return self.best_score

    def reset(self) -> None:
        self.best_score = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self._best_state = None
        self._stopped = False
