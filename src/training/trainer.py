"""공통 학습 루프 (singletask / joint / naive multi 공유).

설계 원칙
---------
- 한 epoch = train + (선택) val 한 사이클
- AMP (mixed precision) 지원
- Best checkpoint 저장 (val 기준 metric)
- TensorBoard / wandb 선택적 연동 가능 (이번 버전은 표준 dict log)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"            # 'adam' / 'adamw'
    scheduler: str = "cosine"          # 'cosine' / 'none'
    amp: bool = True
    grad_clip: Optional[float] = 1.0
    log_every: int = 20
    save_dir: str = "checkpoints"
    run_name: str = "run"
    early_stopping_patience: Optional[int] = None  # None = 사용 안 함


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: Optional[float] = None
    val_metric: Optional[float] = None   # 비교용 (예: macro-F1)
    extras: dict = field(default_factory=dict)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer: Optimizer, cfg: TrainConfig, n_train_steps: int):
    if cfg.scheduler.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_train_steps)
    return None


class Trainer:
    """미니멀 학습 루프.

    사용자 정의 ``step_fn(batch, model) -> dict`` 가 핵심:
        ``dict`` 는 최소한 ``"loss"`` 키 포함. 로깅용 다른 키 가능.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        step_fn: Callable,
        val_metric_fn: Optional[Callable] = None,  # (preds, targets) -> float
        device: str = "cuda",
        cfg: Optional[TrainConfig] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.step_fn = step_fn
        self.val_metric_fn = val_metric_fn
        self.device = device
        self.cfg = cfg or TrainConfig()

        self.optimizer = build_optimizer(self.model, self.cfg)
        self.n_train_steps = self.cfg.epochs * len(train_loader)
        self.scheduler = build_scheduler(self.optimizer, self.cfg, self.n_train_steps)
        self.scaler = GradScaler(device="cuda") if self.cfg.amp and "cuda" in device else None

        self.save_dir = Path(self.cfg.save_dir) / self.cfg.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric: float = -float("inf")
        self.best_epoch: int = -1
        self.history: list[EpochMetrics] = []

    # ─── train ──────────────────────────────────────────────
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running, count = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch:02d} [train]", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                with autocast(device_type="cuda"):
                    out = self.step_fn(batch, self.model, self.device)
                    loss = out["loss"]
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.step_fn(batch, self.model, self.device)
                loss = out["loss"]
                loss.backward()
                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            bs = batch["image"].size(0)
            running += loss.item() * bs
            count += bs
            pbar.set_postfix(loss=f"{running/count:.4f}")

        return running / max(count, 1)

    # ─── eval ───────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        running, count = 0.0, 0
        all_preds, all_targets = [], []
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            out = self.step_fn(batch, self.model, self.device, training=False)
            loss = out.get("loss", torch.tensor(0.0))
            bs = batch["image"].size(0)
            running += loss.item() * bs
            count += bs
            if "preds" in out and "targets" in out:
                all_preds.extend(out["preds"].cpu().tolist())
                all_targets.extend(out["targets"].cpu().tolist())

        result = {"val_loss": running / max(count, 1)}
        if self.val_metric_fn is not None and all_preds:
            result["val_metric"] = float(self.val_metric_fn(all_preds, all_targets))
        return result

    # ─── fit ────────────────────────────────────────────────
    def fit(self) -> list[EpochMetrics]:
        patience_left = self.cfg.early_stopping_patience
        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss = self.train_one_epoch(epoch)
            val = self.evaluate()
            em = EpochMetrics(
                train_loss=tr_loss,
                val_loss=val.get("val_loss"),
                val_metric=val.get("val_metric"),
            )
            self.history.append(em)

            metric = em.val_metric if em.val_metric is not None else -em.train_loss
            improved = metric > self.best_metric
            if improved:
                self.best_metric = metric
                self.best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "metric": metric,
                    "config": self.cfg.__dict__,
                }, self.save_dir / "best.pt")
                if self.cfg.early_stopping_patience:
                    patience_left = self.cfg.early_stopping_patience

            msg = f"[epoch {epoch:02d}] train_loss={tr_loss:.4f}"
            if em.val_loss is not None:
                msg += f"  val_loss={em.val_loss:.4f}"
            if em.val_metric is not None:
                msg += f"  val_metric={em.val_metric:.4f}"
            if improved:
                msg += "  ⭐(best)"
            print(msg)

            if self.cfg.early_stopping_patience and not improved:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  Early stopping at epoch {epoch} (best epoch={self.best_epoch})")
                    break

        # 마지막에 best 가중치 로드
        ckpt = torch.load(self.save_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best (epoch {self.best_epoch}, metric={self.best_metric:.4f})")
        return self.history
