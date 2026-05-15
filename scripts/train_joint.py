"""④ Joint Single-head (4-class) 학습 스크립트.

학습: 4-class CE 손실 (모든 이미지가 정확히 하나의 클래스 보유)
평가: Maturity / Quality 각 외부 평가셋에서, 해당 태스크 슬라이스만 비교
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import TomatoDataset, CLASS_NAMES, joint_label_4class
from src.models.joint_single import JointSingleHeadModel, MATURITY_INDICES, QUALITY_INDICES
from src.training.transforms import build_train_transform, build_eval_transform
from src.training.trainer import Trainer, TrainConfig
from src.evaluation.metrics import evaluate_predictions, format_metrics_report
from src.utils.seed import seed_everything
from src.utils.paths import resolve_package_root


class JointDataset(Dataset):
    """4-class 통합 라벨용 데이터셋."""

    def __init__(self, df, package_root, transform, use_albumentations: bool):
        self.base = TomatoDataset(df, package_root=package_root,
                                  transform=transform, use_albumentations=use_albumentations)
        self.df = df.reset_index(drop=True)
        self.joint_labels = self.df.apply(joint_label_4class, axis=1).values

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["y_joint"] = torch.tensor(int(self.joint_labels[idx]), dtype=torch.long)
        return item


def joint_step(batch, model, device, training: bool = True):
    x = batch["image"].to(device, non_blocking=True)
    y = batch["y_joint"].to(device, non_blocking=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    out = {"loss": loss}
    if not training:
        out["preds"] = logits.argmax(dim=1)
        out["targets"] = y
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "efficientnet_b3", "swin_t"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--package", default=None)
    p.add_argument("--save_dir", default="results/checkpoints")
    args = p.parse_args()

    seed_everything(args.seed)
    pkg = Path(args.package) if args.package else resolve_package_root()

    # train.csv: 모든 4 클래스 데이터 사용
    train_df = pd.read_csv(pkg / "manifest" / "train.csv")
    val_mat_df = pd.read_csv(pkg / "manifest" / "valid_maturity.csv")
    val_qual_df = pd.read_csv(pkg / "manifest" / "valid_quality.csv")

    train_ds = JointDataset(train_df, pkg, build_train_transform(), use_albumentations=True)
    val_mat_ds = JointDataset(val_mat_df, pkg, build_eval_transform(), use_albumentations=True)
    val_qual_ds = JointDataset(val_qual_df, pkg, build_eval_transform(), use_albumentations=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_mat_loader = DataLoader(val_mat_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    val_qual_loader = DataLoader(val_qual_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    # 학습 중 val_metric 은 양쪽 평가셋의 합 (간소화 — fit 후 별도 평가)
    # 여기선 maturity val 로만 비교 (간단히)
    model = JointSingleHeadModel(backbone_name=args.backbone, pretrained=True)
    run_name = f"joint_{args.backbone}_seed{args.seed}"
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      save_dir=args.save_dir, run_name=run_name)

    def val_metric_fn(preds, targets):
        return float(f1_score(targets, preds, average="macro", zero_division=0))

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_mat_loader,
        step_fn=joint_step, val_metric_fn=val_metric_fn,
        device="cuda" if torch.cuda.is_available() else "cpu", cfg=cfg,
    )
    trainer.fit()

    # ── 외부 평가 (Maturity / Quality 각 슬라이스) ──────
    model.eval()
    device = next(model.parameters()).device

    def collect(loader, task_indices):
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                logits = model(x)
                # 해당 task slice 만 argmax
                slice_logits = logits[:, list(task_indices)]
                pred = slice_logits.argmax(dim=1).cpu()
                preds.extend(pred.tolist())
                # batch 의 GT (Maturity: y_mat, Quality: y_qual)
                key = "y_mat" if task_indices == MATURITY_INDICES else "y_qual"
                targets.extend(batch[key].cpu().tolist())
        return preds, targets

    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_results = {}

    for task, loader, indices in [
        ("maturity", val_mat_loader, MATURITY_INDICES),
        ("quality",  val_qual_loader, QUALITY_INDICES),
    ]:
        preds, targets = collect(loader, indices)
        eval_res = evaluate_predictions(
            y_true=targets, y_pred=preds,
            task_name=f"{task} (joint single-head, external)",
            class_names=CLASS_NAMES[task],
        )
        print("\n" + format_metrics_report(eval_res))
        eval_results[task] = eval_res

    (out_dir / "eval.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
