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
    from src.models.backbone import BACKBONE_REGISTRY
    p.add_argument("--backbone", default="resnet50",
                   choices=sorted(BACKBONE_REGISTRY))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--package", default=None)
    p.add_argument("--save_dir", default="results/checkpoints")
    args = p.parse_args()

    seed_everything(args.seed)
    pkg = Path(args.package) if args.package else resolve_package_root()

    # In-distribution split + supplementary external valid
    train_df = pd.read_csv(pkg / "manifest" / "train_split.csv")
    val_df   = pd.read_csv(pkg / "manifest" / "val_split.csv")
    test_df  = pd.read_csv(pkg / "manifest" / "test_split.csv")
    ext_mat_df  = pd.read_csv(pkg / "manifest" / "valid_maturity.csv")
    ext_qual_df = pd.read_csv(pkg / "manifest" / "valid_quality.csv")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Ext maturity: {len(ext_mat_df)}  Ext quality: {len(ext_qual_df)}")

    train_ds = JointDataset(train_df, pkg, build_train_transform(), use_albumentations=True)
    val_ds   = JointDataset(val_df,   pkg, build_eval_transform(),  use_albumentations=True)
    test_ds  = JointDataset(test_df,  pkg, build_eval_transform(),  use_albumentations=True)
    ext_mat_ds  = JointDataset(ext_mat_df,  pkg, build_eval_transform(), use_albumentations=True)
    ext_qual_ds = JointDataset(ext_qual_df, pkg, build_eval_transform(), use_albumentations=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    ext_mat_loader  = DataLoader(ext_mat_ds,  batch_size=args.batch, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    ext_qual_loader = DataLoader(ext_qual_ds, batch_size=args.batch, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    model = JointSingleHeadModel(backbone_name=args.backbone, pretrained=True)
    run_name = f"joint_{args.backbone}_seed{args.seed}"
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      save_dir=args.save_dir, run_name=run_name)

    def val_metric_fn(preds, targets):
        return float(f1_score(targets, preds, average="macro", zero_division=0))

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        step_fn=joint_step, val_metric_fn=val_metric_fn,
        device="cuda" if torch.cuda.is_available() else "cpu", cfg=cfg,
    )
    trainer.fit()

    # ── 평가: test_split 메인 + 외부 보조 ──────────────────
    model.eval()
    device = next(model.parameters()).device

    def collect(loader, task_indices, target_key):
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                logits = model(x)
                slice_logits = logits[:, list(task_indices)]
                preds.extend(slice_logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch[target_key].cpu().tolist())
        return preds, targets

    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_results = {}

    # In-distribution test (Maturity slice & Quality slice 모두)
    for task, indices, key in [
        ("maturity", MATURITY_INDICES, "y_mat"),
        ("quality",  QUALITY_INDICES,  "y_qual"),
    ]:
        preds, targets = collect(test_loader, indices, key)
        # test split 에는 4 클래스 모두 섞여 있으므로, 해당 task GT 가 있는 샘플만 사용
        # JointDataset 에선 모든 row 가 y_mat/y_qual 중 하나만 valid 라벨 (다른 건 MASK_INDEX -1)
        from src.data.dataset import MASK_INDEX
        filt_preds, filt_targets = [], []
        for p, t in zip(preds, targets):
            if t != MASK_INDEX:
                filt_preds.append(p); filt_targets.append(t)
        eval_results[f"{task}_test_indistribution"] = evaluate_predictions(
            y_true=filt_targets, y_pred=filt_preds,
            task_name=f"{task} [test_indistribution]",
            class_names=CLASS_NAMES[task],
        )
        print("\n" + format_metrics_report(eval_results[f"{task}_test_indistribution"]))

    # External OOD
    for task, loader, indices in [
        ("maturity", ext_mat_loader,  MATURITY_INDICES),
        ("quality",  ext_qual_loader, QUALITY_INDICES),
    ]:
        key = "y_mat" if task == "maturity" else "y_qual"
        preds, targets = collect(loader, indices, key)
        eval_results[f"{task}_ext_ood"] = evaluate_predictions(
            y_true=targets, y_pred=preds,
            task_name=f"{task} [ext_ood]",
            class_names=CLASS_NAMES[task],
        )
        print("\n" + format_metrics_report(eval_results[f"{task}_ext_ood"]))

    (out_dir / "eval.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
