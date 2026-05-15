"""⑤ Naive Multi-Head 학습 스크립트.

공유 백본 + 2개 헤드, masked CE 손실 (한 이미지는 한 태스크 라벨만).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import TomatoDataset, CLASS_NAMES, MASK_INDEX
from src.models.naive_multi import NaiveMultiHeadModel, masked_multitask_loss
from src.training.transforms import build_train_transform, build_eval_transform
from src.training.trainer import Trainer, TrainConfig
from src.evaluation.metrics import evaluate_predictions, format_metrics_report
from src.utils.seed import seed_everything
from src.utils.paths import resolve_package_root


def multitask_step(batch, model, device, training: bool = True):
    x = batch["image"].to(device, non_blocking=True)
    y_mat  = batch["y_mat"].to(device, non_blocking=True)
    y_qual = batch["y_qual"].to(device, non_blocking=True)

    out = model(x)
    losses = masked_multitask_loss(out, y_mat, y_qual)
    result = {"loss": losses["loss"]}
    if not training:
        # val 시 어느 태스크 GT 가 있는지에 따라 prediction/target 정리
        # 여기서는 maturity GT 있는 샘플만 추출하여 val_metric 계산
        mask_mat = y_mat != MASK_INDEX
        if mask_mat.any():
            preds = out.maturity_logits[mask_mat].argmax(dim=1)
            result["preds"] = preds
            result["targets"] = y_mat[mask_mat]
    return result


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

    train_df = pd.read_csv(pkg / "manifest" / "train.csv")
    val_mat_df = pd.read_csv(pkg / "manifest" / "valid_maturity.csv")
    val_qual_df = pd.read_csv(pkg / "manifest" / "valid_quality.csv")

    train_ds = TomatoDataset(train_df, package_root=pkg,
                             transform=build_train_transform(), use_albumentations=True)
    val_mat_ds = TomatoDataset(val_mat_df, package_root=pkg,
                               transform=build_eval_transform(), use_albumentations=True)
    val_qual_ds = TomatoDataset(val_qual_df, package_root=pkg,
                                transform=build_eval_transform(), use_albumentations=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_mat_loader = DataLoader(val_mat_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    val_qual_loader = DataLoader(val_qual_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    model = NaiveMultiHeadModel(backbone_name=args.backbone, pretrained=True)
    run_name = f"naive_multi_{args.backbone}_seed{args.seed}"
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      save_dir=args.save_dir, run_name=run_name)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_mat_loader,
        step_fn=multitask_step,
        val_metric_fn=lambda preds, tgt: f1_score(tgt, preds, average="macro", zero_division=0),
        device="cuda" if torch.cuda.is_available() else "cpu", cfg=cfg,
    )
    trainer.fit()

    # ── 외부 평가 ───────────────────────────────────────
    model.eval()
    device = next(model.parameters()).device
    eval_results = {}

    def predict_loader(loader, head: str):
        preds, targets = [], []
        target_key = "y_mat" if head == "maturity" else "y_qual"
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                out = model(x)
                logits = out.maturity_logits if head == "maturity" else out.quality_logits
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch[target_key].tolist())
        return preds, targets

    for task, loader in [("maturity", val_mat_loader), ("quality", val_qual_loader)]:
        preds, targets = predict_loader(loader, head=task)
        eval_res = evaluate_predictions(
            y_true=targets, y_pred=preds,
            task_name=f"{task} (naive multi-head, external)",
            class_names=CLASS_NAMES[task],
        )
        print("\n" + format_metrics_report(eval_res))
        eval_results[task] = eval_res

    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
