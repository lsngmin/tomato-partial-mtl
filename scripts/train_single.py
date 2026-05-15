"""② Single-task Maturity / ③ Single-task Quality 학습 스크립트.

사용법:
    python scripts/train_single.py --task maturity --seed 42
    python scripts/train_single.py --task quality  --seed 42 --backbone resnet50
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

# repo root 를 sys.path 에 추가
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import TomatoDataset, CLASS_NAMES, filter_by_task
from src.models.single_task import SingleTaskModel
from src.training.transforms import build_train_transform, build_eval_transform
from src.training.trainer import Trainer, TrainConfig
from src.evaluation.metrics import evaluate_predictions, format_metrics_report
from src.utils.seed import seed_everything
from src.utils.paths import resolve_package_root


def build_step_fn(task: str):
    """싱글태스크용 step 함수."""
    target_key = "y_mat" if task == "maturity" else "y_qual"

    def step(batch, model, device, training: bool = True):
        x = batch["image"].to(device, non_blocking=True)
        y = batch[target_key].to(device, non_blocking=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        out = {"loss": loss}
        if not training:
            out["preds"] = logits.argmax(dim=1)
            out["targets"] = y
        return out

    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["maturity", "quality"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "efficientnet_b3", "swin_t"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--package", default=None,
                   help="package root (없으면 $TOMATO_PACKAGE 또는 ./package)")
    p.add_argument("--save_dir", default="results/checkpoints")
    args = p.parse_args()

    seed_everything(args.seed)
    pkg = Path(args.package) if args.package else resolve_package_root()
    print(f"Package root: {pkg}")

    # ── 데이터 로드 ───────────────────────────────────────
    train_df = pd.read_csv(pkg / "manifest" / "train.csv")
    train_df = filter_by_task(train_df, args.task)
    print(f"Train ({args.task}): {len(train_df)}")

    val_csv = f"valid_{args.task}.csv"
    val_df = pd.read_csv(pkg / "manifest" / val_csv)
    print(f"Valid ({args.task}): {len(val_df)}")

    train_ds = TomatoDataset(
        train_df, package_root=pkg,
        transform=build_train_transform(), use_albumentations=True,
    )
    val_ds = TomatoDataset(
        val_df, package_root=pkg,
        transform=build_eval_transform(), use_albumentations=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ── 모델 ──────────────────────────────────────────────
    model = SingleTaskModel(backbone_name=args.backbone, num_classes=2, pretrained=True)

    # ── 학습 ──────────────────────────────────────────────
    run_name = f"single_{args.task}_{args.backbone}_seed{args.seed}"
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, save_dir=args.save_dir, run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=build_step_fn(args.task),
        val_metric_fn=lambda preds, targets: f1_score(targets, preds, average="macro", zero_division=0),
        device="cuda" if torch.cuda.is_available() else "cpu",
        cfg=cfg,
    )
    history = trainer.fit()

    # ── 외부 평가 ────────────────────────────────────────
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["y_mat" if args.task == "maturity" else "y_qual"]
            preds = model(x).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.tolist())

    eval_res = evaluate_predictions(
        y_true=all_targets, y_pred=all_preds,
        task_name=f"{args.task} (external)",
        class_names=CLASS_NAMES[args.task],
    )
    print()
    print(format_metrics_report(eval_res))

    # 결과 저장
    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(eval_res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
