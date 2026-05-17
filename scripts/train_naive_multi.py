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

    train_df = pd.read_csv(pkg / "manifest" / "train_split.csv")
    val_df   = pd.read_csv(pkg / "manifest" / "val_split.csv")
    test_df  = pd.read_csv(pkg / "manifest" / "test_split.csv")
    ext_mat_df  = pd.read_csv(pkg / "manifest" / "valid_maturity.csv")
    ext_qual_df = pd.read_csv(pkg / "manifest" / "valid_quality.csv")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Ext maturity: {len(ext_mat_df)}  Ext quality: {len(ext_qual_df)}")

    def ds(df, tr): return TomatoDataset(df, package_root=pkg, transform=tr, use_albumentations=True)
    train_ds = ds(train_df, build_train_transform())
    val_ds   = ds(val_df,   build_eval_transform())
    test_ds  = ds(test_df,  build_eval_transform())
    ext_mat_ds  = ds(ext_mat_df,  build_eval_transform())
    ext_qual_ds = ds(ext_qual_df, build_eval_transform())

    def loader(d, shuffle=False, drop_last=False):
        return DataLoader(d, batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True, drop_last=drop_last)
    train_loader = loader(train_ds, shuffle=True, drop_last=True)
    val_loader   = loader(val_ds)
    test_loader  = loader(test_ds)
    ext_mat_loader  = loader(ext_mat_ds)
    ext_qual_loader = loader(ext_qual_ds)

    model = NaiveMultiHeadModel(backbone_name=args.backbone, pretrained=True)
    run_name = f"naive_multi_{args.backbone}_seed{args.seed}"
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      save_dir=args.save_dir, run_name=run_name)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        step_fn=multitask_step,
        val_metric_fn=lambda preds, tgt: f1_score(tgt, preds, average="macro", zero_division=0),
        device="cuda" if torch.cuda.is_available() else "cpu", cfg=cfg,
    )
    trainer.fit()

    # ── 평가: test_split (in-distribution) + 외부 OOD ───
    model.eval()
    device = next(model.parameters()).device
    eval_results = {}

    def predict_loader(loader_, head: str):
        preds, targets = [], []
        target_key = "y_mat" if head == "maturity" else "y_qual"
        with torch.no_grad():
            for batch in loader_:
                x = batch["image"].to(device)
                out = model(x)
                logits = out.maturity_logits if head == "maturity" else out.quality_logits
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch[target_key].tolist())
        return preds, targets

    from src.data.dataset import MASK_INDEX
    # in-distribution test (test_split 안에 두 task 가 섞여있음 → MASK 인 GT 제외)
    for head in ["maturity", "quality"]:
        preds, targets = predict_loader(test_loader, head=head)
        fp, ft = zip(*[(p, t) for p, t in zip(preds, targets) if t != MASK_INDEX])
        eval_results[f"{head}_test_indistribution"] = evaluate_predictions(
            y_true=list(ft), y_pred=list(fp),
            task_name=f"{head} [test_indistribution]",
            class_names=CLASS_NAMES[head],
        )
        print("\n" + format_metrics_report(eval_results[f"{head}_test_indistribution"]))

    # external OOD
    for head, ldr in [("maturity", ext_mat_loader), ("quality", ext_qual_loader)]:
        preds, targets = predict_loader(ldr, head=head)
        eval_results[f"{head}_ext_ood"] = evaluate_predictions(
            y_true=targets, y_pred=preds,
            task_name=f"{head} [ext_ood]",
            class_names=CLASS_NAMES[head],
        )
        print("\n" + format_metrics_report(eval_results[f"{head}_ext_ood"]))

    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
