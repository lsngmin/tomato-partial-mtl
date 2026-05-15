"""① CV Baseline — HOG + SVM (per-task).

GPU 불필요, CPU 만으로 빠르게 실행.

사용법:
    python scripts/train_cv_baseline.py --task maturity
    python scripts/train_cv_baseline.py --task quality
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CLASS_NAMES
from src.evaluation.metrics import evaluate_predictions, format_metrics_report
from src.utils.seed import seed_everything
from src.utils.paths import resolve_package_root

IMG_SIZE = 128  # HOG 용으로 더 작은 해상도

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)


def extract_hog_color_features(img_bgr) -> np.ndarray:
    """HOG (grayscale) + RGB 히스토그램 색 특징 결합."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h = hog(gray, **HOG_PARAMS)
    # 채널별 8-bin 히스토그램
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist = np.concatenate([
        np.histogram(rgb[:, :, 0], bins=8, range=(0, 256))[0],
        np.histogram(rgb[:, :, 1], bins=8, range=(0, 256))[0],
        np.histogram(rgb[:, :, 2], bins=8, range=(0, 256))[0],
    ]).astype(np.float32)
    hist /= max(hist.sum(), 1)
    return np.concatenate([h, hist]).astype(np.float32)


def load_dataset(df: pd.DataFrame, pkg: Path) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract HOG+color"):
        path = pkg / row["filepath"]
        img_arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(extract_hog_color_features(img))
        # 라벨: 해당 task 의 class index
        cls = row["class_label"]
        task = row["task"]
        y.append(CLASS_NAMES[task].index(cls))
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["maturity", "quality"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--package", default=None)
    p.add_argument("--save_dir", default="results/checkpoints")
    args = p.parse_args()

    seed_everything(args.seed)
    pkg = Path(args.package) if args.package else resolve_package_root()

    # 학습 데이터 (해당 task 만)
    train_df = pd.read_csv(pkg / "manifest" / "train.csv")
    train_df = train_df[train_df["task"] == args.task].reset_index(drop=True)
    val_df = pd.read_csv(pkg / "manifest" / f"valid_{args.task}.csv")

    print(f"Train: {len(train_df)}  Valid: {len(val_df)}")

    t0 = time.time()
    X_train, y_train = load_dataset(train_df, pkg)
    X_val,   y_val   = load_dataset(val_df, pkg)
    print(f"Feature extraction: {time.time()-t0:.1f}s,  shape={X_train.shape}")

    # 표준화 + LinearSVC
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LinearSVC(C=1.0, max_iter=5000, random_state=args.seed)
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    print(f"SVC training: {time.time()-t0:.1f}s")
    preds = clf.predict(X_val_s)

    eval_res = evaluate_predictions(
        y_true=y_val, y_pred=preds,
        task_name=f"{args.task} (HOG+ColorHist+SVM)",
        class_names=CLASS_NAMES[args.task],
    )
    print()
    print(format_metrics_report(eval_res))

    run_name = f"cv_baseline_{args.task}_seed{args.seed}"
    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(eval_res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
