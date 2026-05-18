"""Originals-only 8:1:1 split.

배경:
  - 기존 train_split (10,000장: 원본 + 증강 4배) 에서 모든 모델이 in-dist 100% saturate
  - 증강을 빼고 원본만으로 학습하면 task 가 어려워져서 method 간 차이가 보일 가능성
  - 또한 augmentation leak (같은 원본의 여러 변형) 우려 자체가 사라짐

처리:
  1) package/manifest/train.csv (10,000) 에서 dataset_type == 'original' 만 필터
     → 2000장 (4 클래스 × 500)
  2) Stratified 8:1:1 (Train:Val:Test) split per (task, class)
  3) 기존 train_split.csv 등을 덮어씀
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

REPO = Path(__file__).resolve().parents[1]
PKG_MANIFEST   = REPO / "package" / "manifest"
DATA_PROCESSED = REPO / "data" / "processed"

# ── 1) 로드 + 필터 ──────────────────────────────────────────
df = pd.read_csv(PKG_MANIFEST / "train.csv")
print(f"전체 train.csv: {len(df):,}")

orig = df[df["dataset_type"] == "original"].reset_index(drop=True)
print(f"Original만 필터: {len(orig):,}")
print()
print(orig.groupby(["task", "class_label"]).size().to_string())
assert len(orig) == 2000, f"expected 2000, got {len(orig)}"

# ── 2) Stratified 8:1:1 split ──────────────────────────────
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

def stratified_split(g: pd.DataFrame, seed: int):
    """한 (task, class) 그룹을 8:1:1 로 무작위 분할."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(g))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * TRAIN_RATIO))
    n_val   = int(round(n * VAL_RATIO))
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

splits = {"train": [], "val": [], "test": []}
for (task, cls), g in orig.groupby(["task", "class_label"], sort=True):
    g = g.reset_index(drop=True)
    tr, va, te = stratified_split(g, seed=SEED + hash((task, cls)) % 1000)
    splits["train"].append(g.iloc[tr])
    splits["val"].append(  g.iloc[va])
    splits["test"].append( g.iloc[te])

train_df = pd.concat(splits["train"], ignore_index=True)
val_df   = pd.concat(splits["val"],   ignore_index=True)
test_df  = pd.concat(splits["test"],  ignore_index=True)

print()
print("=== Split sizes ===")
total = len(orig)
for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"  {name:5s}: {len(d):>4,} rows ({len(d)/total*100:.1f}%)")
print(f"  합계 : {len(train_df)+len(val_df)+len(test_df):>4,} (원본: {total})")

print()
print("=== 분할별 클래스 분포 ===")
for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"[{name}]")
    print(d.groupby(["task", "class_label"]).size().unstack(fill_value=0))

# ── 3) Leak 검증 (filepath 기준, 진짜 같은 파일인지) ─────────
# 참고: filename 만 보면 maturity/immature와 maturity/mature 에 같은 basename이
# 존재하는 경우가 4건 있음(예: 같은 basename 다른 폴더). filepath 는 unique 하므로 이걸로 검증.
train_paths = set(train_df["filepath"])
val_paths   = set(val_df["filepath"])
test_paths  = set(test_df["filepath"])
leaks = {
    "train_val":  len(train_paths & val_paths),
    "train_test": len(train_paths & test_paths),
    "val_test":   len(val_paths & test_paths),
}
print()
print("=== Filepath-level leak check ===")
for k, v in leaks.items():
    status = "OK" if v == 0 else "LEAK"
    print(f"  {k:15s}: {v}  [{status}]")
assert all(v == 0 for v in leaks.values()), "filepath-level leak detected"

# 추가 정보: filename collision (서로 다른 폴더, 동일 basename)
all_paths = pd.concat([train_df[['filename','filepath']], val_df[['filename','filepath']], test_df[['filename','filepath']]])
fn_collisions = all_paths.groupby('filename')['filepath'].nunique()
n_fn_col = int((fn_collisions > 1).sum())
print(f"  filename collisions (다른 폴더, 같은 basename): {n_fn_col}")

# ── 4) 저장 (기존 train_split.csv 덮어씀) ───────────────────
train_df.to_csv(PKG_MANIFEST   / "train_split.csv", index=False)
val_df.to_csv(  PKG_MANIFEST   / "val_split.csv",   index=False)
test_df.to_csv( PKG_MANIFEST   / "test_split.csv",  index=False)
train_df.to_csv(DATA_PROCESSED / "train_split.csv", index=False)
val_df.to_csv(  DATA_PROCESSED / "val_split.csv",   index=False)
test_df.to_csv( DATA_PROCESSED / "test_split.csv",  index=False)

meta = {
    "strategy": "Originals-only (no augmentation) 8:1:1 stratified split",
    "seed": SEED,
    "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
    "sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
    "class_distribution": {
        split: {f"{t}/{c}": int(v)
                for (t,c), v in d.groupby(["task","class_label"]).size().to_dict().items()}
        for split, d in [("train", train_df), ("val", val_df), ("test", test_df)]
    },
    "leak_check": leaks,
    "rationale": (
        "Augmented data made all methods saturate at ~100% on in-distribution test, "
        "preventing methodology comparison. Removing augmentation yields 2000 challenging "
        "samples where method differences (negative transfer, etc.) may emerge."
    ),
}
(PKG_MANIFEST / "split_meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
(DATA_PROCESSED / "split_meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print()
print("✅ 저장 완료 (package/manifest + data/processed)")
print(f"   train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
