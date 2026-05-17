"""논문용 Figure 1 생성.

레이아웃:
    ┌────────────────┬────────────────┐
    │ (a) Immature   │ (b) Mature     │
    │   [Maturity]   │   [Maturity]   │
    ├────────────────┼────────────────┤
    │ (c) Fresh      │ (d) Rotten     │
    │   [Quality]    │   [Quality]    │
    └────────────────┴────────────────┘

원본(non-augmented) 이미지만 사용, deterministic seed.
출력: PNG (300 dpi) + PDF (vector).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import resolve_package_root

# 논문 figure 스타일
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# 4 클래스 라벨 정의 (논문용)
PANELS = [
    {"key": "a", "task": "Maturity", "cls": "immature", "label": "Immature"},
    {"key": "b", "task": "Maturity", "cls": "mature",   "label": "Mature"},
    {"key": "c", "task": "Quality",  "cls": "fresh",    "label": "Fresh"},
    {"key": "d", "task": "Quality",  "cls": "rotten",   "label": "Rotten"},
]


def pick_sample(df: pd.DataFrame, task: str, cls: str, seed: int) -> pd.Series:
    """원본만 사용, deterministic 선택."""
    sub = df[
        (df["task"] == task.lower())
        & (df["class_label"] == cls.lower())
        & (df["dataset_type"] == "original")
    ]
    if sub.empty:
        raise RuntimeError(f"No sample for {task}/{cls}")
    return sub.sample(n=1, random_state=seed).iloc[0]


def crop_square(img: Image.Image) -> Image.Image:
    """중앙 정사각형 크롭."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", default=None)
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512, help="resize crop size (px)")
    args = p.parse_args()

    pkg = Path(args.package) if args.package else resolve_package_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 학습 CSV 로드
    df = pd.read_csv(pkg / "manifest" / "train.csv")

    # 4 패널 샘플 선택
    samples = []
    for i, panel in enumerate(PANELS):
        row = pick_sample(df, panel["task"], panel["cls"], seed=args.seed + i)
        img_path = pkg / row["filepath"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = crop_square(im)
            im = im.resize((args.size, args.size), Image.LANCZOS)
        samples.append({"panel": panel, "image": np.array(im), "src": str(img_path)})

    # 2×2 그리드 — 논문 관행대로 figure 안에 caption 없음 (LaTeX 에서 별도)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6), constrained_layout=False)
    axes = axes.flatten()

    for ax, s in zip(axes, samples):
        ax.imshow(s["image"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # 패널 라벨: 굵게 "(a) Immature" + 작은 글씨 task 표시
        title = f"$\\bf{{({s['panel']['key']})\\ {s['panel']['label']}}}$"
        subtitle = f"{s['panel']['task']} task"
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=8,
                     fontfamily="serif")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                        wspace=0.05, hspace=0.30)

    png_path = out_dir / "figure_1_class_samples.png"
    pdf_path = out_dir / "figure_1_class_samples.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved:\n  {png_path}\n  {pdf_path}")
    print("\nSample sources (for caption):")
    for s in samples:
        print(f"  ({s['panel']['key']}) {s['src']}")


if __name__ == "__main__":
    main()
