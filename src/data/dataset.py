"""PyTorch Dataset for partial multi-task tomato classification.

CSV 스키마(공통):
    filepath, task, dataset_type, class_label, filename,
    is_augmented, original_basename, split, content_hash

태스크 인코딩:
    task ∈ {"maturity", "quality"}
    class_label ∈ {"immature","mature","fresh","rotten"}

레이블 인코딩 정책:
    Maturity head : 0 = immature, 1 = mature
    Quality  head : 0 = fresh,    1 = rotten
    한 이미지는 두 태스크 중 한 라벨만 가짐(Disjoint).
    "없는" 태스크 라벨은 MASK_INDEX (-1) 로 채워, 손실 계산 시 ignore_index 로 사용.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MASK_INDEX = -1  # 손실 계산 시 ignore_index 로 사용

TASK_NAMES = ("maturity", "quality")
CLASS_NAMES = {
    "maturity": ("immature", "mature"),
    "quality":  ("fresh",   "rotten"),
}


def _label_to_int(task: str, class_label: str) -> int:
    """class_label 문자열을 0/1 정수 라벨로 변환."""
    return CLASS_NAMES[task].index(class_label)


class TomatoDataset(Dataset):
    """Partial MTL Tomato Dataset.

    한 샘플당 두 라벨 인덱스 (maturity_y, quality_y) 를 반환하되,
    해당 이미지가 갖지 않는 태스크는 MASK_INDEX 로 마스킹.

    Parameters
    ----------
    df : pd.DataFrame
        통합 스키마 CSV (train.csv / valid_maturity.csv / valid_quality.csv 그대로).
    package_root : str | Path
        ``df["filepath"]`` 가 패키지 루트 기준 상대경로일 때 절대경로 보정용.
        절대경로면 ``None`` 가능.
    transform : Callable, optional
        ``PIL.Image`` 를 받아 변환된 텐서/이미지를 반환하는 함수.
        Albumentations 호출 시는 ``np.ndarray`` 가 필요하므로
        ``transform`` 가 직접 처리하도록 한다.
    use_albumentations : bool
        ``True`` 면 ``transform`` 에 ``image=np.ndarray`` 인자로 호출 후
        ``transformed["image"]`` 텐서를 추출.

    Returns (per item)
    ------------------
    dict with keys:
        image : torch.Tensor          (C,H,W) 정규화된 이미지
        y_mat : int (-1 or 0/1)
        y_qual: int (-1 or 0/1)
        task  : str ("maturity" / "quality")
        meta  : dict (filepath, content_hash, dataset_type, original_basename)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        package_root: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        use_albumentations: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.package_root = Path(package_root) if package_root is not None else None
        self.transform = transform
        self.use_albumentations = use_albumentations

        required = {"filepath", "task", "class_label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # task & class_label 정합성 검증
        bad_task = ~self.df["task"].isin(TASK_NAMES)
        if bad_task.any():
            raise ValueError(f"Unknown task values: {set(self.df.loc[bad_task,'task'])}")

        for t, classes in CLASS_NAMES.items():
            mask = self.df["task"] == t
            bad_cls = ~self.df.loc[mask, "class_label"].isin(classes)
            if bad_cls.any():
                raise ValueError(
                    f"Unknown class_label for task '{t}': "
                    f"{set(self.df.loc[mask][bad_cls]['class_label'])}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, fp: str) -> Path:
        p = Path(fp)
        if not p.is_absolute() and self.package_root is not None:
            p = self.package_root / p
        return p

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = self._resolve_path(row["filepath"])

        # 이미지 로드
        with Image.open(path) as img_pil:
            img_pil = img_pil.convert("RGB")
            if self.use_albumentations:
                img = np.array(img_pil)  # H,W,C uint8
            else:
                img = img_pil.copy()

        # transform
        if self.transform is not None:
            if self.use_albumentations:
                out = self.transform(image=img)
                image = out["image"]  # already tensor (C,H,W) if ToTensorV2 사용
            else:
                image = self.transform(img)
        else:
            image = img  # PIL 그대로

        # 라벨 인코딩 (한 태스크만 채움, 다른 태스크는 MASK_INDEX)
        task = row["task"]
        cls_idx = _label_to_int(task, row["class_label"])
        if task == "maturity":
            y_mat, y_qual = cls_idx, MASK_INDEX
        else:
            y_mat, y_qual = MASK_INDEX, cls_idx

        return {
            "image": image,
            "y_mat": torch.tensor(y_mat, dtype=torch.long),
            "y_qual": torch.tensor(y_qual, dtype=torch.long),
            "task": task,
            "meta": {
                "filepath": str(path),
                "content_hash": row.get("content_hash", None),
                "dataset_type": row.get("dataset_type", None),
                "original_basename": row.get("original_basename", None),
            },
        }


def filter_by_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """한 태스크만 필터링 (싱글태스크 ②, ③ 학습용)."""
    if task not in TASK_NAMES:
        raise ValueError(f"task must be one of {TASK_NAMES}, got {task!r}")
    return df[df["task"] == task].reset_index(drop=True)


def joint_label_4class(row: pd.Series) -> int:
    """④ Joint single-head 용 4-class 라벨 인코딩.

    0: maturity/immature
    1: maturity/mature
    2: quality/fresh
    3: quality/rotten
    """
    task, cls = row["task"], row["class_label"]
    if task == "maturity":
        return CLASS_NAMES["maturity"].index(cls)  # 0,1
    else:
        return 2 + CLASS_NAMES["quality"].index(cls)  # 2,3


JOINT_CLASS_NAMES = (
    "maturity/immature", "maturity/mature",
    "quality/fresh",     "quality/rotten",
)
