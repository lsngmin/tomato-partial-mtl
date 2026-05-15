"""Image transforms for train and eval.

학습 시 보수적 증강만 적용 (데이터 제공자가 이미 증강 적용했으므로 강한 추가 증강 회피).
평가 시 무작위성 없음 — Resize + Normalize 만.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_IMG_SIZE = 224


def build_train_transform(img_size: int = DEFAULT_IMG_SIZE) -> A.Compose:
    """학습용 transform — Albumentations 기반.

    Returns
    -------
    A.Compose
        ``transform(image=np.ndarray)`` 로 호출, ``out["image"]`` 가 (C,H,W) 텐서.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
            p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(),
    ])


def build_eval_transform(img_size: int = DEFAULT_IMG_SIZE) -> A.Compose:
    """평가용 transform — 결정론적 (resize + center crop + normalize)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(),
    ])


def build_tta_transforms(img_size: int = DEFAULT_IMG_SIZE) -> list[A.Compose]:
    """Test-Time Augmentation 용 transform 리스트.

    각 transform 으로 별도 추론 → 예측 평균. 평가셋 자체는 증식하지 않음.
    """
    base = [
        A.Resize(height=img_size, width=img_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.Rotate(limit=(-10, -10), p=1.0)] + base),
        A.Compose([A.Rotate(limit=(10, 10), p=1.0)] + base),
    ]
