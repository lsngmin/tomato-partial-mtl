"""⑤ Naive Multi-Head 모델.

공유 백본 + 2개의 task-specific head.
한 이미지는 한 태스크 라벨만 가지므로 masked CE 손실 사용 (학습 시).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, FEATURE_DIM
from ..data.dataset import MASK_INDEX


@dataclass
class MultiHeadOutput:
    maturity_logits: torch.Tensor   # (B, 2)
    quality_logits:  torch.Tensor   # (B, 2)
    feature:         torch.Tensor   # (B, D) — 분석/ablation 용


class NaiveMultiHeadModel(nn.Module):
    """공유 백본 + 두 개의 독립 분류 헤드.

    Parameters
    ----------
    backbone_name : str
    pretrained    : bool
    dropout       : float

    Forward
    -------
    x : (B, 3, H, W)
        → MultiHeadOutput
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        feat_dim = FEATURE_DIM[backbone_name]
        self.head_maturity = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 2),
        )
        self.head_quality = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 2),
        )
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> MultiHeadOutput:
        feat = self.backbone(x)
        return MultiHeadOutput(
            maturity_logits=self.head_maturity(feat),
            quality_logits=self.head_quality(feat),
            feature=feat,
        )


def masked_multitask_loss(
    out: MultiHeadOutput,
    y_mat: torch.Tensor,
    y_qual: torch.Tensor,
    weight_mat: float = 1.0,
    weight_qual: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Disjoint label 환경에서 사용하는 masked CE loss.

    한 샘플은 두 태스크 중 한 라벨만 가지며, 없는 태스크 라벨은
    MASK_INDEX(-1) 로 채워져 있다. ``ignore_index=MASK_INDEX`` 의 CE 로
    해당 태스크 손실을 자연스럽게 제외.

    Parameters
    ----------
    out : MultiHeadOutput
    y_mat, y_qual : (B,) long tensor
        해당 태스크 라벨 없을 때 MASK_INDEX (-1).
    weight_mat, weight_qual : float
        태스크별 손실 가중치.

    Returns
    -------
    dict
        ``loss``, ``loss_mat``, ``loss_qual`` 모두 스칼라 텐서.
        해당 배치에 한 태스크 샘플만 있으면 다른 태스크 손실은 0 (혹은 NaN 방지).
    """
    # 마스크 적용된 CE — 모든 라벨이 MASK 면 CE 가 NaN 이 될 수 있으므로 분기
    has_mat  = (y_mat  != MASK_INDEX).any()
    has_qual = (y_qual != MASK_INDEX).any()

    loss_mat = (
        F.cross_entropy(out.maturity_logits, y_mat, ignore_index=MASK_INDEX)
        if has_mat else out.maturity_logits.sum() * 0.0
    )
    loss_qual = (
        F.cross_entropy(out.quality_logits, y_qual, ignore_index=MASK_INDEX)
        if has_qual else out.quality_logits.sum() * 0.0
    )

    total = weight_mat * loss_mat + weight_qual * loss_qual
    return {"loss": total, "loss_mat": loss_mat, "loss_qual": loss_qual}


if __name__ == "__main__":
    m = NaiveMultiHeadModel("resnet50", pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    out = m(x)
    print("Maturity logits:", out.maturity_logits.shape)
    print("Quality  logits:", out.quality_logits.shape)
    print("Feature dim:    ", out.feature.shape)

    # Masked loss 테스트
    y_mat  = torch.tensor([0, 1, MASK_INDEX, MASK_INDEX])
    y_qual = torch.tensor([MASK_INDEX, MASK_INDEX, 0, 1])
    losses = masked_multitask_loss(out, y_mat, y_qual)
    print("Loss:", {k: v.item() for k, v in losses.items()})
