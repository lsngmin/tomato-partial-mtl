"""④ Joint Single-head 모델.

두 태스크를 하나의 4-class 분류로 통합:
    0: maturity/immature
    1: maturity/mature
    2: quality/fresh
    3: quality/rotten

평가 시 모델 출력에서 해당 태스크 클래스만 추출하여 비교.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone, FEATURE_DIM

NUM_JOINT_CLASSES = 4
MATURITY_INDICES = (0, 1)   # 0=immature, 1=mature
QUALITY_INDICES  = (2, 3)   # 2=fresh,    3=rotten


class JointSingleHeadModel(nn.Module):
    """Backbone + 단일 4-class 헤드.

    Parameters
    ----------
    backbone_name : str
    pretrained    : bool
    dropout       : float

    Forward
    -------
    x : (B, 3, H, W)
        → logits : (B, 4)
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
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, NUM_JOINT_CLASSES),
        )
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)

    def predict_maturity(self, logits: torch.Tensor) -> torch.Tensor:
        """4-class logits 중 maturity 클래스만 추출하여 argmax."""
        return logits[:, list(MATURITY_INDICES)].argmax(dim=1)

    def predict_quality(self, logits: torch.Tensor) -> torch.Tensor:
        """4-class logits 중 quality 클래스만 추출하여 argmax."""
        return logits[:, list(QUALITY_INDICES)].argmax(dim=1)


if __name__ == "__main__":
    m = JointSingleHeadModel("resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("JointSingleHeadModel out:", y.shape)  # (2, 4)
