"""② Single-task Maturity / ③ Single-task Quality.

각 태스크별 독립 모델. 백본 + 1 분류 헤드 (2-class).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone, FEATURE_DIM


class SingleTaskModel(nn.Module):
    """Backbone + 1-head 분류기 (이진 분류).

    Parameters
    ----------
    backbone_name : str
    num_classes   : int (기본 2)
    pretrained    : bool
    dropout       : float
        헤드 앞 dropout 확률.

    Forward
    -------
    x : (B, 3, H, W)
        → logits : (B, num_classes)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        feat_dim = FEATURE_DIM[backbone_name]
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.backbone_name = backbone_name
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


if __name__ == "__main__":
    m = SingleTaskModel("resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("SingleTaskModel out:", y.shape)  # (2, 2)
