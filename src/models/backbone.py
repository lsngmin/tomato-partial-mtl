"""Backbone wrappers using torchvision pretrained weights.

설계 원칙
---------
1. torchvision 의 사전학습 가중치를 사용 (timm 같은 라이브러리 wrapper 회피).
2. 백본은 분류 헤드 제거하고 (B, D) feature 벡터 반환.
3. 백본별 feature dimension 을 ``FEATURE_DIM`` 에 명시.
4. ``forward`` 출력 형태 일관: ``(B, D)``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B0_Weights,
    Swin_T_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    ShuffleNet_V2_X1_0_Weights,
)


# 백본별 feature 차원 (분류 헤드 제거 후 출력)
FEATURE_DIM: dict[str, int] = {
    # 무거운 reference
    "resnet50":              2048,
    "efficientnet_b3":       1536,
    "swin_t":                 768,
    # 경량 (스마트팜 / edge 배포 후보)
    "mobilenet_v2":          1280,   # 3.4M params
    "mobilenet_v3_small":     576,   # 2.5M params
    "mobilenet_v3_large":     960,   # 5.4M params
    "efficientnet_b0":       1280,   # 5.3M params
    "shufflenet_v2_x1_0":    1024,   # 2.3M params
}


# ─────────────────────────────────────────────────────────────
# 각 백본을 (B, D) feature 추출기로 변환
# ─────────────────────────────────────────────────────────────

def _build_resnet50(pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    # 마지막 FC 제거 → (B, 2048) feature
    m.fc = nn.Identity()
    return m


def _build_efficientnet_b3(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b3(weights=weights)
    # classifier 의 dropout + linear 제거 → (B, 1536) feature
    m.classifier = nn.Identity()
    return m


def _build_swin_t(pretrained: bool = True) -> nn.Module:
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.swin_t(weights=weights)
    # head 제거 → (B, 768) feature
    m.head = nn.Identity()
    return m


# ─────────────────────────────────────────────────────────────
# 경량 백본 (스마트팜 / edge 배포)
# ─────────────────────────────────────────────────────────────

def _build_mobilenet_v2(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier = nn.Identity()  # → (B, 1280)
    return m


def _build_mobilenet_v3_small(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier = nn.Identity()  # → (B, 576)
    return m


def _build_mobilenet_v3_large(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    m.classifier = nn.Identity()  # → (B, 960)
    return m


def _build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier = nn.Identity()  # → (B, 1280)
    return m


def _build_shufflenet_v2_x1_0(pretrained: bool = True) -> nn.Module:
    weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.shufflenet_v2_x1_0(weights=weights)
    m.fc = nn.Identity()  # → (B, 1024)
    return m


BACKBONE_REGISTRY: dict[str, Callable[[bool], nn.Module]] = {
    # 무거운 reference
    "resnet50":              _build_resnet50,
    "efficientnet_b3":       _build_efficientnet_b3,
    "swin_t":                _build_swin_t,
    # 경량
    "mobilenet_v2":          _build_mobilenet_v2,
    "mobilenet_v3_small":    _build_mobilenet_v3_small,
    "mobilenet_v3_large":    _build_mobilenet_v3_large,
    "efficientnet_b0":       _build_efficientnet_b0,
    "shufflenet_v2_x1_0":    _build_shufflenet_v2_x1_0,
}


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """이름으로 백본 인스턴스 생성.

    Parameters
    ----------
    name : str
        ``"resnet50" / "efficientnet_b3" / "swin_t"`` 중 하나.
    pretrained : bool
        ImageNet 사전학습 가중치 로드 여부.

    Returns
    -------
    nn.Module
        ``forward(x) -> (B, D)`` 형태로 feature 만 반환하는 모듈.
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone {name!r}; choose from {list(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name](pretrained)


# ─────────────────────────────────────────────────────────────
# Sanity test (개별 실행 시)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    print(f"{'backbone':22s} {'feature':>12s}  {'params (M)':>12s}")
    for name in BACKBONE_REGISTRY:
        m = build_backbone(name, pretrained=False)
        m.eval()
        with torch.no_grad():
            y = m(x)
        expected = FEATURE_DIM[name]
        assert y.shape == (2, expected), (name, y.shape, expected)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {name:20s} {str(tuple(y.shape)):>12s}  {n_params:>10.2f}M")
