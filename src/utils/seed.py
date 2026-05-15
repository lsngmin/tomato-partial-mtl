"""재현성을 위한 시드 고정."""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """모든 무작위 소스 시드 고정.

    Parameters
    ----------
    seed : int
    deterministic : bool
        True 면 cuDNN deterministic 설정 (재현성 ↑, 속도 ↓).
        False 면 benchmark 모드로 속도 우선.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
