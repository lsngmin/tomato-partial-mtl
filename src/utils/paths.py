"""프로젝트 경로 헬퍼."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_package_root(env_var: str = "TOMATO_PACKAGE") -> Path:
    """학습 데이터 패키지 루트 경로 반환.

    우선순위:
    1) 환경변수 ``TOMATO_PACKAGE``
    2) ``./package`` (현재 작업 디렉토리)
    3) ``<repo>/package``
    """
    if env_var in os.environ:
        return Path(os.environ[env_var]).resolve()

    cwd = Path.cwd()
    for candidate in [cwd / "package", cwd.parent / "package"]:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Package root not found. Set ${env_var} or place 'package/' next to repo."
    )
