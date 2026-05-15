"""평가 지표 및 통계적 신뢰구간.

지원 지표
---------
- Accuracy
- Balanced Accuracy
- Macro / Per-class F1
- Per-class Precision, Recall
- Confusion matrix
- Bootstrap 95% CI (특히 Maturity Immature n=93 한계 대응)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class BinaryMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    class0_precision: float; class0_recall: float; class0_f1: float
    class1_precision: float; class1_recall: float; class1_f1: float
    confusion: list[list[int]]   # 2x2

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: tuple[int, int] = (0, 1),
) -> BinaryMetrics:
    """이진 분류 지표 일괄 계산."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    return BinaryMetrics(
        n=len(y_true),
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        macro_f1=float(macro_f1),
        class0_precision=float(prec[0]), class0_recall=float(rec[0]), class0_f1=float(f1[0]),
        class1_precision=float(prec[1]), class1_recall=float(rec[1]), class1_f1=float(f1[1]),
        confusion=cm.astype(int).tolist(),
    )


# ─────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """주어진 metric 의 부트스트랩 신뢰구간 계산.

    Parameters
    ----------
    y_true, y_pred : array-like
    metric_fn : callable(y_true, y_pred) -> float
    n_boot : int (기본 1000)
    alpha : float (기본 0.05 → 95% CI)
    seed : int

    Returns
    -------
    dict
        ``{"point": float, "lo": float, "hi": float, "se": float}``
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y_true, y_pred))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[i] = float(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            boots[i] = np.nan

    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boots.std(ddof=1)),
    }


def per_class_recall_ci(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    target_class: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """특정 클래스의 recall 부트스트랩 CI — 소수 클래스(Immature 93장) 분석용."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def _recall(yt, yp):
        mask = yt == target_class
        if mask.sum() == 0:
            return np.nan
        return float((yp[mask] == target_class).mean())

    return bootstrap_ci(y_true, y_pred, _recall, n_boot=n_boot, seed=seed)


# ─────────────────────────────────────────────────────────────
# 통합 평가 + 포맷 출력
# ─────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
    class_names: tuple[str, str],
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> dict:
    """한 태스크의 모든 지표 + 부트스트랩 CI 통합."""
    m = binary_metrics(y_true, y_pred)
    out = {
        "task": task_name,
        "class_names": list(class_names),
        **m.to_dict(),
    }

    # 메인 메트릭에 대한 부트스트랩 CI
    out["ci_balanced_accuracy"] = bootstrap_ci(
        y_true, y_pred,
        balanced_accuracy_score, n_boot=bootstrap_n, seed=seed,
    )
    out["ci_macro_f1"] = bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        n_boot=bootstrap_n, seed=seed,
    )

    # 클래스별 recall CI (소수 클래스 정보 노출)
    out["ci_class0_recall"] = per_class_recall_ci(y_true, y_pred, 0, bootstrap_n, seed)
    out["ci_class1_recall"] = per_class_recall_ci(y_true, y_pred, 1, bootstrap_n, seed)
    return out


def format_metrics_report(metrics: dict) -> str:
    """가독성 좋은 콘솔 리포트 생성."""
    cn = metrics["class_names"]
    lines = [
        f"=== {metrics['task']}  (n={metrics['n']}) ===",
        f"  Accuracy           : {metrics['accuracy']:.4f}",
        f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}"
        f"  [95% CI {metrics['ci_balanced_accuracy']['lo']:.4f}, {metrics['ci_balanced_accuracy']['hi']:.4f}]",
        f"  Macro F1           : {metrics['macro_f1']:.4f}"
        f"  [95% CI {metrics['ci_macro_f1']['lo']:.4f}, {metrics['ci_macro_f1']['hi']:.4f}]",
        "",
        f"  Class {cn[0]:10s} : P={metrics['class0_precision']:.3f}  R={metrics['class0_recall']:.3f}"
        f"  F1={metrics['class0_f1']:.3f}"
        f"  [recall CI {metrics['ci_class0_recall']['lo']:.3f}, {metrics['ci_class0_recall']['hi']:.3f}]",
        f"  Class {cn[1]:10s} : P={metrics['class1_precision']:.3f}  R={metrics['class1_recall']:.3f}"
        f"  F1={metrics['class1_f1']:.3f}"
        f"  [recall CI {metrics['ci_class1_recall']['lo']:.3f}, {metrics['ci_class1_recall']['hi']:.3f}]",
        "",
        f"  Confusion (rows=true, cols=pred):",
        f"            {cn[0]:>8s}  {cn[1]:>8s}",
        f"    {cn[0]:8s}  {metrics['confusion'][0][0]:>8d}  {metrics['confusion'][0][1]:>8d}",
        f"    {cn[1]:8s}  {metrics['confusion'][1][0]:>8d}  {metrics['confusion'][1][1]:>8d}",
    ]
    return "\n".join(lines)
