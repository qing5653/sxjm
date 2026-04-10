from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GM11Result:
    a: float
    b: float
    fitted: np.ndarray
    forecast: np.ndarray
    residuals: np.ndarray
    c_ratio: float
    p_small_error: float


class GM11:
    """GM(1,1) 灰色预测模型。"""

    def __init__(self, x0: np.ndarray):
        self.x0 = np.asarray(x0, dtype=float)
        if self.x0.ndim != 1 or len(self.x0) < 4:
            raise ValueError("GM(1,1) 输入序列必须为一维且长度不少于4")

    def fit_predict(self, steps: int) -> GM11Result:
        x0 = self.x0
        x1 = np.cumsum(x0)
        z1 = 0.5 * (x1[1:] + x1[:-1])

        b_mat = np.column_stack((-z1, np.ones_like(z1)))
        y_vec = x0[1:]
        a, b = np.linalg.lstsq(b_mat, y_vec, rcond=None)[0]

        # 还原全序列：包含训练段和未来 steps 段
        k = np.arange(0, len(x0) + steps + 1)
        x1_hat = (x0[0] - b / a) * np.exp(-a * k) + b / a
        x0_hat = np.diff(x1_hat)

        fitted = x0_hat[: len(x0)]
        forecast = x0_hat[len(x0) :]
        residuals = x0 - fitted

        # 后验差比与小误差概率
        sx = np.std(x0, ddof=1)
        se = np.std(residuals, ddof=1)
        c_ratio = float(se / sx) if sx > 0 else float("inf")
        threshold = 0.6745 * sx
        p_small_error = float(np.mean(np.abs(residuals - residuals.mean()) < threshold))

        return GM11Result(
            a=float(a),
            b=float(b),
            fitted=fitted,
            forecast=forecast,
            residuals=residuals,
            c_ratio=c_ratio,
            p_small_error=p_small_error,
        )
