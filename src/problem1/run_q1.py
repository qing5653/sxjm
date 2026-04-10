from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gm11 import GM11


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "年度数据_补充版.csv"
OUT_DIR = ROOT / "results" / "problem1"

# 题目背景给出的 2025 年对外依存度
DEPENDENCY_2025 = 72.7


def _extract_year_cols(columns: Iterable[str]) -> list[int]:
    years = []
    for c in columns:
        m = pd.Series([str(c)]).str.extract(r"(\d{4})").iloc[0, 0]
        if pd.notna(m):
            years.append(int(m))
    return years


def _load_annual_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"不支持的数据格式: {path.suffix}")

    if "指标" not in df.columns:
        raise ValueError("年度数据缺少 '指标' 列")
    df["指标"] = df["指标"].astype(str).str.strip()
    return df


def _extract_series(df: pd.DataFrame, indicator: str, value_name: str) -> pd.Series:
    row_mask = df["指标"].str.fullmatch(indicator)
    if not row_mask.any():
        raise ValueError(f"未找到指标: {indicator}")

    row = df.loc[row_mask].iloc[0]
    year_cols = [c for c in df.columns if pd.notna(c) and str(c) != "指标"]
    years = _extract_year_cols(year_cols)
    if not years:
        raise ValueError("未识别到年份列")

    vals = []
    for y in years:
        col = f"{y}年"
        vals.append(pd.to_numeric(row.get(col, np.nan), errors="coerce"))

    return pd.Series(vals, index=years, dtype=float, name=value_name).dropna().sort_index()


def _calc_dependency(import_series: pd.Series, production_series: pd.Series, dep_2025: float) -> pd.Series:
    merged = pd.concat([import_series, production_series], axis=1)
    merged.columns = ["import_10k_tons", "production_10k_tons"]

    valid_prod = merged["production_10k_tons"].notna().sum()
    if valid_prod >= 3:
        dependency = merged["import_10k_tons"] / (merged["import_10k_tons"] + merged["production_10k_tons"]) * 100
    else:
        imp_2025 = float(import_series.loc[2025])
        dep = dep_2025 / 100.0
        domestic_const = imp_2025 * (1 - dep) / dep
        dependency = import_series / (import_series + domestic_const) * 100

    dependency.name = "dependency_percent"
    return dependency


def _save_plots(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    years_hist = history_df["year"].to_numpy()
    imp_hist = history_df["import_10k_tons"].to_numpy()
    years_f = forecast_df["year"].to_numpy()

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(years_hist, imp_hist, marker="o", label="History")
    plt.plot(years_f, forecast_df["forecast_10k_tons"], marker="o", linestyle="--", label="Forecast")
    plt.plot(years_f, forecast_df["forecast_opt_10k_tons"], marker="o", linestyle="--", label="Forecast Opt")
    if "forecast_blend_10k_tons" in forecast_df.columns:
        plt.plot(years_f, forecast_df["forecast_blend_10k_tons"], marker="o", linestyle="-.", label="Forecast Blend")
    plt.fill_between(
        years_f,
        forecast_df["lower_95"],
        forecast_df["upper_95"],
        alpha=0.2,
        label="95% CI",
    )
    plt.title("Crude Oil Import: History and Forecast")
    plt.xlabel("Year")
    plt.ylabel("Import (10k tons)")
    plt.grid(alpha=0.3)
    plt.legend()
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "问题1_进口量历史与预测.png", dpi=180)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(history_df["year"], history_df["growth_rate_percent"], marker="o", label="Growth Rate")
    plt.plot(history_df["year"], history_df["dependency_percent"], marker="s", label="External Dependency")
    plt.title("Growth Rate and External Dependency (2021-2025)")
    plt.xlabel("Year")
    plt.ylabel("Percent (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "问题1_增长率与依存度.png", dpi=180)
    plt.close(fig2)


def _run_backtest(import_series: pd.Series, holdout_year: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_year not in import_series.index:
        raise ValueError(f"回测年份 {holdout_year} 不在数据中")

    train = import_series.loc[import_series.index < holdout_year]
    if len(train) < 4:
        raise ValueError("回测训练数据不足，至少需要4期")

    gm_bt = GM11(train.to_numpy(dtype=float))
    bt_result = gm_bt.fit_predict(steps=1)

    y_true = float(import_series.loc[holdout_year])
    y_pred_base = float(bt_result.forecast[0])

    # 回测训练段残差修正，得到优化预测
    residuals = train.to_numpy(dtype=float) - bt_result.fitted
    y = residuals[1:]
    x = np.column_stack([np.ones(len(residuals) - 1), residuals[:-1]])
    c, phi = np.linalg.lstsq(x, y, rcond=None)[0]
    y_pred_opt = float(y_pred_base + (c + phi * residuals[-1]))

    abs_err_base = abs(y_true - y_pred_base)
    rel_err_base = abs_err_base / y_true
    abs_err_opt = abs(y_true - y_pred_opt)
    rel_err_opt = abs_err_opt / y_true

    backtest_df = pd.DataFrame(
        {
            "year": [holdout_year],
            "actual_10k_tons": [y_true],
            "predicted_base_10k_tons": [y_pred_base],
            "predicted_opt_10k_tons": [y_pred_opt],
            "abs_error_base": [abs_err_base],
            "relative_error_base": [rel_err_base],
            "abs_error_opt": [abs_err_opt],
            "relative_error_opt": [rel_err_opt],
        }
    )

    metric_df = pd.DataFrame(
        {
            "metric": ["MAE_base", "RMSE_base", "MAPE_base", "MAE_opt", "RMSE_opt", "MAPE_opt"],
            "value": [
                abs_err_base,
                abs_err_base,
                rel_err_base * 100,
                abs_err_opt,
                abs_err_opt,
                rel_err_opt * 100,
            ],
            "unit": ["10k_tons", "10k_tons", "%", "10k_tons", "10k_tons", "%"],
        }
    )
    return backtest_df, metric_df


def _gm_residual_ar1_optimize(
    x_actual: np.ndarray,
    gm_fitted: np.ndarray,
    gm_forecast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """用 AR(1) 对 GM 残差做修正，返回修正后的拟合与预测。"""
    residuals = x_actual - gm_fitted
    if len(residuals) < 3:
        raise ValueError("残差长度不足，无法进行AR(1)修正")

    y = residuals[1:]
    x = np.column_stack([np.ones(len(residuals) - 1), residuals[:-1]])
    c, phi = np.linalg.lstsq(x, y, rcond=None)[0]

    # 训练段一步修正残差（使用真实上一期残差）
    res_hat_train = np.empty_like(residuals)
    res_hat_train[0] = residuals[0]
    for i in range(1, len(residuals)):
        res_hat_train[i] = c + phi * residuals[i - 1]

    # 未来段递推残差预测
    res_hat_future = np.empty_like(gm_forecast)
    prev = residuals[-1]
    for i in range(len(gm_forecast)):
        curr = c + phi * prev
        res_hat_future[i] = curr
        prev = curr

    fitted_opt = gm_fitted + res_hat_train
    forecast_opt = gm_forecast + res_hat_future

    res_opt = x_actual - fitted_opt
    sx = np.std(x_actual, ddof=1)
    se = np.std(res_opt, ddof=1)
    c_ratio = float(se / sx) if sx > 0 else float("inf")
    p_small_error = float(np.mean(np.abs(res_opt - res_opt.mean()) < 0.6745 * sx))
    mape = float(np.mean(np.abs(res_opt) / x_actual) * 100)

    metric_opt_df = pd.DataFrame(
        {
            "metric": ["c_residual", "phi_residual", "C_opt", "P_opt", "MAPE_opt"],
            "value": [float(c), float(phi), c_ratio, p_small_error, mape],
            "unit": ["-", "-", "-", "-", "%"],
        }
    )
    return fitted_opt, forecast_opt, metric_opt_df


def _predict_one_step_with_opt(train_values: np.ndarray) -> tuple[float, float]:
    """基于训练段做一步预测，返回(原始GM预测, 残差修正预测)。"""
    gm = GM11(train_values)
    result = gm.fit_predict(steps=1)
    base_pred = float(result.forecast[0])

    residuals = train_values - result.fitted
    y = residuals[1:]
    x = np.column_stack([np.ones(len(residuals) - 1), residuals[:-1]])
    c, phi = np.linalg.lstsq(x, y, rcond=None)[0]
    res_next = float(c + phi * residuals[-1])
    opt_pred = base_pred + res_next
    return base_pred, opt_pred


def _run_rolling_backtest(import_series: pd.Series, start_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    years = [int(y) for y in import_series.index if int(y) >= start_year]
    rows = []
    for y in years:
        train = import_series.loc[import_series.index < y].to_numpy(dtype=float)
        if len(train) < 4:
            continue
        true_v = float(import_series.loc[y])
        base_pred, opt_pred = _predict_one_step_with_opt(train)
        rows.append(
            {
                "year": y,
                "actual_10k_tons": true_v,
                "pred_base_10k_tons": base_pred,
                "pred_opt_10k_tons": opt_pred,
                "ape_base_percent": abs(base_pred - true_v) / true_v * 100,
                "ape_opt_percent": abs(opt_pred - true_v) / true_v * 100,
            }
        )

    if not rows:
        raise ValueError("滚动回测可用样本不足")

    detail_df = pd.DataFrame(rows)
    mape_base = float(detail_df["ape_base_percent"].mean())
    mape_opt = float(detail_df["ape_opt_percent"].mean())

    # 按滚动回测误差给出自适应融合权重，误差越小权重越大
    eps = 1e-9
    w_opt = float((1.0 / (mape_opt + eps)) / ((1.0 / (mape_opt + eps)) + (1.0 / (mape_base + eps))))

    metric_df = pd.DataFrame(
        {
            "metric": ["rolling_mape_base", "rolling_mape_opt", "blend_weight_opt"],
            "value": [mape_base, mape_opt, w_opt],
            "unit": ["%", "%", "-"],
        }
    )
    return detail_df, metric_df, w_opt


def _cleanup_legacy_outputs() -> None:
    legacy_names = [
        "问题1_GM11检验指标.csv",
        "问题1_GM11优化检验指标.csv",
        "问题1_2016_2025历史汇总.csv",
        "问题1_历史与预测汇总.csv",
        "问题1_回测结果.csv",
        "问题1_回测准确率指标.csv",
        "问题1_滚动回测明细.csv",
        "问题1_滚动回测指标.csv",
        "问题1_回测对比图.png",
    ]
    for name in legacy_names:
        p = OUT_DIR / name
        if p.exists():
            p.unlink()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    annual_df = _load_annual_data(DATA_FILE)
    import_series = _extract_series(annual_df, "原油进口量\(万吨\)", "import_10k_tons")
    production_series = _extract_series(annual_df, "原油产量\(万吨\)", "production_10k_tons")
    dependency_series = _calc_dependency(import_series, production_series, DEPENDENCY_2025)

    recent = import_series.loc[2021:2025]
    growth = recent.pct_change() * 100

    recent_prod = production_series.reindex(recent.index)
    recent_df = pd.DataFrame(
        {
            "year": recent.index,
            "import_10k_tons": recent.values,
            "production_10k_tons": recent_prod.values,
            "growth_rate_percent": growth.values,
            "dependency_percent": dependency_series.loc[2021:2025].values,
        }
    )

    gm = GM11(import_series.values)
    result = gm.fit_predict(steps=3)

    fitted_opt, forecast_opt, metric_opt_df = _gm_residual_ar1_optimize(
        x_actual=import_series.values,
        gm_fitted=result.fitted,
        gm_forecast=result.forecast,
    )

    _, rolling_metric_df, w_opt = _run_rolling_backtest(import_series, start_year=2022)

    future_years = np.arange(import_series.index.max() + 1, import_series.index.max() + 4)
    # 用训练残差标准差构建近似区间
    sigma = float(np.std(result.residuals, ddof=1))
    lower = result.forecast - 1.96 * sigma
    upper = result.forecast + 1.96 * sigma

    forecast_df = pd.DataFrame(
        {
            "year": future_years,
            "forecast_10k_tons": result.forecast,
            "forecast_opt_10k_tons": forecast_opt,
            "forecast_blend_10k_tons": w_opt * forecast_opt + (1 - w_opt) * result.forecast,
            "lower_95": lower,
            "upper_95": upper,
        }
    )

    backtest_df, backtest_metric_df = _run_backtest(import_series, holdout_year=2025)

    recent_path = OUT_DIR / "问题1_2021_2025分析.csv"
    forecast_path = OUT_DIR / "问题1_2026_2028预测.csv"
    eval_path = OUT_DIR / "问题1_模型评估.csv"

    eval_df = pd.DataFrame(
        {
            "metric": [
                "C_base",
                "P_base",
                "C_opt",
                "P_opt",
                "MAPE_opt_in_sample",
                "holdout_actual_2025",
                "holdout_pred_base_2025",
                "holdout_pred_opt_2025",
                "holdout_mape_base",
                "holdout_mape_opt",
                "rolling_mape_base",
                "rolling_mape_opt",
                "blend_weight_opt",
            ],
            "value": [
                result.c_ratio,
                result.p_small_error,
                float(metric_opt_df.loc[metric_opt_df["metric"] == "C_opt", "value"].iloc[0]),
                float(metric_opt_df.loc[metric_opt_df["metric"] == "P_opt", "value"].iloc[0]),
                float(metric_opt_df.loc[metric_opt_df["metric"] == "MAPE_opt", "value"].iloc[0]),
                float(backtest_df.loc[0, "actual_10k_tons"]),
                float(backtest_df.loc[0, "predicted_base_10k_tons"]),
                float(backtest_df.loc[0, "predicted_opt_10k_tons"]),
                float(backtest_metric_df.loc[backtest_metric_df["metric"] == "MAPE_base", "value"].iloc[0]),
                float(backtest_metric_df.loc[backtest_metric_df["metric"] == "MAPE_opt", "value"].iloc[0]),
                float(rolling_metric_df.loc[rolling_metric_df["metric"] == "rolling_mape_base", "value"].iloc[0]),
                float(rolling_metric_df.loc[rolling_metric_df["metric"] == "rolling_mape_opt", "value"].iloc[0]),
                w_opt,
            ],
            "unit": ["-", "-", "-", "-", "%", "10k_tons", "10k_tons", "10k_tons", "%", "%", "%", "%", "-"],
        }
    )

    _cleanup_legacy_outputs()
    recent_df.to_csv(recent_path, index=False, encoding="utf-8-sig")
    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")

    _save_plots(recent_df, forecast_df)

    print("问题1计算完成：")
    print(f"- {recent_path}")
    print(f"- {forecast_path}")
    print(f"- {eval_path}")
    print(f"- {OUT_DIR / '问题1_进口量历史与预测.png'}")
    print(f"- {OUT_DIR / '问题1_增长率与依存度.png'}")


if __name__ == "__main__":
    main()
