from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp


ROOT = Path(__file__).resolve().parents[2]
PROBLEM1_DIR = ROOT / "results" / "problem1"
PROBLEM3_DIR = ROOT / "results" / "problem3"

sys.path.append(str(ROOT / "src" / "problem1"))
sys.path.append(str(ROOT / "src" / "problem3"))

from gm11 import GM11  # noqa: E402
import run_q3 as q3  # noqa: E402


FIG_DPI = 300


def _set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFC",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.size": 10,
            "font.family": "DejaVu Serif",
            "grid.color": "#D8D8D8",
            "grid.alpha": 0.45,
            "grid.linestyle": "-",
        }
    )


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "GPT-5.3-Codex", "Title": out_path.stem},
    )


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def _extract_import_series() -> tuple[np.ndarray, np.ndarray]:
    data_path = ROOT / "data" / "年度数据_补充版.csv"
    df = pd.read_csv(data_path)
    if "指标" not in df.columns:
        raise ValueError("年度数据缺少'指标'列")

    row_mask = df["指标"].astype(str).str.contains("原油进口量", regex=False)
    if not row_mask.any():
        raise ValueError("未找到'原油进口量'行")
    row = df.loc[row_mask].iloc[0]

    years = []
    vals = []
    for c in df.columns:
        if c == "指标":
            continue
        m = re.search(r"(\d{4})", str(c))
        if not m:
            continue
        y = int(m.group(1))
        v = pd.to_numeric(row[c], errors="coerce")
        if pd.notna(v):
            years.append(y)
            vals.append(float(v))

    order = np.argsort(years)
    years_arr = np.array(years, dtype=int)[order]
    vals_arr = np.array(vals, dtype=float)[order]
    return years_arr, vals_arr


def generate_residual_diagnostic_plot() -> Path:
    years, x = _extract_import_series()
    gm = GM11(x)
    res = gm.fit_predict(steps=0)

    base_residual = x - res.fitted

    y = base_residual[1:]
    x_lag = np.column_stack([np.ones(len(base_residual) - 1), base_residual[:-1]])
    c, phi = np.linalg.lstsq(x_lag, y, rcond=None)[0]

    res_hat = np.empty_like(base_residual)
    res_hat[0] = base_residual[0]
    for i in range(1, len(base_residual)):
        res_hat[i] = c + phi * base_residual[i - 1]
    opt_residual = x - (res.fitted + res_hat)

    max_lag = min(6, len(opt_residual) - 1)
    acf_vals = [1.0]
    for k in range(1, max_lag + 1):
        a = opt_residual[k:]
        b = opt_residual[:-k]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            acf_vals.append(0.0)
        else:
            acf_vals.append(float(np.corrcoef(a, b)[0, 1]))

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))

    axes[0].plot(years, base_residual, marker="o", linewidth=1.8, color="#B07AA1", label="Base Residual")
    axes[0].plot(years, opt_residual, marker="s", linewidth=2.0, color="#4E79A7", label="Optimized Residual")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("(a) Residual Time Series")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Residual (10k tons)")
    axes[0].legend()
    axes[0].grid(alpha=0.35)

    axes[1].hist(opt_residual, bins=6, color="#76B7B2", edgecolor="white", alpha=0.9)
    axes[1].axvline(np.mean(opt_residual), color="#E15759", linestyle="--", linewidth=1.2)
    axes[1].set_title("(b) Residual Distribution")
    axes[1].set_xlabel("Residual (10k tons)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.25)

    lags = np.arange(0, max_lag + 1)
    stem = axes[2].stem(lags, acf_vals)
    plt.setp(stem.markerline, markersize=5, markerfacecolor="#4E79A7")
    plt.setp(stem.stemlines, color="#4E79A7", linewidth=1.6)
    plt.setp(stem.baseline, color="#333333", linewidth=0.8)
    conf = 1.96 / np.sqrt(len(opt_residual))
    axes[2].axhline(conf, color="#E15759", linestyle="--", linewidth=1.0)
    axes[2].axhline(-conf, color="#E15759", linestyle="--", linewidth=1.0)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("(c) Residual ACF")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("ACF")
    axes[2].grid(alpha=0.25)

    summary_text = (
        f"Mean={np.mean(opt_residual):.2f}\n"
        f"Std={np.std(opt_residual, ddof=1):.2f}\n"
        f"Max|res|={np.max(np.abs(opt_residual)):.2f}"
    )
    axes[1].text(0.98, 0.95, summary_text, transform=axes[1].transAxes, ha="right", va="top", fontsize=8,
                 bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#BBBBBB"})

    fig.suptitle("Problem 1 Backtest Residual Diagnostics", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = PROBLEM1_DIR / "问题1_回测残差诊断图.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def generate_tornado_plot() -> Path:
    s1_path = PROBLEM3_DIR / "问题3_敏感性分析_S1汇总.csv"
    s1 = pd.read_csv(s1_path)
    base = s1.loc[s1["case"] == "baseline"].iloc[0]

    param_cases = {
        "Emergency Procurement": ("emergency_procure_ratio_minus10", "emergency_procure_ratio_plus10"),
        "Reserve Release Cap": ("reserve_cap_ratio_minus10", "reserve_cap_ratio_plus10"),
        "Demand Cut Cap": ("demand_cut_ratio_minus10", "demand_cut_ratio_plus10"),
    }

    p_minus = []
    p_plus = []
    e_minus = []
    e_plus = []
    labels = []
    for label, (minus_case, plus_case) in param_cases.items():
        r_minus = s1.loc[s1["case"] == minus_case].iloc[0]
        r_plus = s1.loc[s1["case"] == plus_case].iloc[0]
        phys_minus = float(r_minus["physical_shortage_10k_tons"] - base["physical_shortage_10k_tons"])
        phys_plus = float(r_plus["physical_shortage_10k_tons"] - base["physical_shortage_10k_tons"])
        eff_minus = float(r_minus["effective_shortage_10k_tons"] - base["effective_shortage_10k_tons"])
        eff_plus = float(r_plus["effective_shortage_10k_tons"] - base["effective_shortage_10k_tons"])
        labels.append(label)
        p_minus.append(phys_minus)
        p_plus.append(phys_plus)
        e_minus.append(eff_minus)
        e_plus.append(eff_plus)

    rank_score = [max(abs(a), abs(b)) for a, b in zip(p_minus, p_plus)]
    order = np.argsort(rank_score)[::-1]
    labels = [labels[i] for i in order]
    p_minus = [p_minus[i] for i in order]
    p_plus = [p_plus[i] for i in order]
    e_minus = [e_minus[i] for i in order]
    e_plus = [e_plus[i] for i in order]

    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)

    axes[0].barh(y, p_minus, height=0.34, color="#F28E2B", label="-10%")
    axes[0].barh(y, p_plus, height=0.34, color="#4E79A7", alpha=0.85, label="+10%")
    axes[0].axvline(0, color="#333333", linewidth=1.0)
    axes[0].set_title("(a) Physical Shortage Delta")
    axes[0].set_xlabel("Change (10k tons)")
    axes[0].set_yticks(y, labels)
    axes[0].grid(axis="x", alpha=0.35)

    axes[1].barh(y, e_minus, height=0.34, color="#F28E2B", label="-10%")
    axes[1].barh(y, e_plus, height=0.34, color="#4E79A7", alpha=0.85, label="+10%")
    axes[1].axvline(0, color="#333333", linewidth=1.0)
    axes[1].set_title("(b) Effective Shortage Delta")
    axes[1].set_xlabel("Change (10k tons)")
    axes[1].grid(axis="x", alpha=0.35)

    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Problem 3 Tornado Sensitivity (S1)", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = PROBLEM3_DIR / "问题3_敏感性龙卷风图.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def generate_pressure_heatmap() -> Path:
    kpi_path = PROBLEM3_DIR / "问题3_关键指标评估.csv"
    kpi = pd.read_csv(kpi_path)

    metrics = [
        "physical_shortage_10k_tons",
        "effective_shortage_10k_tons",
        "demand_cut_dependency_ratio",
        "hhi_concentration",
    ]
    labels = ["Physical Shortage\n(10k tons)", "Effective Shortage\n(10k tons)", "Demand-Cut\nRatio", "HHI"]
    data = kpi.set_index("scenario")[metrics].copy()

    norm = data.copy()
    for col in metrics:
        cmin = float(data[col].min())
        cmax = float(data[col].max())
        if cmax - cmin < 1e-12:
            norm[col] = 0.0
        else:
            norm[col] = (data[col] - cmin) / (cmax - cmin)

    fig = plt.figure(figsize=(8.8, 5.0))
    im = plt.imshow(norm.to_numpy(), aspect="auto", cmap="RdYlGn_r")
    plt.xticks(range(len(labels)), labels, rotation=12)
    plt.yticks(range(len(norm.index)), norm.index.tolist())
    plt.title("Problem 3 Scenario Pressure Matrix", fontsize=12, fontweight="bold")
    cb = plt.colorbar(im, label="Normalized Pressure")
    cb.ax.tick_params(labelsize=8)

    for i, scen in enumerate(norm.index):
        for j, m in enumerate(metrics):
            raw = float(data.loc[scen, m])
            txt = f"{raw:.4f}" if "ratio" in m or "hhi" in m else f"{raw:.2f}"
            color = "white" if norm.iloc[i, j] > 0.65 else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    for i in range(norm.shape[0] + 1):
        plt.axhline(i - 0.5, color="white", linewidth=0.8)
    for j in range(norm.shape[1] + 1):
        plt.axvline(j - 0.5, color="white", linewidth=0.8)

    fig.tight_layout()
    out_path = PROBLEM3_DIR / "问题3_场景压力矩阵热图.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def _solve_weighted_s1(
    base_df: pd.DataFrame,
    q_min: float,
    scenario: q3.Scenario,
    w_cost: float,
    w_risk: float,
    w_div: float,
) -> dict[str, float] | None:
    countries = base_df["贸易伙伴名称"].tolist()
    idx_map = {c: i for i, c in enumerate(countries)}
    price = base_df.set_index("贸易伙伴名称")["price_rmb_per_kg"].to_dict()
    risk = base_df.set_index("贸易伙伴名称")["risk_score"].to_dict()

    cap = q3.build_capacity(base_df, scenario)
    route_cap, route_emergency_extra = q3.route_capacity(base_df, scenario)
    emergency_cap = q3.build_emergency_capacity(base_df, scenario)

    mean_price = float(base_df["price_rmb_per_kg"].mean())
    mean_risk = float(base_df["risk_score"].mean())
    cost_norm = {c: price[c] / mean_price for c in countries}
    risk_norm = {c: risk[c] / mean_risk for c in countries}

    model = pulp.LpProblem("Pareto_S1", pulp.LpMinimize)
    x = {c: pulp.LpVariable(f"x_{idx_map[c]}", lowBound=0, cat="Continuous") for c in countries}
    xe = {c: pulp.LpVariable(f"xe_{idx_map[c]}", lowBound=0, cat="Continuous") for c in countries}
    y = {c: pulp.LpVariable(f"y_{idx_map[c]}", lowBound=0, upBound=1, cat="Binary") for c in countries}
    reserve = pulp.LpVariable("reserve_release", lowBound=0, cat="Continuous")
    demand_cut = pulp.LpVariable("demand_cut", lowBound=0, cat="Continuous")
    shortage = pulp.LpVariable("shortage", lowBound=0, cat="Continuous")

    total_import = pulp.lpSum([x[c] + xe[c] for c in countries])

    cost_expr = pulp.lpSum(cost_norm[c] * x[c] + 1.25 * cost_norm[c] * xe[c] for c in countries)
    risk_expr = pulp.lpSum(risk_norm[c] * x[c] + 1.15 * risk_norm[c] * xe[c] for c in countries)
    div_expr = pulp.lpSum(y[c] for c in countries)

    model += (
        w_cost * cost_expr
        + w_risk * risk_expr
        - w_div * (q_min / len(countries)) * div_expr
        + 8.0 * demand_cut
        + 30.0 * shortage
    )

    model += total_import + reserve + demand_cut + shortage >= q_min
    model += total_import + reserve <= q_min * 1.03
    model += demand_cut <= scenario.demand_cut_ratio * q_min
    model += reserve <= scenario.reserve_cap_ratio * q_min

    min_lot = 80.0
    for c in countries:
        model += x[c] <= float(cap.loc[c]) * y[c]
        model += xe[c] <= float(emergency_cap.loc[c])
        model += x[c] >= min_lot * y[c]
        model += x[c] + xe[c] <= scenario.country_share_cap * q_min

    model += pulp.lpSum(y[c] for c in countries) >= 8

    me_members = [c for c in countries if c in q3.MIDDLE_EAST]
    if me_members:
        model += pulp.lpSum(x[c] + xe[c] for c in me_members) <= scenario.middle_east_cap * q_min

    for route, members in q3.ROUTE_GROUPS.items():
        valid = [c for c in members if c in x]
        if valid:
            model += pulp.lpSum(x[c] + xe[c] for c in valid) <= route_cap[route] + route_emergency_extra[route]

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    if pulp.LpStatus[model.status] != "Optimal":
        return None

    vals = np.array([float((x[c].value() or 0.0) + (xe[c].value() or 0.0)) for c in countries], dtype=float)
    total = float(vals.sum())
    if total < 1e-9:
        return None

    shares = vals / total
    hhi = float((shares**2).sum())
    cost_score = float(sum((cost_norm[c] * (x[c].value() or 0.0) + 1.25 * cost_norm[c] * (xe[c].value() or 0.0)) for c in countries))
    risk_score = float(sum((risk_norm[c] * (x[c].value() or 0.0) + 1.15 * risk_norm[c] * (xe[c].value() or 0.0)) for c in countries))

    return {
        "w_cost": w_cost,
        "w_risk": w_risk,
        "w_div": w_div,
        "cost_index": cost_score,
        "risk_index": risk_score,
        "hhi_concentration": hhi,
    }


def _is_dominated(row: pd.Series, df: pd.DataFrame) -> bool:
    cond = (
        (df["cost_index"] <= row["cost_index"])
        & (df["risk_index"] <= row["risk_index"])
        & (df["hhi_concentration"] <= row["hhi_concentration"])
        & (
            (df["cost_index"] < row["cost_index"])
            | (df["risk_index"] < row["risk_index"])
            | (df["hhi_concentration"] < row["hhi_concentration"])
        )
    )
    return bool(cond.any())


def generate_pareto_frontier_plot() -> tuple[Path, Path]:
    base_df, q_min = q3.load_base_data()
    scenario = [s for s in q3.scenario_set() if s.name == "S1"][0]

    rows: list[dict[str, float]] = []
    for wc in [0.20, 0.35, 0.50, 0.65]:
        for wr in [0.20, 0.35, 0.50, 0.65]:
            for wd in [0.05, 0.10, 0.15]:
                s = wc + wr + wd
                result = _solve_weighted_s1(base_df, q_min, scenario, wc / s, wr / s, wd / s)
                if result is not None:
                    rows.append(result)

    df = pd.DataFrame(rows).drop_duplicates(subset=["cost_index", "risk_index", "hhi_concentration"]).reset_index(drop=True)
    df["is_pareto"] = ~df.apply(lambda r: _is_dominated(r, df), axis=1)

    csv_path = PROBLEM3_DIR / "问题3_帕累托解集.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    others = df[~df["is_pareto"]]
    pareto = df[df["is_pareto"]]

    # Knee point: minimum normalized aggregate across three objectives.
    pareto_norm = pareto[["cost_index", "risk_index", "hhi_concentration"]].copy()
    for c in pareto_norm.columns:
        cmin = float(pareto_norm[c].min())
        cmax = float(pareto_norm[c].max())
        pareto_norm[c] = 0.0 if (cmax - cmin) < 1e-12 else (pareto_norm[c] - cmin) / (cmax - cmin)
    knee_idx = (pareto_norm.sum(axis=1)).idxmin()
    knee_row = pareto.loc[knee_idx]

    sc1 = axes[0].scatter(
        others["cost_index"],
        others["risk_index"],
        c=others["hhi_concentration"],
        cmap="Blues",
        s=32,
        alpha=0.65,
        edgecolors="none",
        label="Feasible",
    )
    axes[0].scatter(
        pareto["cost_index"],
        pareto["risk_index"],
        c=pareto["hhi_concentration"],
        cmap="Reds",
        s=56,
        edgecolors="#333333",
        linewidths=0.4,
        label="Pareto",
    )
    axes[0].scatter(
        [knee_row["cost_index"]],
        [knee_row["risk_index"]],
        s=95,
        marker="*",
        color="#2F4B7C",
        edgecolors="white",
        linewidths=0.8,
        label="Knee",
        zorder=5,
    )
    pareto_sorted = pareto.sort_values("cost_index")
    axes[0].plot(pareto_sorted["cost_index"], pareto_sorted["risk_index"], color="#E15759", linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel("Cost Index")
    axes[0].set_ylabel("Risk Index")
    axes[0].set_title("(a) Cost-Risk Plane")
    axes[0].legend(loc="upper right")
    cbar1 = fig.colorbar(sc1, ax=axes[0], fraction=0.045, pad=0.02)
    cbar1.set_label("HHI")

    sc2 = axes[1].scatter(
        df["risk_index"],
        df["hhi_concentration"],
        c=df["cost_index"],
        cmap="viridis",
        s=np.clip(df["w_div"] * 650, 25, 120),
        alpha=0.8,
        edgecolors="#222222",
        linewidths=0.25,
    )
    axes[1].set_xlabel("Risk Index")
    axes[1].set_ylabel("HHI Concentration")
    axes[1].set_title("(b) Risk-Concentration Plane")
    cbar2 = fig.colorbar(sc2, ax=axes[1], fraction=0.045, pad=0.02)
    cbar2.set_label("Cost Index")
    axes[1].text(
        0.02,
        0.98,
        "Marker size ~ diversity weight",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#BBBBBB"},
    )
    axes[1].scatter(
        [knee_row["risk_index"]],
        [knee_row["hhi_concentration"]],
        s=95,
        marker="*",
        color="#2F4B7C",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    axes[1].annotate(
        "Knee",
        xy=(knee_row["risk_index"], knee_row["hhi_concentration"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color="#2F4B7C",
        weight="bold",
    )

    fig.suptitle("Problem 3 Pareto Frontier (S1 Weight Sweep)", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()

    img_path = PROBLEM3_DIR / "问题3_帕累托前沿图.png"
    _save_figure(fig, img_path)
    plt.close(fig)
    return img_path, csv_path


def _aggregate_countries(series: pd.Series, categories: list[str]) -> pd.Series:
    out = pd.Series(0.0, index=categories + ["OTHER"], dtype=float)
    for c, v in series.items():
        if c in categories:
            out[c] += float(v)
        else:
            out["OTHER"] += float(v)
    return out


def _build_transfer_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    n = len(left)
    flows = np.zeros((n, n), dtype=float)

    stay = np.minimum(left, right)
    for i in range(n):
        flows[i, i] = stay[i]

    supply = left - stay
    demand = right - stay
    i = 0
    j = 0
    while i < n and j < n:
        if supply[i] <= 1e-9:
            i += 1
            continue
        if demand[j] <= 1e-9:
            j += 1
            continue
        f = min(supply[i], demand[j])
        flows[i, j] += f
        supply[i] -= f
        demand[j] -= f
    return flows


def _draw_single_alluvial(ax: plt.Axes, labels: list[str], left_vals: np.ndarray, right_vals: np.ndarray, title: str) -> None:
    total = max(float(left_vals.sum()), float(right_vals.sum()), 1e-9)
    left_h = left_vals / total
    right_h = right_vals / total

    gap = 0.006
    left_y0 = np.cumsum(np.r_[0.0, left_h[:-1] + gap])
    right_y0 = np.cumsum(np.r_[0.0, right_h[:-1] + gap])

    flow = _build_transfer_matrix(left_h, right_h)
    left_off = np.zeros(len(labels), dtype=float)
    right_off = np.zeros(len(labels), dtype=float)

    palette = plt.get_cmap("tab20c")(np.linspace(0, 1, len(labels)))
    xs = np.linspace(0.24, 0.76, 70)
    t = _smoothstep((xs - xs.min()) / (xs.max() - xs.min()))

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = flow[i, j]
            if v <= 1e-6:
                continue
            yl0 = left_y0[i] + left_off[i]
            yr0 = right_y0[j] + right_off[j]
            yl1 = yl0 + v
            yr1 = yr0 + v

            yb = yl0 + (yr0 - yl0) * t
            yt = yl1 + (yr1 - yl1) * t
            ax.fill_between(xs, yb, yt, color=palette[i], alpha=0.52, linewidth=0)

            left_off[i] += v
            right_off[j] += v

    for i, lbl in enumerate(labels):
        ax.add_patch(plt.Rectangle((0.08, left_y0[i]), 0.10, left_h[i], facecolor=palette[i], alpha=0.88, ec="white", lw=0.7))
        ax.add_patch(plt.Rectangle((0.82, right_y0[i]), 0.10, right_h[i], facecolor=palette[i], alpha=0.88, ec="white", lw=0.7))
        if left_h[i] > 0.05:
            ax.text(0.075, left_y0[i] + left_h[i] / 2, lbl, ha="right", va="center", fontsize=7)
        if right_h[i] > 0.05:
            ax.text(0.925, right_y0[i] + right_h[i] / 2, lbl, ha="left", va="center", fontsize=7)

    ax.text(0.13, 1.02, "Baseline", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(0.87, 1.02, "Scenario", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def generate_supply_reconstruction_sankey() -> Path:
    detail = pd.read_csv(PROBLEM3_DIR / "问题3_国别调配方案.csv")
    baseline = detail[detail["scenario"] == "S0"].groupby("country", as_index=True)["baseline_import_10k_tons"].first()

    top_countries_cn = baseline.sort_values(ascending=False).head(6).index.tolist()
    labels_cn = top_countries_cn + ["OTHER"]
    labels_en = [q3.COUNTRY_EN.get(c, c) if c != "OTHER" else "Other" for c in labels_cn]
    base_agg = _aggregate_countries(baseline, top_countries_cn).reindex(labels_cn)

    scenarios = ["S1", "S2", "S3", "S4"]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))

    for ax, scen in zip(axes.flatten(), scenarios):
        scen_series = detail[detail["scenario"] == scen].groupby("country", as_index=True)["optimized_import_10k_tons"].sum()
        scen_agg = _aggregate_countries(scen_series, top_countries_cn).reindex(labels_cn)
        _draw_single_alluvial(ax, labels_en, base_agg.to_numpy(dtype=float), scen_agg.to_numpy(dtype=float), f"Baseline -> {scen}")

    fig.suptitle("Problem 3 Supply Reconstruction Sankey", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = PROBLEM3_DIR / "问题3_供给重构桑基图.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    _set_publication_style()
    PROBLEM1_DIR.mkdir(parents=True, exist_ok=True)
    PROBLEM3_DIR.mkdir(parents=True, exist_ok=True)

    p1 = generate_residual_diagnostic_plot()
    p2 = generate_tornado_plot()
    p3 = generate_pressure_heatmap()
    p4, p4_csv = generate_pareto_frontier_plot()
    p5 = generate_supply_reconstruction_sankey()

    print("增强可视化生成完成：")
    print(f"- {p1}")
    print(f"- {p2}")
    print(f"- {p3}")
    print(f"- {p4}")
    print(f"- {p4_csv}")
    print(f"- {p5}")


if __name__ == "__main__":
    main()
