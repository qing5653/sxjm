from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "2025年原油进口15国数据.csv"
RISK_FILE = ROOT / "results" / "problem2" / "问题2_国别综合风险评分.csv"
Q1_FORECAST_FILE = ROOT / "results" / "problem1" / "问题1_2026_2028预测.csv"
OUT_DIR = ROOT / "results" / "problem3"

# 以 2026 年预测值作为应急保障需求，若不存在则回退到 2025 年实际进口总量。
DEFAULT_Q_MIN = 57773.0

MIDDLE_EAST = {"沙特阿拉伯", "伊拉克", "阿联酋", "阿曼", "科威特", "卡塔尔"}
TOP3_2025 = {"俄罗斯", "沙特阿拉伯", "伊拉克"}

COUNTRY_EN = {
    "俄罗斯": "Russia",
    "沙特阿拉伯": "Saudi Arabia",
    "伊拉克": "Iraq",
    "马来西亚": "Malaysia",
    "巴西": "Brazil",
    "阿联酋": "UAE",
    "阿曼": "Oman",
    "安哥拉": "Angola",
    "科威特": "Kuwait",
    "加拿大": "Canada",
    "印度尼西亚": "Indonesia",
    "哥伦比亚": "Colombia",
    "厄瓜多尔": "Ecuador",
    "卡塔尔": "Qatar",
    "刚果民主共和国": "DR Congo",
}

ROUTE_GROUPS = {
    "gulf_route": ["沙特阿拉伯", "伊拉克", "阿联酋", "阿曼", "科威特", "卡塔尔"],
    "asia_route": ["马来西亚", "印度尼西亚"],
    "atlantic_route": ["巴西", "安哥拉", "哥伦比亚", "厄瓜多尔", "刚果民主共和国"],
    "north_route": ["俄罗斯", "加拿大"],
}


@dataclass
class Scenario:
    name: str
    desc: str
    country_shock: dict[str, float]
    route_cap_ratio: dict[str, float]
    route_emergency_ratio: dict[str, float]
    middle_east_cap: float
    country_share_cap: float
    reserve_cap_ratio: float
    emergency_procure_ratio: float
    demand_cut_ratio: float


def load_base_data() -> tuple[pd.DataFrame, float]:
    raw = pd.read_csv(DATA_FILE)
    risk_df = pd.read_csv(RISK_FILE)

    base = raw[["贸易伙伴名称", "第一数量", "人民币"]].copy()
    base["import_10k_tons_base"] = pd.to_numeric(base["第一数量"], errors="coerce") / 1e7
    base["price_rmb_per_kg"] = pd.to_numeric(base["人民币"], errors="coerce") / pd.to_numeric(base["第一数量"], errors="coerce")
    base = base[["贸易伙伴名称", "import_10k_tons_base", "price_rmb_per_kg"]]

    risk_keep = risk_df[["贸易伙伴名称", "risk_score"]].copy()
    merged = base.merge(risk_keep, on="贸易伙伴名称", how="left")
    if merged["risk_score"].isna().any():
        missing = merged.loc[merged["risk_score"].isna(), "贸易伙伴名称"].tolist()
        raise ValueError(f"问题2风险分缺失，请先运行问题2。缺失国家: {missing}")

    q_min = DEFAULT_Q_MIN
    if Q1_FORECAST_FILE.exists():
        q1 = pd.read_csv(Q1_FORECAST_FILE)
        row = q1.loc[q1["year"] == 2026]
        if not row.empty:
            if "forecast_blend_10k_tons" in row.columns:
                q_min = float(row.iloc[0]["forecast_blend_10k_tons"])
            elif "forecast_10k_tons" in row.columns:
                q_min = float(row.iloc[0]["forecast_10k_tons"])

    return merged.sort_values("import_10k_tons_base", ascending=False).reset_index(drop=True), q_min


def scenario_set() -> list[Scenario]:
    return [
        Scenario(
            name="S0",
            desc="常态对照",
            country_shock={},
            route_cap_ratio={"gulf_route": 1.10, "asia_route": 1.08, "atlantic_route": 1.08, "north_route": 1.10},
            route_emergency_ratio={"gulf_route": 0.06, "asia_route": 0.08, "atlantic_route": 0.08, "north_route": 0.08},
            middle_east_cap=0.50,
            country_share_cap=0.22,
            reserve_cap_ratio=0.10,
            emergency_procure_ratio=0.08,
            demand_cut_ratio=0.005,
        ),
        Scenario(
            name="S1",
            desc="霍尔木兹海峡封锁",
            country_shock={c: 0.0 for c in MIDDLE_EAST},
            route_cap_ratio={"gulf_route": 0.00, "asia_route": 1.12, "atlantic_route": 1.25, "north_route": 1.20},
            route_emergency_ratio={"gulf_route": 0.00, "asia_route": 0.50, "atlantic_route": 0.70, "north_route": 0.55},
            middle_east_cap=0.00,
            country_share_cap=0.20,
            reserve_cap_ratio=0.25,
            emergency_procure_ratio=0.45,
            demand_cut_ratio=0.03,
        ),
        Scenario(
            name="S2",
            desc="俄乌冲突升级",
            country_shock={"俄罗斯": 0.5},
            route_cap_ratio={"gulf_route": 1.05, "asia_route": 1.05, "atlantic_route": 1.15, "north_route": 0.78},
            route_emergency_ratio={"gulf_route": 0.10, "asia_route": 0.12, "atlantic_route": 0.20, "north_route": 0.18},
            middle_east_cap=0.45,
            country_share_cap=0.22,
            reserve_cap_ratio=0.15,
            emergency_procure_ratio=0.15,
            demand_cut_ratio=0.01,
        ),
        Scenario(
            name="S3",
            desc="南海紧张",
            country_shock={"马来西亚": 0.5},
            route_cap_ratio={"gulf_route": 1.05, "asia_route": 0.82, "atlantic_route": 1.15, "north_route": 1.05},
            route_emergency_ratio={"gulf_route": 0.12, "asia_route": 0.18, "atlantic_route": 0.18, "north_route": 0.12},
            middle_east_cap=0.45,
            country_share_cap=0.22,
            reserve_cap_ratio=0.12,
            emergency_procure_ratio=0.12,
            demand_cut_ratio=0.01,
        ),
        Scenario(
            name="S4",
            desc="综合中断（前三大来源国减半）",
            country_shock={c: 0.5 for c in TOP3_2025},
            route_cap_ratio={"gulf_route": 0.70, "asia_route": 1.08, "atlantic_route": 1.20, "north_route": 0.82},
            route_emergency_ratio={"gulf_route": 0.16, "asia_route": 0.12, "atlantic_route": 0.24, "north_route": 0.16},
            middle_east_cap=0.35,
            country_share_cap=0.22,
            reserve_cap_ratio=0.25,
            emergency_procure_ratio=0.25,
            demand_cut_ratio=0.015,
        ),
    ]


def build_capacity(base_df: pd.DataFrame, scenario: Scenario) -> pd.Series:
    cap = base_df.set_index("贸易伙伴名称")["import_10k_tons_base"] * 1.35
    for country, ratio in scenario.country_shock.items():
        if country in cap.index:
            cap.loc[country] = cap.loc[country] * ratio
    return cap


def build_emergency_capacity(base_df: pd.DataFrame, scenario: Scenario) -> pd.Series:
    base = base_df.set_index("贸易伙伴名称")["import_10k_tons_base"]
    ecap = base * scenario.emergency_procure_ratio
    for country, ratio in scenario.country_shock.items():
        if country in ecap.index:
            ecap.loc[country] = ecap.loc[country] * ratio
    return ecap


def route_capacity(base_df: pd.DataFrame, scenario: Scenario) -> tuple[dict[str, float], dict[str, float]]:
    base_map = base_df.set_index("贸易伙伴名称")["import_10k_tons_base"].to_dict()
    out: dict[str, float] = {}
    emergency_extra: dict[str, float] = {}
    for route, members in ROUTE_GROUPS.items():
        base_total = float(sum(base_map.get(c, 0.0) for c in members))
        out[route] = base_total * scenario.route_cap_ratio.get(route, 1.0)
        emergency_extra[route] = base_total * scenario.route_emergency_ratio.get(route, 0.0)
    return out, emergency_extra


def optimize_scenario(base_df: pd.DataFrame, q_min: float, scenario: Scenario) -> tuple[pd.DataFrame, dict[str, float]]:
    countries = base_df["贸易伙伴名称"].tolist()
    idx_map = {c: i for i, c in enumerate(countries)}
    price = base_df.set_index("贸易伙伴名称")["price_rmb_per_kg"].to_dict()
    risk = base_df.set_index("贸易伙伴名称")["risk_score"].to_dict()
    base_import = base_df.set_index("贸易伙伴名称")["import_10k_tons_base"].to_dict()

    cap = build_capacity(base_df, scenario)
    route_cap, route_emergency_extra = route_capacity(base_df, scenario)
    emergency_cap = build_emergency_capacity(base_df, scenario)

    # 用价格和风险的无量纲化加权构成单目标。
    mean_price = float(base_df["price_rmb_per_kg"].mean())
    mean_risk = float(base_df["risk_score"].mean())
    cost_norm = {c: price[c] / mean_price for c in countries}
    risk_norm = {c: risk[c] / mean_risk for c in countries}

    def solve_with_params(
        min_suppliers: int, reserve_ratio_scale: float, route_scale: float
    ) -> tuple[
        pulp.LpProblem,
        dict[str, pulp.LpVariable],
        dict[str, pulp.LpVariable],
        dict[str, pulp.LpVariable],
        pulp.LpVariable,
        pulp.LpVariable,
        pulp.LpVariable,
    ]:
        model = pulp.LpProblem(f"Emergency_{scenario.name}", pulp.LpMinimize)

        x = {c: pulp.LpVariable(f"x_{idx_map[c]}", lowBound=0, cat="Continuous") for c in countries}
        xe = {c: pulp.LpVariable(f"xe_{idx_map[c]}", lowBound=0, cat="Continuous") for c in countries}
        y = {c: pulp.LpVariable(f"y_{idx_map[c]}", lowBound=0, upBound=1, cat="Binary") for c in countries}
        reserve = pulp.LpVariable("reserve_release", lowBound=0, cat="Continuous")
        demand_cut = pulp.LpVariable("demand_cut", lowBound=0, cat="Continuous")
        shortage = pulp.LpVariable("shortage", lowBound=0, cat="Continuous")

        total_import = pulp.lpSum([x[c] + xe[c] for c in countries])

        # 多目标线性加权：成本 + 风险 - 多元化激励。
        objective = (
            0.58 * pulp.lpSum(cost_norm[c] * x[c] for c in countries)
            + 0.58 * 1.25 * pulp.lpSum(cost_norm[c] * xe[c] for c in countries)
            + 0.37 * pulp.lpSum(risk_norm[c] * x[c] for c in countries)
            + 0.37 * 1.15 * pulp.lpSum(risk_norm[c] * xe[c] for c in countries)
            - 0.09 * (q_min / len(countries)) * pulp.lpSum(y[c] for c in countries)
            + 8.0 * demand_cut
            + 30.0 * shortage
        )
        model += objective

        # 保障需求约束：进口 + 储备释放 >= 最低需求。
        model += total_import + reserve + demand_cut + shortage >= q_min, "demand_floor"
        model += total_import + reserve <= q_min * 1.03, "demand_cap"
        model += demand_cut <= scenario.demand_cut_ratio * q_min, "demand_cut_cap"

        # 储备释放能力约束。
        model += reserve <= scenario.reserve_cap_ratio * reserve_ratio_scale * q_min, "reserve_cap"

        min_lot = 80.0
        for c in countries:
            model += x[c] <= float(cap.loc[c]) * y[c], f"country_cap_{idx_map[c]}"
            model += xe[c] <= float(emergency_cap.loc[c]), f"country_emergency_cap_{idx_map[c]}"
            model += x[c] >= min_lot * y[c], f"country_min_lot_{idx_map[c]}"
            model += x[c] + xe[c] <= scenario.country_share_cap * q_min, f"country_share_cap_{idx_map[c]}"

        model += pulp.lpSum(y[c] for c in countries) >= min_suppliers, "supplier_count"

        me_members = [c for c in countries if c in MIDDLE_EAST]
        if me_members:
            model += (
                pulp.lpSum(x[c] + xe[c] for c in me_members) <= scenario.middle_east_cap * q_min,
                "middle_east_cap",
            )

        for route, members in ROUTE_GROUPS.items():
            valid = [c for c in members if c in x]
            if valid:
                model += (
                    pulp.lpSum(x[c] + xe[c] for c in valid)
                    <= route_cap[route] * route_scale + route_emergency_extra[route],
                    f"route_cap_{route}",
                )

        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        return model, x, xe, y, reserve, demand_cut, shortage

    def zero_clip(value: float, eps: float = 1e-3) -> float:
        return 0.0 if abs(value) < eps else value

    model, x, xe, y, reserve, demand_cut, shortage = solve_with_params(min_suppliers=8, reserve_ratio_scale=1.0, route_scale=1.0)
    if pulp.LpStatus[model.status] != "Optimal":
        # 回退策略：在不改变核心逻辑下，放宽路线能力与储备释放上限，保证场景可求解。
        model, x, xe, y, reserve, demand_cut, shortage = solve_with_params(min_suppliers=7, reserve_ratio_scale=1.25, route_scale=1.10)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"场景 {scenario.name} 求解失败，状态={pulp.LpStatus[model.status]}")

    rows = []
    for c in countries:
        val_regular = float(x[c].value() or 0.0)
        val_emergency = float(xe[c].value() or 0.0)
        val = val_regular + val_emergency
        rows.append(
            {
                "scenario": scenario.name,
                "scenario_desc": scenario.desc,
                "country": c,
                "baseline_import_10k_tons": float(base_import[c]),
                "optimized_regular_10k_tons": val_regular,
                "optimized_emergency_10k_tons": val_emergency,
                "optimized_import_10k_tons": val,
                "delta_10k_tons": val - float(base_import[c]),
                "share_percent": val / q_min * 100,
                "risk_score": float(risk[c]),
                "price_rmb_per_kg": float(price[c]),
            }
        )

    detail_df = pd.DataFrame(rows).sort_values("optimized_import_10k_tons", ascending=False).reset_index(drop=True)

    import_total = float(detail_df["optimized_import_10k_tons"].sum())
    emergency_total = float(detail_df["optimized_emergency_10k_tons"].sum())
    reserve_val = float(reserve.value() or 0.0)
    demand_cut_val = float(demand_cut.value() or 0.0)
    shortage_val = float(shortage.value() or 0.0)
    delivered = import_total + reserve_val
    physical_shortage_val = max(q_min - delivered, 0.0)
    effective_shortage_val = max(q_min - (delivered + demand_cut_val), 0.0)
    shortage_val = zero_clip(shortage_val)
    physical_shortage_val = zero_clip(physical_shortage_val)
    effective_shortage_val = zero_clip(effective_shortage_val)

    me_import = float(detail_df.loc[detail_df["country"].isin(MIDDLE_EAST), "optimized_import_10k_tons"].sum())
    weighted_risk = float(
        (detail_df["optimized_import_10k_tons"] * detail_df["risk_score"]).sum() / max(import_total, 1e-9)
    )
    weighted_price = float(
        (detail_df["optimized_import_10k_tons"] * detail_df["price_rmb_per_kg"]).sum() / max(import_total, 1e-9)
    )
    shares = detail_df["optimized_import_10k_tons"] / max(import_total, 1e-9)
    hhi = float((shares**2).sum())

    summary = {
        "scenario": scenario.name,
        "scenario_desc": scenario.desc,
        "objective_value": float(pulp.value(model.objective)),
        "q_min_10k_tons": q_min,
        "import_total_10k_tons": import_total,
        "emergency_import_10k_tons": emergency_total,
        "reserve_release_10k_tons": reserve_val,
        "demand_cut_10k_tons": demand_cut_val,
        "delivered_total_10k_tons": delivered,
        "physical_shortage_10k_tons": physical_shortage_val,
        "effective_shortage_10k_tons": effective_shortage_val,
        "demand_cut_dependency_ratio": demand_cut_val / q_min,
        "shortage_10k_tons": shortage_val,
        "physical_supply_ratio": delivered / q_min,
        "supply_coverage_ratio": (delivered + demand_cut_val) / q_min,
        "middle_east_share_percent": me_import / q_min * 100,
        "weighted_risk_score": weighted_risk,
        "weighted_price_rmb_per_kg": weighted_price,
        "hhi_concentration": hhi,
        "active_supplier_count": int((detail_df["optimized_import_10k_tons"] > 1e-6).sum()),
    }
    return detail_df, summary


def save_plots(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 图1：情景供给结构（常规进口+应急进口+储备）
    fig1 = plt.figure(figsize=(9.5, 5.2))
    x = np.arange(len(summary_df))
    imp = summary_df["import_total_10k_tons"].to_numpy()
    emerg = summary_df["emergency_import_10k_tons"].to_numpy()
    res = summary_df["reserve_release_10k_tons"].to_numpy()
    dcut = summary_df["demand_cut_10k_tons"].to_numpy()
    base = np.clip(imp - emerg, 0.0, None)
    plt.bar(x, base, label="Regular Import", color="#4E79A7")
    plt.bar(x, emerg, bottom=base, label="Emergency Import", color="#76B7B2")
    plt.bar(x, res, bottom=base + emerg, label="Reserve", color="#F28E2B")
    plt.bar(x, dcut, bottom=base + emerg + res, label="Demand Cut", color="#EDC948")
    plt.xticks(x, summary_df["scenario"].tolist())
    plt.ylabel("Supply (10k tons)")
    plt.title("Q3 Emergency Supply Composition by Scenario")
    plt.legend()
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "问题3_情景供给结构.png", dpi=220)
    plt.close(fig1)

    # 图2：风险-集中度散点（展示结构质量）
    fig2 = plt.figure(figsize=(8.8, 5.2))
    plt.scatter(
        summary_df["hhi_concentration"],
        summary_df["weighted_risk_score"],
        s=220,
        c=summary_df["reserve_release_10k_tons"],
        cmap="YlOrRd",
        edgecolors="black",
        linewidths=0.7,
    )
    for _, row in summary_df.iterrows():
        plt.text(row["hhi_concentration"] + 0.0012, row["weighted_risk_score"] + 0.15, row["scenario"], fontsize=9)
    plt.xlabel("HHI Concentration")
    plt.ylabel("Weighted Risk Score")
    plt.title("Q3 Risk-Concentration Tradeoff")
    cbar = plt.colorbar()
    cbar.set_label("Reserve Release (10k tons)")
    plt.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "问题3_风险-集中度散点图.png", dpi=220)
    plt.close(fig2)

    # 图3：各场景Top8来源国配置热力图
    top_countries = (
        detail_df.groupby("country")["optimized_import_10k_tons"].mean().sort_values(ascending=False).head(8).index.tolist()
    )
    heat = (
        detail_df[detail_df["country"].isin(top_countries)]
        .pivot(index="country", columns="scenario", values="optimized_import_10k_tons")
        .reindex(index=top_countries)
        .fillna(0.0)
    )

    fig3 = plt.figure(figsize=(8.8, 5.6))
    im = plt.imshow(heat.to_numpy(), aspect="auto", cmap="Blues")
    heat_idx_label = [COUNTRY_EN.get(c, c) for c in heat.index.tolist()]
    plt.yticks(range(len(heat.index)), heat_idx_label)
    plt.xticks(range(len(heat.columns)), heat.columns.tolist())
    plt.title("Q3 Top Suppliers Allocation Heatmap")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f"{heat.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, label="Import (10k tons)")
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "问题3_国别调配热力图.png", dpi=220)
    plt.close(fig3)


def main() -> None:
    if not RISK_FILE.exists():
        raise FileNotFoundError(f"未找到问题2输出文件: {RISK_FILE}，请先运行问题2。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_df, q_min = load_base_data()

    detail_frames = []
    summary_rows = []
    for s in scenario_set():
        detail_df, summary = optimize_scenario(base_df, q_min, s)
        detail_frames.append(detail_df)
        summary_rows.append(summary)

    all_detail = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)

    kpi_df = summary_df[
        [
            "scenario",
            "scenario_desc",
            "emergency_import_10k_tons",
            "demand_cut_10k_tons",
            "physical_shortage_10k_tons",
            "effective_shortage_10k_tons",
            "demand_cut_dependency_ratio",
            "shortage_10k_tons",
            "physical_supply_ratio",
            "supply_coverage_ratio",
            "middle_east_share_percent",
            "weighted_risk_score",
            "weighted_price_rmb_per_kg",
            "hhi_concentration",
            "active_supplier_count",
        ]
    ].copy()

    summary_path = OUT_DIR / "问题3_情景优化结果.csv"
    detail_path = OUT_DIR / "问题3_国别调配方案.csv"
    kpi_path = OUT_DIR / "问题3_关键指标评估.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    all_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    kpi_df.to_csv(kpi_path, index=False, encoding="utf-8-sig")

    save_plots(summary_df, all_detail)

    print("问题3计算完成：")
    print(f"- {summary_path}")
    print(f"- {detail_path}")
    print(f"- {kpi_path}")
    print(f"- {OUT_DIR / '问题3_情景供给结构.png'}")
    print(f"- {OUT_DIR / '问题3_风险-集中度散点图.png'}")
    print(f"- {OUT_DIR / '问题3_国别调配热力图.png'}")


if __name__ == "__main__":
    main()
