from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

import run_q3 as q3


OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "problem3"


def evaluate_case(case_name: str, scenarios: list[q3.Scenario], base_df: pd.DataFrame, q_min: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for sc in scenarios:
        _, summary = q3.optimize_scenario(base_df, q_min, sc)
        rows.append(
            {
                "case": case_name,
                "scenario": sc.name,
                "scenario_desc": sc.desc,
                "emergency_procure_ratio": sc.emergency_procure_ratio,
                "reserve_cap_ratio": sc.reserve_cap_ratio,
                "demand_cut_ratio": sc.demand_cut_ratio,
                "physical_supply_ratio": summary["physical_supply_ratio"],
                "supply_coverage_ratio": summary["supply_coverage_ratio"],
                "physical_shortage_10k_tons": summary["physical_shortage_10k_tons"],
                "effective_shortage_10k_tons": summary["effective_shortage_10k_tons"],
                "shortage_10k_tons": summary["shortage_10k_tons"],
                "demand_cut_10k_tons": summary["demand_cut_10k_tons"],
                "demand_cut_dependency_ratio": summary["demand_cut_dependency_ratio"],
                "hhi_concentration": summary["hhi_concentration"],
                "weighted_risk_score": summary["weighted_risk_score"],
                "active_supplier_count": summary["active_supplier_count"],
            }
        )
    return rows


def scale_one_param(scenarios: list[q3.Scenario], field: str, factor: float) -> list[q3.Scenario]:
    out: list[q3.Scenario] = []
    for sc in scenarios:
        val = getattr(sc, field)
        out.append(replace(sc, **{field: max(val * factor, 0.0)}))
    return out


def scale_three_params(scenarios: list[q3.Scenario], factor: float) -> list[q3.Scenario]:
    out: list[q3.Scenario] = []
    for sc in scenarios:
        out.append(
            replace(
                sc,
                emergency_procure_ratio=max(sc.emergency_procure_ratio * factor, 0.0),
                reserve_cap_ratio=max(sc.reserve_cap_ratio * factor, 0.0),
                demand_cut_ratio=max(sc.demand_cut_ratio * factor, 0.0),
            )
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_df, q_min = q3.load_base_data()
    base_scenarios = q3.scenario_set()

    all_rows: list[dict[str, float | str]] = []
    all_rows.extend(evaluate_case("baseline", base_scenarios, base_df, q_min))

    for factor, tag in [(0.9, "minus10"), (1.1, "plus10")]:
        all_rows.extend(
            evaluate_case(
                f"emergency_procure_ratio_{tag}",
                scale_one_param(base_scenarios, "emergency_procure_ratio", factor),
                base_df,
                q_min,
            )
        )
        all_rows.extend(
            evaluate_case(
                f"reserve_cap_ratio_{tag}",
                scale_one_param(base_scenarios, "reserve_cap_ratio", factor),
                base_df,
                q_min,
            )
        )
        all_rows.extend(
            evaluate_case(
                f"demand_cut_ratio_{tag}",
                scale_one_param(base_scenarios, "demand_cut_ratio", factor),
                base_df,
                q_min,
            )
        )
        all_rows.extend(
            evaluate_case(
                f"joint_{tag}",
                scale_three_params(base_scenarios, factor),
                base_df,
                q_min,
            )
        )

    detail_df = pd.DataFrame(all_rows)

    # 汇总重点看 S1（霍尔木兹封锁）在不同扰动下是否出现“崩溃”。
    s1_df = detail_df[detail_df["scenario"] == "S1"].copy()
    base_row = s1_df.loc[s1_df["case"] == "baseline"].iloc[0]
    s1_df["delta_physical_ratio"] = s1_df["physical_supply_ratio"] - base_row["physical_supply_ratio"]
    s1_df["delta_effective_ratio"] = s1_df["supply_coverage_ratio"] - base_row["supply_coverage_ratio"]
    s1_df["delta_physical_shortage_10k_tons"] = s1_df["physical_shortage_10k_tons"] - base_row["physical_shortage_10k_tons"]
    s1_df["delta_effective_shortage_10k_tons"] = s1_df["effective_shortage_10k_tons"] - base_row["effective_shortage_10k_tons"]
    s1_df["delta_shortage_10k_tons"] = s1_df["shortage_10k_tons"] - base_row["shortage_10k_tons"]
    s1_df["delta_hhi"] = s1_df["hhi_concentration"] - base_row["hhi_concentration"]

    def robust_label(row: pd.Series) -> str:
        if row["effective_shortage_10k_tons"] > 1e-6 or row["physical_supply_ratio"] < 0.95:
            return "fragile"
        if row["demand_cut_dependency_ratio"] > 0.02 or row["physical_supply_ratio"] < 0.98:
            return "pressured"
        return "stable"

    s1_df["robust_flag"] = s1_df.apply(robust_label, axis=1)

    detail_path = OUT_DIR / "问题3_敏感性分析_参数扰动明细.csv"
    summary_path = OUT_DIR / "问题3_敏感性分析_S1汇总.csv"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    s1_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("问题3敏感性分析完成：")
    print(f"- {detail_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
