from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "2025年原油进口15国数据.csv"
OUT_DIR = ROOT / "results" / "problem2"

# 评价等级（从低风险到高风险）
RISK_LEVELS = ["Very Low", "Low", "Medium", "High", "Very High"]
LEVEL_SCORES = np.array([20, 40, 60, 80, 100], dtype=float)

# AHP：准则层权重矩阵（B1~B5）
# B1 地缘政治风险、B2 运输通道风险、B3 经济成本风险、B4 供应稳定性风险、B5 新能源替代风险
AHP_MATRIX = np.array(
    [
        [1, 3, 4, 2, 5],
        [1 / 3, 1, 2, 1 / 2, 3],
        [1 / 4, 1 / 2, 1, 1 / 2, 2],
        [1 / 2, 2, 2, 1, 3],
        [1 / 5, 1 / 3, 1 / 2, 1 / 3, 1],
    ],
    dtype=float,
)

# Saaty 随机一致性指标 RI
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

# 专家先验分值（0-100，越大风险越高）
GEO_RISK = {
    "俄罗斯": 55,
    "沙特阿拉伯": 70,
    "伊拉克": 82,
    "马来西亚": 45,
    "巴西": 40,
    "阿联酋": 62,
    "阿曼": 60,
    "安哥拉": 68,
    "科威特": 65,
    "加拿大": 25,
    "印度尼西亚": 48,
    "哥伦比亚": 58,
    "厄瓜多尔": 62,
    "卡塔尔": 66,
    "刚果民主共和国": 90,
}

CHANNEL_RISK = {
    "俄罗斯": 42,
    "沙特阿拉伯": 85,
    "伊拉克": 88,
    "马来西亚": 50,
    "巴西": 38,
    "阿联酋": 82,
    "阿曼": 72,
    "安哥拉": 45,
    "科威特": 85,
    "加拿大": 35,
    "印度尼西亚": 48,
    "哥伦比亚": 42,
    "厄瓜多尔": 44,
    "卡塔尔": 84,
    "刚果民主共和国": 52,
}

STABILITY_RISK = {
    "俄罗斯": 58,
    "沙特阿拉伯": 40,
    "伊拉克": 88,
    "马来西亚": 35,
    "巴西": 45,
    "阿联酋": 30,
    "阿曼": 32,
    "安哥拉": 62,
    "科威特": 35,
    "加拿大": 20,
    "印度尼西亚": 40,
    "哥伦比亚": 55,
    "厄瓜多尔": 58,
    "卡塔尔": 28,
    "刚果民主共和国": 95,
}

DISTANCE_RISK = {
    "俄罗斯": 25,
    "沙特阿拉伯": 58,
    "伊拉克": 62,
    "马来西亚": 22,
    "巴西": 78,
    "阿联酋": 56,
    "阿曼": 52,
    "安哥拉": 74,
    "科威特": 60,
    "加拿大": 82,
    "印度尼西亚": 28,
    "哥伦比亚": 80,
    "厄瓜多尔": 84,
    "卡塔尔": 58,
    "刚果民主共和国": 72,
}

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

LEVEL_ORDER = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5}


def ahp_weights_and_consistency(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_idx].real)
    w = eigvecs[:, max_idx].real
    w = w / w.sum()

    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1)
    ri = RI_TABLE[n]
    cr = ci / ri if ri > 0 else 0.0
    return w, lambda_max, ci, cr


def minmax_scale(series: pd.Series) -> pd.Series:
    s_min = float(series.min())
    s_max = float(series.max())
    if s_max == s_min:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return (series - s_min) / (s_max - s_min) * 100


def fuzzy_membership(score: float) -> np.ndarray:
    """将 0-100 分映射到 5 个风险等级隶属度。"""
    centers = np.array([10, 30, 50, 70, 90], dtype=float)
    width = 20.0
    m = np.maximum(1 - np.abs(score - centers) / width, 0)
    s = float(m.sum())
    return m / s if s > 0 else np.array([0, 0, 1, 0, 0], dtype=float)


def build_country_scores(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["import_10k_tons"] = pd.to_numeric(work["第一数量"], errors="coerce") / 1e7
    work["rmb"] = pd.to_numeric(work["人民币"], errors="coerce")
    work["price_rmb_per_kg"] = work["rmb"] / pd.to_numeric(work["第一数量"], errors="coerce")

    total_import = float(work["import_10k_tons"].sum())
    work["share_percent"] = work["import_10k_tons"] / total_import * 100

    work["B1_geo"] = work["贸易伙伴名称"].map(GEO_RISK).astype(float)
    work["B2_channel"] = work["贸易伙伴名称"].map(CHANNEL_RISK).astype(float)
    work["B4_stability"] = work["贸易伙伴名称"].map(STABILITY_RISK).astype(float)

    dist = work["贸易伙伴名称"].map(DISTANCE_RISK).astype(float)
    price_scaled = minmax_scale(work["price_rmb_per_kg"])
    # 经济风险：单位成本与运输距离综合
    work["B3_economic"] = 0.6 * price_scaled + 0.4 * dist

    # 新能源替代风险在国别层面用“进口依赖集中度”代理
    work["B5_new_energy"] = minmax_scale(work["share_percent"])

    return work


def fuzzy_evaluate(country_df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    crit_cols = ["B1_geo", "B2_channel", "B3_economic", "B4_stability", "B5_new_energy"]

    memberships = []
    for _, row in country_df.iterrows():
        # R: 5x5，每行对应一个准则、每列对应一个评价等级
        r = np.vstack([fuzzy_membership(float(row[c])) for c in crit_cols])
        b = weights @ r
        score = float(b @ LEVEL_SCORES)
        level = RISK_LEVELS[int(np.argmax(b))]
        memberships.append((score, level, *b.tolist()))

    mem_df = pd.DataFrame(
        memberships,
        columns=[
            "risk_score",
            "risk_level",
            "m_very_low",
            "m_low",
            "m_medium",
            "m_high",
            "m_very_high",
        ],
    )

    out = pd.concat([country_df.reset_index(drop=True), mem_df], axis=1)
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return out


def evaluate_model_quality(result_df: pd.DataFrame, weights: np.ndarray, n_sim: int = 1000) -> pd.DataFrame:
    """输出可用于论文的模型质量指标。"""
    # 1) 区分度：离散系数（越大说明分层更明显）
    score_mean = float(result_df["risk_score"].mean())
    score_std = float(result_df["risk_score"].std(ddof=1))
    cv_score = score_std / score_mean if score_mean > 0 else 0.0

    # 2) 分级覆盖度：是否有多等级分布
    level_count = int(result_df["risk_level"].nunique())

    # 3) 稳健性：对权重做随机扰动，观察排名稳定性
    rng = np.random.default_rng(2026)
    base_rank = result_df["risk_score"].rank(ascending=False, method="average")

    crit_cols = ["B1_geo", "B2_channel", "B3_economic", "B4_stability", "B5_new_energy"]
    rank_corr_list = []
    top3_overlap_list = []
    top5_overlap_list = []

    base_top3 = set(result_df.nlargest(3, "risk_score")["贸易伙伴名称"].tolist())
    base_top5 = set(result_df.nlargest(5, "risk_score")["贸易伙伴名称"].tolist())

    for _ in range(n_sim):
        noise = rng.normal(1.0, 0.1, size=len(weights))
        w_perturb = np.clip(weights * noise, 1e-6, None)
        w_perturb = w_perturb / w_perturb.sum()

        scores = (result_df[crit_cols].to_numpy() @ w_perturb).astype(float)
        sim_df = result_df[["贸易伙伴名称"]].copy()
        sim_df["sim_score"] = scores
        sim_rank = sim_df["sim_score"].rank(ascending=False, method="average")

        corr = float(np.corrcoef(base_rank.to_numpy(dtype=float), sim_rank.to_numpy(dtype=float))[0, 1])
        rank_corr_list.append(corr)

        top3 = set(sim_df.nlargest(3, "sim_score")["贸易伙伴名称"].tolist())
        top5 = set(sim_df.nlargest(5, "sim_score")["贸易伙伴名称"].tolist())
        top3_overlap_list.append(len(base_top3 & top3) / 3)
        top5_overlap_list.append(len(base_top5 & top5) / 5)

    # 4) 解释性：指标贡献占比（基于加权均值）
    contrib_raw = np.array(
        [
            float((result_df["B1_geo"] * weights[0]).mean()),
            float((result_df["B2_channel"] * weights[1]).mean()),
            float((result_df["B3_economic"] * weights[2]).mean()),
            float((result_df["B4_stability"] * weights[3]).mean()),
            float((result_df["B5_new_energy"] * weights[4]).mean()),
        ]
    )
    contrib_share = contrib_raw / contrib_raw.sum()

    quality_df = pd.DataFrame(
        {
            "metric": [
                "score_std",
                "score_cv",
                "risk_level_count",
                "spearman_rank_corr_mean",
                "spearman_rank_corr_std",
                "top3_overlap_mean",
                "top5_overlap_mean",
                "contrib_B1_geo",
                "contrib_B2_channel",
                "contrib_B3_economic",
                "contrib_B4_stability",
                "contrib_B5_new_energy",
            ],
            "value": [
                score_std,
                cv_score,
                level_count,
                float(np.mean(rank_corr_list)),
                float(np.std(rank_corr_list, ddof=1)),
                float(np.mean(top3_overlap_list)),
                float(np.mean(top5_overlap_list)),
                float(contrib_share[0]),
                float(contrib_share[1]),
                float(contrib_share[2]),
                float(contrib_share[3]),
                float(contrib_share[4]),
            ],
            "unit": ["score", "-", "count", "-", "-", "ratio", "ratio", "ratio", "ratio", "ratio", "ratio", "ratio"],
            "good_if": [
                "higher better",
                "higher better",
                ">=3",
                ">=0.85",
                "lower better",
                ">=0.70",
                ">=0.80",
                "interpretability",
                "interpretability",
                "interpretability",
                "interpretability",
                "interpretability",
            ],
        }
    )
    return quality_df


def save_plots(result_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    en_names = result_df["贸易伙伴名称"].map(COUNTRY_EN).fillna(result_df["贸易伙伴名称"])

    # 图1：论文风格横向排名图（颜色映射分级）
    ranked = result_df.copy()
    ranked["country_en"] = en_names
    ranked = ranked.sort_values("risk_score", ascending=True)

    palette = {
        "Very Low": "#4CAF50",
        "Low": "#8BC34A",
        "Medium": "#FFC107",
        "High": "#FF7043",
        "Very High": "#D84315",
    }

    fig1 = plt.figure(figsize=(10.5, 7.2))
    colors = ranked["risk_level"].map(palette).tolist()
    bars = plt.barh(ranked["country_en"], ranked["risk_score"], color=colors)
    plt.xlabel("Comprehensive Risk Score")
    plt.title("Q2 Country Risk Ranking (AHP-Fuzzy)")
    plt.xlim(0, max(100, float(ranked["risk_score"].max()) + 5))
    for b, v in zip(bars, ranked["risk_score"]):
        plt.text(v + 0.7, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "问题2_国别风险排名.png", dpi=220)
    plt.close(fig1)

    top5 = result_df.head(5)
    heat_data = top5[["B1_geo", "B2_channel", "B3_economic", "B4_stability", "B5_new_energy"]].to_numpy()
    # 图2：更专业的自定义热力图
    cmap = LinearSegmentedColormap.from_list("paper_heat", ["#f7fbff", "#6baed6", "#08306b"])
    fig2 = plt.figure(figsize=(8.8, 5.2))
    im = plt.imshow(heat_data, aspect="auto", cmap=cmap)
    plt.yticks(range(len(top5)), top5["贸易伙伴名称"].map(COUNTRY_EN).tolist())
    plt.xticks(range(5), ["Geo", "Channel", "Economic", "Stability", "NewEnergy"])
    plt.title("Q2 Top-5 Risk Factor Heatmap")
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            plt.text(j, i, f"{heat_data[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, label="Risk Score")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "问题2_TOP5风险热力图.png", dpi=220)
    plt.close(fig2)

    # 图3：风险-占比气泡图（论文里展示风险与依赖程度关系）
    fig3 = plt.figure(figsize=(9.8, 6.2))
    size = np.clip(result_df["import_10k_tons"] / result_df["import_10k_tons"].max() * 1800, 120, 1800)
    color_num = result_df["risk_level"].map(LEVEL_ORDER).astype(float)
    sc = plt.scatter(
        result_df["share_percent"],
        result_df["risk_score"],
        s=size,
        c=color_num,
        cmap="RdYlGn_r",
        alpha=0.78,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in result_df.iterrows():
        plt.text(row["share_percent"] + 0.15, row["risk_score"] + 0.2, COUNTRY_EN.get(row["贸易伙伴名称"], row["贸易伙伴名称"]), fontsize=7)
    plt.xlabel("Import Share (%)")
    plt.ylabel("Comprehensive Risk Score")
    plt.title("Q2 Risk-Share Bubble Map")
    cbar = plt.colorbar(sc, ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(["VL", "L", "M", "H", "VH"])
    cbar.set_label("Risk Level")
    plt.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "问题2_风险占比气泡图.png", dpi=220)
    plt.close(fig3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(DATA_FILE)
    country_df = build_country_scores(raw_df)

    weights, lambda_max, ci, cr = ahp_weights_and_consistency(AHP_MATRIX)
    result_df = fuzzy_evaluate(country_df, weights)
    quality_df = evaluate_model_quality(result_df, weights, n_sim=1200)

    weight_df = pd.DataFrame(
        {
            "criterion": ["B1_geo", "B2_channel", "B3_economic", "B4_stability", "B5_new_energy"],
            "weight": weights,
        }
    )
    consistency_df = pd.DataFrame(
        {
            "metric": ["lambda_max", "CI", "CR", "consistency_pass"],
            "value": [lambda_max, ci, cr, str(cr < 0.1)],
        }
    )

    output_df = result_df[
        [
            "贸易伙伴名称",
            "import_10k_tons",
            "share_percent",
            "price_rmb_per_kg",
            "B1_geo",
            "B2_channel",
            "B3_economic",
            "B4_stability",
            "B5_new_energy",
            "risk_score",
            "risk_level",
            "m_very_low",
            "m_low",
            "m_medium",
            "m_high",
            "m_very_high",
        ]
    ]

    country_path = OUT_DIR / "问题2_国别综合风险评分.csv"
    weight_path = OUT_DIR / "问题2_AHP权重.csv"
    consistency_path = OUT_DIR / "问题2_AHP一致性检验.csv"
    quality_path = OUT_DIR / "问题2_模型质量评估.csv"

    output_df.to_csv(country_path, index=False, encoding="utf-8-sig")
    weight_df.to_csv(weight_path, index=False, encoding="utf-8-sig")
    consistency_df.to_csv(consistency_path, index=False, encoding="utf-8-sig")
    quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")

    save_plots(output_df)

    print("问题2计算完成：")
    print(f"- {country_path}")
    print(f"- {weight_path}")
    print(f"- {consistency_path}")
    print(f"- {quality_path}")
    print(f"- {OUT_DIR / '问题2_国别风险排名.png'}")
    print(f"- {OUT_DIR / '问题2_TOP5风险热力图.png'}")
    print(f"- {OUT_DIR / '问题2_风险占比气泡图.png'}")


if __name__ == "__main__":
    main()
