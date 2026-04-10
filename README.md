# sxjm

数学建模工程（中国原油进口布局和安全策略）

## 目录结构

```text
sxjm/
├── data/                              # 原始数据附件
│   ├── 年度数据_补充版.csv
│   └── 2025年原油进口15国数据.csv
├── src/
│   ├── common/                        # 公共模块（预留）
│   └── problem1/
│       ├── gm11.py                    # GM(1,1) 灰色预测模型
│       └── run_q1.py                  # 问题1主程序
├── results/
│   └── problem1/                      # 问题1输出结果
├── 2026年南京理工大学数学建模竞赛A题.md
├── 建模分析.md
└── requirements.txt
```

## 问题1运行方式

```bash
/home/fishros/sxjm/.venv/bin/python src/problem1/run_q1.py
```

运行后在 `results/problem1/` 下生成：

- `问题1_2021_2025分析.csv`（近五年进口量、产量、增长率、依存度）
- `问题1_2026_2028预测.csv`（基础预测、优化预测、融合预测及95%区间）
- `问题1_模型评估.csv`（基础与优化精度、回测与滚动评估、融合权重）
- `问题1_进口量历史与预测.png`
- `问题1_增长率与依存度.png`

## 问题2运行方式

```bash
/home/fishros/sxjm/.venv/bin/python src/problem2/run_q2.py
```

运行后在 `results/problem2/` 下生成：

- `问题2_国别综合风险评分.csv`
- `问题2_AHP权重.csv`
- `问题2_AHP一致性检验.csv`
- `问题2_模型质量评估.csv`
- `问题2_国别风险排名.png`
- `问题2_TOP5风险热力图.png`
- `问题2_风险占比气泡图.png`
