#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_fix_plr.py —— 修正 PLR 定义并给出可粘贴的统计量

问题：现数据中 PLR = 血小板计数 / 淋巴细胞「百分比」。
     反推验证：以 data_with_regions.xlsx 计算，PLR 列与 plt/lymph% 的相对误差中位数为 0.0001，
     而与 plt/(lymph%*WBC/100) 相差约 7 倍——即分子分母量纲不一致。
     后果：PLR 中位数 22.8，而文献中 AMI/ICU 人群 PLR 通常在 150–250，无法与任何研究比较。

正确：PLR = 血小板计数(10^9/L) / 淋巴细胞绝对计数(10^9/L)
     淋巴细胞绝对计数 = 白细胞计数 × 淋巴细胞百分比 / 100

用法：
    python 02_fix_plr.py --input 统计分析数据.xlsx --sheet Dataset_5 \
                         --out plr_fixed.xlsx [--group-col 匹配结果] [--death-label 死亡]
"""

import argparse
import sys

import numpy as np
import pandas as pd


def pick(df, keywords, exclude=()):
    """按关键词找列名，返回第一个命中的列。"""
    for col in df.columns:
        low = str(col).lower()
        if any(k.lower() in low for k in keywords) and not any(e.lower() in low for e in exclude):
            return col
    return None


def describe(s, label):
    s = pd.Series(s).dropna()
    if s.empty:
        print(f"  {label}: 无有效值")
        return
    print(f"  {label}: n={len(s)}, 中位 {s.median():.2f} "
          f"(Q1 {s.quantile(.25):.2f}–Q3 {s.quantile(.75):.2f}), 最小 {s.min():.2f}, 最大 {s.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="含血小板/淋巴/白细胞的表格（xlsx 或 csv）")
    ap.add_argument("--sheet", default=0, help="xlsx 的 sheet 名或序号")
    ap.add_argument("--out", default="plr_fixed.xlsx")
    ap.add_argument("--group-col", default=None, help="分组列（如结局列 匹配结果）")
    ap.add_argument("--death-label", default=None, help="结局列中代表死亡的取值")
    args = ap.parse_args()

    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input, sheet_name=args.sheet)

    c_plt = pick(df, ["platelet", "血小板"], exclude=["plr", "比值"])
    c_lym = pick(df, ["lymph", "淋巴"], exclude=["plr", "比值", "neutrophil", "中性"])
    c_wbc = pick(df, ["white_blood", "wbc", "白细胞"], exclude=["plr", "比值", "neutrophil", "中性", "lymph"])
    c_plr = pick(df, ["plr"], exclude=[])

    missing = [n for n, c in [("血小板", c_plt), ("淋巴细胞", c_lym), ("白细胞", c_wbc)] if c is None]
    if missing:
        sys.exit(f"未找到列：{missing}；现有列：{list(df.columns)}")

    print(f"识别列：血小板={c_plt} | 淋巴细胞={c_lym} | 白细胞={c_wbc} | 原PLR={c_plr}")
    med_wbc = pd.to_numeric(df[c_wbc], errors="coerce").median()
    if med_wbc > 40:
        sys.exit(f"白细胞列中位数为 {med_wbc:.1f}，疑似选成了百分比列，请用 --wbc-col 指定正确列名")
    print(f"样本量：{len(df)}")

    plt_v = pd.to_numeric(df[c_plt], errors="coerce")
    lym_v = pd.to_numeric(df[c_lym], errors="coerce")
    wbc_v = pd.to_numeric(df[c_wbc], errors="coerce")

    # 自动识别淋巴细胞是百分比还是绝对值：百分比取值多在 0–100，且与 WBC 无强相关
    is_pct = lym_v.dropna().quantile(0.95) <= 100
    print(f"淋巴细胞列判定为：{'百分比' if is_pct else '绝对值 10^9/L'}")

    lymph_abs = wbc_v * lym_v / 100.0 if is_pct else lym_v
    plr_new = plt_v / lymph_abs.replace(0, np.nan)

    print("\n修正前后对比：")
    if c_plr:
        describe(df[c_plr], "原 PLR")
    describe(plr_new, "新 PLR")

    df["PLR_corrected"] = plr_new

    if args.group_col and args.group_col in df.columns:
        print(f"\n按 {args.group_col} 分组（中位数 Q1–Q3）：")
        for g, sub in df.groupby(args.group_col):
            describe(sub["PLR_corrected"], str(g))
        if args.death_label:
            a = df.loc[df[args.group_col].astype(str) == args.death_label, "PLR_corrected"].dropna()
            b = df.loc[df[args.group_col].astype(str) != args.death_label, "PLR_corrected"].dropna()
            if len(a) > 5 and len(b) > 5:
                from scipy.stats import mannwhitneyu
                p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                print(f"Mann-Whitney U 检验：P = {p:.4g}")

    df.to_excel(args.out, index=False)
    print(f"\n已写出：{args.out}")
    print("下一步：用 PLR_corrected 替换原 PLR 列，重跑 53 变量流程；")
    print("        表 1 的 PLR 行、表 6 的 GRA 排序、SHAP 贡献度均需同步更新。")


if __name__ == "__main__":
    main()
