#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_export_arterial_ph.py — export blood-gas pH (itemid 50820) for the analysis cohort.

Why: the pH column used in the first extraction merged blood-gas pH (itemid 50820, fluid = Blood)
with urine pH (itemid 51491, fluid = Urine) by label. The mixture produces the impossible
distribution seen in the analysis file (median 6.50, minimum 5.00). This script re-extracts only
the blood-gas item, takes the earliest value within the first 24 h of the first ICU admission,
and applies a physiological range filter.

Input : the labevents export that already carries icu_intime / icu_outtime per row
        (columns: subject_id, icu_intime, charttime, itemid, valuenum, fluid, label)
Output: arterial_ph_first24h.csv  ->  subject_id, ph_first24h, charttime

Usage:
    python 06_export_arterial_ph.py --labevents labevents.csv --out ./outputs
"""
import argparse, os

import numpy as np
import pandas as pd

PH_BLOOD = 50820      # blood gas pH
PH_URINE = 51491      # urine pH -- must NOT be mixed in
LOW, HIGH = 6.80, 7.80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labevents", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.labevents, low_memory=False)

    # 1. 两种 pH 的分布对照（用于证明混样）
    ph = df[df["itemid"].isin([PH_BLOOD, PH_URINE])].copy()
    print("== pH 按 itemid 的分布 ==")
    print(ph.groupby(["itemid", "fluid"])["valuenum"]
            .agg(n="size", median="median", q1=lambda s: s.quantile(.25),
                 q3=lambda s: s.quantile(.75), lo="min", hi="max").round(2).to_string())

    # 2. 只保留血气 pH
    ph = ph[(ph["itemid"] == PH_BLOOD) & ph["valuenum"].between(LOW, HIGH)].copy()
    for c in ("icu_intime", "charttime"):
        ph[c] = pd.to_datetime(ph[c], dayfirst=True, errors="coerce")
    ph = ph.dropna(subset=["icu_intime", "charttime", "valuenum"])

    # 3. 首次 ICU 入院，取入 ICU 后 24 h 内最早一次
    ph["stay_order"] = ph.groupby("subject_id")["icu_intime"].rank(method="dense")
    ph = ph[ph["stay_order"] == 1]
    ph = ph[(ph["charttime"] >= ph["icu_intime"]) &
            (ph["charttime"] < ph["icu_intime"] + pd.Timedelta(hours=24))]
    ph = ph.sort_values(["subject_id", "charttime"]).groupby("subject_id", as_index=False).first()

    out = ph[["subject_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "ph_first24h"})
    path = os.path.join(args.out, "arterial_ph_first24h.csv")
    out.to_csv(path, index=False)

    print(f"\n== 导出 ==")
    print(f"覆盖 subject 数: {len(out)}")
    print(f"中位数 {out.ph_first24h.median():.2f} "
          f"(Q1 {out.ph_first24h.quantile(.25):.2f}-Q3 {out.ph_first24h.quantile(.75):.2f}), "
          f"范围 {out.ph_first24h.min():.2f}-{out.ph_first24h.max():.2f}")
    print("写出:", path)


if __name__ == "__main__":
    main()
