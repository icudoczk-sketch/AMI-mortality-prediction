#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_sensitivity.py — the two sensitivity analyses reported in the manuscript.

(1) Split stability: the primary pipeline (XGBoost + within-fold univariate screening) is refitted
    under five random partitions; test-set AUCs are summarised as mean, SD and range.
(2) Multicollinearity: one member of each pair with |r| > 0.9 is removed and the model refitted.

Usage:
    python 03_sensitivity.py --data data_with_regions.xlsx --sheet Sheet4 --out outputs
"""
import argparse, contextlib, json, os, warnings

import threadpoolctl
threadpoolctl.threadpool_limits = lambda limits=None, user_api=None: contextlib.nullcontext()

import numpy as np
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("run_analysis", os.path.join(_here, "03_run_analysis.py"))
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

warnings.filterwarnings("ignore")
SEEDS = [1, 7, 42, 2024, 31337]


def fit_predict(X, y, idx_tr, idx_te, cols=None, seed=42):
    imp = KNNImputer(n_neighbors=5).fit(X[idx_tr])
    sc = StandardScaler().fit(imp.transform(X[idx_tr]))
    Ztr, Zte = sc.transform(imp.transform(X[idx_tr])), sc.transform(imp.transform(X[idx_te]))
    if cols is not None:
        Ztr, Zte = Ztr[:, cols], Zte[:, cols]
    Zb, yb = SMOTE(random_state=seed).fit_resample(Ztr, y[idx_tr])
    if cols is None:
        keep = list(np.where(ra.univariate_p(Zb, yb) < 0.05)[0])
        keep = keep or list(range(Zb.shape[1]))
        Zb, Zte = Zb[:, keep], Zte[:, keep]
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                      random_state=seed, n_jobs=1).fit(Zb, yb)
    return roc_auc_score(y[idx_te], m.predict_proba(Zte)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sheet", default="Sheet4")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    _, X, y, preds = ra.load_cohort(args.data, args.sheet)
    out = {}

    aucs = []
    for seed in SEEDS:
        a, b = train_test_split(np.arange(len(y)), test_size=ra.TEST_FRAC, stratify=y, random_state=seed)
        auc = fit_predict(X, y, a, b, seed=seed)
        aucs.append(auc)
        print(f"seed {seed}: test AUC {auc:.3f}")
    out["split_stability"] = dict(seeds=SEEDS, aucs=aucs, mean=float(np.mean(aucs)),
                                  sd=float(np.std(aucs, ddof=1)),
                                  min=float(min(aucs)), max=float(max(aucs)))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=ra.TEST_FRAC,
                                      stratify=y, random_state=ra.SEED)
    imp = KNNImputer(n_neighbors=5).fit(X[idx_tr])
    Z = StandardScaler().fit_transform(imp.transform(X[idx_tr]))
    C = np.corrcoef(Z.T)
    drop = {j for i in range(len(preds)) for j in range(i + 1, len(preds)) if abs(C[i, j]) > 0.9}
    keep = [j for j in range(len(preds)) if j not in drop]
    auc_pruned = fit_predict(X, y, idx_tr, idx_te, cols=keep)
    print(f"pruned: {len(preds)} -> {len(keep)} variables, test AUC {auc_pruned:.3f}")
    out["collinearity"] = dict(n_original=len(preds), n_retained=len(keep),
                               dropped=[preds[j] for j in sorted(drop)], auc=auc_pruned)

    with open(os.path.join(args.out, "sensitivity.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("done ->", os.path.join(args.out, "sensitivity.json"))


if __name__ == "__main__":
    main()
