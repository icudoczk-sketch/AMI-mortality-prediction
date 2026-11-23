#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_run_analysis.py — nested (leakage-free) analysis pipeline for the AMI 30-day mortality study.

Reproduces Tables 1-6 and the data behind Figs 2-6, S1 and S2.

Design (matches Methods of the manuscript):
  * 50 laboratory predictors measured in the first 24 h; arterial blood-gas pH excluded
    (extraction could not be restricted to arterial specimens); all composite indices are
    recomputed from their raw components, PLR from the absolute lymphocyte count.
  * 70/30 split stratified by outcome -> training / test.
  * 30% of the training partition held out as the calibration subset -> development subset.
  * Every data-dependent step (univariate screening, GRA ranking, LASSO, scaling, SMOTE) is
    fitted inside the training fold only, and re-learned in each cross-validation fold.
  * Four feature strategies x four model families, pooled out-of-fold and single test-set
    evaluation; DeLong tests on the test set; isotonic and Platt recalibration fitted on the
    calibration subset and applied once to the test set; SHAP (TreeSHAP) and decision-curve
    analysis for the primary model.

Usage:
    python 03_run_analysis.py --data data_with_regions.xlsx --sheet Sheet4 --out ./outputs
"""
import argparse, contextlib, json, os, pickle, warnings

import threadpoolctl
threadpoolctl.threadpool_limits = lambda limits=None, user_api=None: contextlib.nullcontext()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, chi2, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.impute import KNNImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, DMatrix

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- configuration
SEED = 42
TEST_FRAC = 0.30
CAL_FRAC = 0.30          # of the training partition
N_FOLDS = 5
MODELS = ["XGBoost", "RF", "LR", "SVM"]
STRATEGIES = ["all", "univariate", "gra16", "lasso"]

RAW = {  # column names in the extracted analysis file
    "plt": "血小板计数_Platelet_Count", "wbc": "白细胞_White_Blood_Cells",
    "lym": "淋巴细胞_Lymphocytes", "neu": "中性白细胞_Neutrophils",
    "alb": "白蛋白_Albumin", "cr": "肌酐_Creatinine", "bun": "尿素氮_Urea_Nitrogen",
    "ld": "乳酸脱氢酶_LD__Lactate_Dehydrogenase__LD_", "lac": "乳酸_Lactate",
    "rdw": "RDW公司_RDW", "ph": "酸碱度_pH", "outcome": "匹配结果",
}

LABELS = {
    "CK_MB指数": "CK-MB index", "MCHC公司": "MCHC", "MCH公司": "MCH", "PTT公司": "PTT",
    "RDW公司": "RDW", "pO2": "pO2", "丙氨酸氨基转移酶": "ALT", "中性白细胞": "Neutrophils (%)",
    "乳酸_Lactate": "Lactate", "乳酸脱氢酶": "LD", "二氧化碳": "pCO2", "单核细胞": "Monocytes (%)",
    "嗜碱性粒细胞": "Basophils (%)", "嗜酸性粒细胞": "Eosinophils (%)", "基本超额": "Base excess",
    "天冬氨酸": "AST", "尿素氮": "Urea nitrogen", "微型计算机": "MCV", "氯化物": "Chloride",
    "淋巴细胞": "Lymphocytes (%)", "白细胞": "WBC", "白蛋白": "Albumin", "碱性磷酸酶": "ALP",
    "碳酸氢盐": "Bicarbonate", "磅_PT": "PT", "磷酸盐": "Phosphate", "红细胞": "RBC",
    "肌酐": "Creatinine", "肌酸激酶_MB同工酶": "CK-MB", "肌酸激酶": "CK",
    "肌钙蛋白T": "Troponin T", "胆红素": "Total bilirubin", "英寸_磅": "INR (PT)",
    "葡萄糖": "Glucose", "血小板计数": "Platelet", "血红蛋白": "Hemoglobin",
    "血细胞比容": "Hematocrit", "计算总CO2": "Calculated total CO2", "钙_总量": "Total calcium",
    "钠": "Sodium", "钾": "Potassium", "镁": "Magnesium", "阴离子间隙": "Anion gap",
}
DERIVED = {"PLR", "NLR", "SII", "CAR", "UCR", "L/A", "LAR", "RPR", "ANC"}


def label(name):
    if name in DERIVED:
        return name
    for k, v in LABELS.items():
        if name.startswith(k):
            return v
    return name


# ----------------------------------------------------------------------------- data
def load_cohort(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    y = (df[RAW["outcome"]] == "死亡").astype(int).values

    df["PLR"] = df[RAW["plt"]] / (df[RAW["wbc"]] * df[RAW["lym"]] / 100.0)
    df["NLR"] = df[RAW["neu"]] / df[RAW["lym"]]
    df["SII"] = df[RAW["plt"]] * df["NLR"]
    df["CAR"] = df[RAW["cr"]] / df[RAW["alb"]]
    df["UCR"] = df[RAW["bun"]] / df[RAW["cr"]]
    df["L/A"] = df[RAW["lac"]] / df[RAW["alb"]]
    df["LAR"] = df[RAW["ld"]] / df[RAW["alb"]]
    df["RPR"] = df[RAW["rdw"]] / df[RAW["plt"]]
    df["ANC"] = df[RAW["wbc"]] * df[RAW["neu"]] / 100.0

    drop = {"subject_id", RAW["outcome"], RAW["ph"],
            "CAR血清肌酐与白蛋白比值", "GLR葡萄糖淋巴细胞比值", "UCR血尿素氮与肌酐比值",
            "L/A乳酸/白蛋白比值", "RPR红细胞分布宽度-血小板比值", "LAR乳酸脱氢酶与血清白蛋白比值",
            "PLR血小板与淋巴细胞比值", "abs_neutrophils_max", "SII", "NLR"}
    preds = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return df, df[preds].astype(float).values, y, preds


# ----------------------------------------------------------------------------- helpers
def make_model(name):
    if name == "LR":
        return LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      random_state=SEED, n_jobs=-1)
    if name == "SVM":
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                             random_state=SEED, n_jobs=1)
    raise ValueError(name)


def fit_prep(Xtr):
    imp = KNNImputer(n_neighbors=5).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return imp, sc


def univariate_p(Z, y):
    out = []
    for j in range(Z.shape[1]):
        try:
            out.append(sm.Logit(y, sm.add_constant(Z[:, [j]])).fit(disp=0).pvalues[1])
        except Exception:
            out.append(1.0)
    return np.array(out)


def univariate_or(Z, y):
    out = []
    for j in range(Z.shape[1]):
        try:
            out.append(float(np.exp(sm.Logit(y, sm.add_constant(Z[:, [j]])).fit(disp=0).params[1])))
        except Exception:
            out.append(1.0)
    return np.array(out)


def screen(Z, y):
    keep = list(np.where(univariate_p(Z, y) < 0.05)[0])
    return keep if keep else list(range(Z.shape[1]))


def gra(Z, y, rho=0.5):
    lo, hi = Z.min(0), Z.max(0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    N = (Z - lo) / span
    ors = univariate_or(Z, y)
    N[:, ors < 1] = 1 - N[:, ors < 1]          # reverse-score protective variables
    D = np.abs(y.astype(float)[:, None] - N)
    Xi = (D.min() + rho * D.max()) / (D + rho * D.max())
    return Xi.mean(0), ors


def lasso(Z, y):
    grid = [0.005, 0.01, 0.03, 0.1, 0.3, 1.0]
    best_C, best_auc = 0.1, -1
    for C in grid:
        aucs = []
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Z, y):
            m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000).fit(Z[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(Z[va])[:, 1]))
        a = float(np.mean(aucs))
        if a > best_auc:
            best_C, best_auc = C, a
    m = LogisticRegression(penalty="l1", C=best_C, solver="liblinear", max_iter=2000).fit(Z, y)
    sel = list(np.where(m.coef_[0] != 0)[0])
    return (sel if len(sel) >= 3 else list(range(Z.shape[1]))), best_C


def columns_for(strategy, Ztr, ytr):
    if strategy == "all":
        return list(range(Ztr.shape[1]))
    if strategy == "univariate":
        return screen(Ztr, ytr)
    if strategy == "gra16":
        g, _ = gra(Ztr, ytr)
        return list(np.argsort(-g)[:16])
    if strategy == "lasso":
        s, _ = lasso(Ztr, ytr)
        return s
    raise ValueError(strategy)


def midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def delong(y, p1, p2):
    o = np.argsort(-y); yy = y[o]; m = int(yy.sum()); n = len(yy) - m
    pred = np.vstack([p1[o], p2[o]])
    tx = np.vstack([midrank(pred[r, :m]) for r in range(2)])
    ty = np.vstack([midrank(pred[r, m:]) for r in range(2)])
    tz = np.vstack([midrank(pred[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * (1 - norm.cdf(abs(z))))


def metrics(y, p, thr=0.5):
    pr = (p >= thr).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    tn = int(((pr == 0) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    pp = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return dict(AUC=roc_auc_score(y, p), PRAUC=average_precision_score(y, p),
                Acc=(tp + tn) / len(y), Sens=se, Spec=sp, PPV=pp, NPV=npv,
                F1=2 * pp * se / max(pp + se, 1e-9), Brier=brier_score_loss(y, p))


def calib_metrics(y, p):
    eps = 1e-6
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression().fit(lg.reshape(-1, 1), y)
    order = np.argsort(p); gs = np.array_split(order, 10); chisq = 0.0
    for g in gs:
        ob = y[g].sum(); ex = p[g].sum(); k = len(g)
        if 0 < ex < k:
            chisq += (ob - ex) ** 2 / (ex * (1 - ex / k))
    return dict(Brier=brier_score_loss(y, p), slope=float(lr.coef_[0][0]),
                intercept=float(lr.intercept_[0]), HL=float(chisq),
                HL_P=float(1 - chi2.cdf(chisq, df=8)))


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sheet", default="Sheet4")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, X, y, preds = load_cohort(args.data, args.sheet)
    print(f"cohort {len(y)}  deaths {int(y.sum())} ({y.mean():.3f})  predictors {len(preds)}")

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_FRAC,
                                      stratify=y, random_state=SEED)
    idx_dv, idx_ca = train_test_split(idx_tr, test_size=CAL_FRAC,
                                      stratify=y[idx_tr], random_state=SEED)
    y_dv, y_ca, y_te = y[idx_dv], y[idx_ca], y[idx_te]
    print(f"dev {len(idx_dv)} ({y_dv.sum()}) | calib {len(idx_ca)} ({y_ca.sum()}) | test {len(idx_te)} ({y_te.sum()})")

    oof, test_auc, pred = {}, {}, {}
    for s in STRATEGIES:
        for m in MODELS:
            Xd = X[idx_dv]
            o = np.zeros(len(idx_dv))
            for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xd, y_dv):
                imp, sc = fit_prep(Xd[tr])
                Ztr, ytr = SMOTE(random_state=SEED).fit_resample(sc.transform(imp.transform(Xd[tr])), y_dv[tr])
                Zva = sc.transform(imp.transform(Xd[va]))
                cols = columns_for(s, Ztr, ytr)
                mdl = make_model(m).fit(Ztr[:, cols], ytr)
                o[va] = mdl.predict_proba(Zva[:, cols])[:, 1]
            oof[(s, m)] = roc_auc_score(y_dv, o)

            imp, sc = fit_prep(Xd)
            Ztr, ytr = SMOTE(random_state=SEED).fit_resample(sc.transform(imp.transform(Xd)), y_dv)
            cols = columns_for(s, Ztr, ytr)
            mdl = make_model(m).fit(Ztr[:, cols], ytr)
            p_te = mdl.predict_proba(sc.transform(imp.transform(X[idx_te]))[:, cols])[:, 1]
            p_ca = mdl.predict_proba(sc.transform(imp.transform(X[idx_ca]))[:, cols])[:, 1]
            test_auc[(s, m)] = roc_auc_score(y_te, p_te)
            pred[(s, m)] = dict(p_te=p_te, p_ca=p_ca, oof=o, n_vars=len(cols),
                                vars=[preds[c] for c in cols])
            print(f"  {s:11s} {m:8s} OOF {oof[(s,m)]:.3f}  test {test_auc[(s,m)]:.3f}  vars {len(cols)}")

    PRIM_S, PRIM_M = "univariate", "XGBoost"

    # ---- Table 2 ----
    t2 = pd.DataFrame({s: {m: f"{oof[(s,m)]:.3f} / {test_auc[(s,m)]:.3f}" for m in MODELS} for s in STRATEGIES})
    t2.index.name = "Model"
    t2.to_csv(f"{args.out}/table2.csv")

    # ---- Table 3: DeLong on the test set ----
    pairs = [(PRIM_M, PRIM_M, "LR"), (PRIM_M, PRIM_M, "RF"), (PRIM_M, PRIM_M, "SVM"),
             ("gra16", "XGBoost", None), ("all", "XGBoost", None), ("lasso", "XGBoost", None)]
    rows = []
    for p1, m1, m2 in pairs:
        if m2 is None:
            s1 = p1; a1, a2, z, p = delong(y_te, pred[(s1, "XGBoost")]["p_te"],
                                           pred[(PRIM_S, "XGBoost")]["p_te"])
            rows.append(dict(comparison=f"{s1} XGBoost vs {PRIM_S} XGBoost", AUC_1=a1, AUC_2=a2, P=p))
        else:
            a1, a2, z, p = delong(y_te, pred[(PRIM_S, m1)]["p_te"], pred[(PRIM_S, m2)]["p_te"])
            rows.append(dict(comparison=f"{m1} vs {m2} ({PRIM_S})", AUC_1=a1, AUC_2=a2, P=p))
    pd.DataFrame(rows).to_csv(f"{args.out}/table3.csv", index=False)

    # ---- Table 4 + bootstrap CI ----
    rng = np.random.default_rng(SEED)
    t4 = []
    for m in MODELS:
        d = metrics(y_te, pred[(PRIM_S, m)]["p_te"])
        v = []
        for _ in range(2000):
            i = rng.integers(0, len(y_te), len(y_te))
            if len(np.unique(y_te[i])) < 2:
                continue
            v.append(roc_auc_score(y_te[i], pred[(PRIM_S, m)]["p_te"][i]))
        d.update(model=m, lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        t4.append(d)
    pd.DataFrame(t4).to_csv(f"{args.out}/table4.csv", index=False)

    # ---- calibration ----
    fpr, tpr, thr = roc_curve(y_ca, pred[(PRIM_S, PRIM_M)]["p_ca"])
    youden = float(thr[int(np.argmax(tpr - fpr))])
    p_ca = pred[(PRIM_S, PRIM_M)]["p_ca"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, y_ca)
    platt = LogisticRegression().fit(logit(p_ca).reshape(-1, 1), y_ca)
    p_iso = iso.predict(pred[(PRIM_S, PRIM_M)]["p_te"])
    p_plt = platt.predict_proba(logit(pred[(PRIM_S, PRIM_M)]["p_te"]).reshape(-1, 1))[:, 1]
    t5 = {"Raw": calib_metrics(y_te, pred[(PRIM_S, PRIM_M)]["p_te"]),
          "Isotonic": calib_metrics(y_te, p_iso), "Platt": calib_metrics(y_te, p_plt)}
    pd.DataFrame([{"method": k, **v} for k, v in t5.items()]).to_csv(f"{args.out}/table5.csv", index=False)

    # ---- Table 6: GRA ranking ----
    imp_d, sc_d = fit_prep(X[idx_dv])
    Z_d = sc_d.transform(imp_d.transform(X[idx_dv]))
    grades, ors = gra(Z_d, y_dv)
    order = np.argsort(-grades)[:16]
    t6 = []
    for rank, j in enumerate(order, start=1):
        orv, lo_, hi_, pv = np.nan, np.nan, np.nan, np.nan
        try:
            fit = sm.Logit(y_dv, sm.add_constant(Z_d[:, [j]])).fit(disp=0)
            orv, pv = float(np.exp(fit.params[1])), float(fit.pvalues[1])
            lo_, hi_ = float(np.exp(fit.params[1] - 1.96 * fit.bse[1])), float(np.exp(fit.params[1] + 1.96 * fit.bse[1]))
        except Exception:
            pass
        t6.append(dict(rank=rank, variable=label(preds[j]), raw=preds[j],
                       grade=float(grades[j]), OR=orv, lo=lo_, hi=hi_, P=pv))
    pd.DataFrame(t6).to_csv(f"{args.out}/table6.csv", index=False)

    # ---- SHAP (TreeSHAP through XGBoost) ----
    Ztr, ytr = SMOTE(random_state=SEED).fit_resample(Z_d, y_dv)
    cols = columns_for(PRIM_S, Ztr, ytr)
    xgb = make_model("XGBoost").fit(Ztr[:, cols], ytr)
    contrib = xgb.get_booster().predict(DMatrix(Z_d[:, cols]), pred_contribs=True)[:, :-1]
    shap_mean = np.abs(contrib).mean(0)
    top = np.argsort(-shap_mean)[:15]

    # ---- Table 1 ----
    t1 = []
    for j, c in enumerate(preds):
        a = X[y == 0, j]; b = X[y == 1, j]
        t1.append(dict(variable=label(c), raw=c,
                       survivors=f"{np.median(a):.2f} ({np.percentile(a,25):.2f}–{np.percentile(a,75):.2f})",
                       non_survivors=f"{np.median(b):.2f} ({np.percentile(b,25):.2f}–{np.percentile(b,75):.2f})",
                       P=float(mannwhitneyu(a, b, alternative="two-sided").pvalue)))
    pd.DataFrame(t1).to_csv(f"{args.out}/table1.csv", index=False)

    # ---- artefacts for the figure script ----
    np.savez_compressed(
        f"{args.out}/artefacts.npz",
        y_te=y_te, y_ca=y_ca, y_dv=y_dv,
        p_te=np.array([pred[(PRIM_S, m)]["p_te"] for m in MODELS]),
        p_iso=p_iso, p_plt=p_plt, p_raw=pred[(PRIM_S, PRIM_M)]["p_te"],
        Z_imp=Z_d, shap=contrib,
        shap_names=np.array([label(preds[c]) for c in cols]),
        corr=np.corrcoef(Z_d.T), labels=np.array([label(p) for p in preds]),
        grades=np.array([r["grade"] for r in t6]), grade_names=np.array([r["variable"] for r in t6]),
        thresholds=np.linspace(0.05, 0.60, 56),
    )

    summary = dict(cohort=len(y), deaths=int(y.sum()), n_pred=len(preds),
                   n_dev=len(idx_dv), n_cal=len(idx_ca), n_te=len(idx_te),
                   deaths_dev=int(y_dv.sum()), deaths_cal=int(y_ca.sum()), deaths_te=int(y_te.sum()),
                   youden=youden,
                   oof={f"{s}|{m}": oof[(s, m)] for s in STRATEGIES for m in MODELS},
                   test={f"{s}|{m}": test_auc[(s, m)] for s in STRATEGIES for m in MODELS},
                   table4=t4, table5={k: v for k, v in t5.items()},
                   table6=t6, delong=rows,
                   shap_top=[dict(variable=label(preds[cols[j]]), mean_abs=float(shap_mean[j])) for j in top])
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=float)
    with open(f"{args.out}/state.pkl", "wb") as f:
        pickle.dump(dict(pred=pred, idx_dv=idx_dv, idx_ca=idx_ca, idx_te=idx_te, y=y, X=X, preds=preds), f)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
