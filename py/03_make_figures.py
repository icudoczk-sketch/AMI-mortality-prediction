#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_make_figures.py — regenerate Figs 2-6, S1 and S2 from the artefacts written by 03_run_analysis.py.

All panels are drawn at 7 in (or narrow) width and 9 pt text so that the exported PNGs satisfy the
PLOS ONE figure specifications (300-600 dpi, <=7.5 x 8.75 in).

Usage:
    python 03_make_figures.py --artefacts outputs/artefacts.npz --out outputs/figures
"""
import argparse, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
MODEL_NAMES = ["XGBoost", "Random Forest", "Logistic Regression", "SVM"]
COLORS = ["#C00000", "#2E75B6", "#548235", "#7030A0"]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def heatmap(corr, labels, path):
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4.6)
    ax.set_yticklabels(labels, fontsize=4.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r", fontsize=7); cb.ax.tick_params(labelsize=7)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefacts", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A = np.load(args.artefacts, allow_pickle=True)

    labels = list(A["labels"])
    heatmap(A["corr"], labels, f"{args.out}/Fig2_correlation_heatmap.png")
    heatmap(A["corr"], labels, f"{args.out}/S1_Fig_correlation_heatmap.png")

    # Fig 3: GRA grades
    g, gn = A["grades"], list(A["grade_names"])
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.barh(range(len(g)), g[::-1], color="#2E75B6", height=0.68)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(gn[::-1], fontsize=7)
    ax.set_xlim(0.75, max(g) + 0.01)
    ax.set_xlabel("Grey relational grade", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    save(fig, f"{args.out}/Fig3_GRA.png")

    # Fig 4: ROC
    y_te, P = A["y_te"], A["p_te"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for i, (name, c) in enumerate(zip(MODEL_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_te, P[i])
        ax.plot(fpr, tpr, color=c, lw=1.4, label=f"{name} (AUC {roc_auc_score(y_te, P[i]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.9)
    ax.set_xlabel("1 \u2212 Specificity", fontsize=9); ax.set_ylabel("Sensitivity", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.tick_params(labelsize=8)
    save(fig, f"{args.out}/Fig4_ROC_test.png")

    # Fig 5: calibration
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for p, lab, c in [(A["p_raw"], "Raw", "#888888"),
                      (A["p_iso"], "Isotonic", "#2E75B6"),
                      (A["p_plt"], "Platt", "#C00000")]:
        fr, tp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(tp, fr, "o-", ms=3.4, lw=1.2, color=c, label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.9)
    ax.set_xlabel("Predicted probability", fontsize=9); ax.set_ylabel("Observed proportion", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5); ax.tick_params(labelsize=8)
    save(fig, f"{args.out}/Fig5_calibration.png")

    # Fig 6: SHAP beeswarm
    import shap
    plt.figure(figsize=(6.4, 4.8))
    shap.summary_plot(A["shap"], A["Z_imp"][:, :A["shap"].shape[1]],
                      feature_names=list(A["shap_names"]), max_display=15, show=False,
                      plot_size=(6.4, 4.8))
    fig = plt.gcf(); fig.set_size_inches(6.4, 4.8)
    for a in fig.axes:
        a.tick_params(labelsize=7)
    save(fig, f"{args.out}/Fig6_SHAP.png")

    # S2 Fig: decision curve
    ths = A["thresholds"]; p = A["p_plt"]
    nb = []
    for t in ths:
        pr = p >= t
        nb.append(((pr) & (y_te == 1)).sum() / len(y_te) - ((pr) & (y_te == 0)).sum() / len(y_te) * t / (1 - t))
    prev = y_te.mean(); nb_all = prev - (1 - prev) * ths / (1 - ths)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(ths, nb, color="#C00000", lw=1.4, label="Platt-recalibrated model")
    ax.plot(ths, nb_all, color="grey", lw=1.0, ls="--", label="Treat all")
    ax.axhline(0, color="black", lw=0.8); ax.set_ylim(-0.05, 0.30)
    ax.set_xlabel("Threshold probability", fontsize=9); ax.set_ylabel("Net benefit", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5); ax.tick_params(labelsize=8)
    save(fig, f"{args.out}/S2_Fig_DCA_testset.png")


if __name__ == "__main__":
    main()
