#!/usr/bin/env python3
"""Correlate OptIForest performance with dataset characteristics.

Reads ``consolidated_summary.csv`` and reports Pearson and Spearman
correlations of AUC-ROC / AUC-PR against dataset size, dimensionality and
anomaly rate, plus scatter plots. Writes only under ``--out-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NEW_DATASETS = {"annthyroid", "fraud"}

FEATURES = {
    "n_samples": "Nº de amostras (log)",
    "n_features": "Nº de features (log)",
    "anomaly_rate_pct": "Taxa de anomalia (%)",
}
METRICS = {"mean_auc_roc": "AUC-ROC", "mean_auc_pr": "AUC-PR"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results2/analysis/consolidated_summary.csv")
    parser.add_argument("--out-dir", default="results2/analysis")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Log-scale the heavy-tailed size/dimension features for linear correlation.
    df["log_n_samples"] = np.log10(df["n_samples"])
    df["log_n_features"] = np.log10(df["n_features"])
    feat_cols = {
        "n_samples": "log_n_samples",
        "n_features": "log_n_features",
        "anomaly_rate_pct": "anomaly_rate_pct",
    }

    lines = ["dataset_char,metric,pearson_r,spearman_rho,n"]
    print(f"{'characteristic':<18}{'metric':<10}{'Pearson r':>11}{'Spearman':>11}")
    for feat, col in feat_cols.items():
        for metric in METRICS:
            sub = df[[col, metric]].dropna()
            r = sub[col].corr(sub[metric], method="pearson")
            rho = sub[col].corr(sub[metric], method="spearman")
            print(f"{feat:<18}{METRICS[metric]:<10}{r:>11.2f}{rho:>11.2f}")
            lines.append(f"{feat},{metric},{r:.3f},{rho:.3f},{len(sub)}")
    (out_dir / "correlations.csv").write_text("\n".join(lines) + "\n")

    # Scatter grid: rows = characteristics, cols = metrics.
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for i, (feat, flabel) in enumerate(FEATURES.items()):
        col = feat_cols[feat]
        for j, (metric, mlabel) in enumerate(METRICS.items()):
            ax = axes[i][j]
            sub = df.dropna(subset=[col, metric])
            colors = ["#d1495b" if n in NEW_DATASETS else "#30638e"
                      for n in sub["dataset"]]
            ax.scatter(sub[col], 100 * sub[metric], c=colors)
            for _, row in sub.iterrows():
                ax.annotate(row["dataset"], (row[col], 100 * row[metric]),
                            fontsize=6, alpha=0.7)
            # Trend line.
            if len(sub) > 1:
                z = np.polyfit(sub[col], 100 * sub[metric], 1)
                xs = np.linspace(sub[col].min(), sub[col].max(), 50)
                ax.plot(xs, np.polyval(z, xs), "--", color="gray", lw=1)
            r = sub[col].corr(100 * sub[metric], method="pearson")
            ax.set_xlabel(flabel)
            ax.set_ylabel(mlabel + " (%)")
            ax.set_title(f"{mlabel} vs {flabel}  (r={r:.2f})", fontsize=9)
    fig.suptitle("Desempenho vs. características do dataset "
                 "(vermelho = datasets novos)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(plots_dir / "correlations.png", dpi=150)
    plt.close(fig)

    # Subplots individuais (um arquivo por "quadradinho"), para empilhar
    # verticalmente em coluna unica no relatorio.
    short = {"n_samples": "samples", "n_features": "features",
             "anomaly_rate_pct": "anomaly"}
    for metric, mlabel in METRICS.items():
        mtag = "roc" if metric == "mean_auc_roc" else "pr"
        for feat, flabel in FEATURES.items():
            col = feat_cols[feat]
            fig, ax = plt.subplots(figsize=(5.6, 3.7))
            sub = df.dropna(subset=[col, metric])
            colors = ["#d1495b" if n in NEW_DATASETS else "#30638e"
                      for n in sub["dataset"]]
            ax.scatter(sub[col], 100 * sub[metric], c=colors, s=42)
            for _, row in sub.iterrows():
                ax.annotate(row["dataset"], (row[col], 100 * row[metric]),
                            fontsize=8, alpha=0.75)
            if len(sub) > 1:
                z = np.polyfit(sub[col], 100 * sub[metric], 1)
                xs = np.linspace(sub[col].min(), sub[col].max(), 50)
                ax.plot(xs, np.polyval(z, xs), "--", color="gray", lw=1.2)
            r = sub[col].corr(100 * sub[metric], method="pearson")
            ax.set_xlabel(flabel, fontsize=11)
            ax.set_ylabel(mlabel + " (%)", fontsize=11)
            ax.set_title(f"{mlabel} vs {flabel}  (r={r:.2f})", fontsize=12)
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            fig.savefig(plots_dir / f"corr_{mtag}_{short[feat]}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"\n  -> {out_dir / 'correlations.csv'}")
    print(f"  -> {plots_dir / 'correlations.png'}")
    print(f"  -> {plots_dir / 'corr_{{roc,pr}}_{{samples,features,anomaly}}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
