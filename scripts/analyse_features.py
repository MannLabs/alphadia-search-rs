#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "score",
    "mean_correlation",
    "median_correlation",
    "correlation_std",
    "intensity_correlation",
    "num_fragments",
    "num_scans",
    "num_over_95",
    "num_over_90",
    "num_over_80",
    "num_over_50",
]


def main():
    parser = argparse.ArgumentParser(
        description="Analyse candidate features with target/decoy histograms"
    )
    parser.add_argument(
        "--features_path",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/candidate_features.parquet",
        help="Path to candidate_features.parquet",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save the figure (default: same dir as features, feature_histograms.pdf)",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=500_000,
        help="Subsample to N rows for plotting speed (default: 500000, 0 = all)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.features_path)
    logger.info(f"Loaded {len(df):,} rows from {args.features_path}")

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    logger.info(f"Features to plot: {available_features}")

    if "decoy" not in df.columns:
        raise ValueError("No 'decoy' column found in the features dataframe")

    df["label"] = df["decoy"].map({0: "target", 1: "decoy"})
    logger.info(
        f"Targets: {(df['decoy'] == 0).sum():,}, Decoys: {(df['decoy'] == 1).sum():,}"
    )

    if args.sample_n > 0 and len(df) > args.sample_n:
        df = df.sample(n=args.sample_n, random_state=42)
        logger.info(f"Subsampled to {len(df):,} rows")

    n_features = len(available_features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(available_features):
        ax = axes[i]
        sns.histplot(
            data=df,
            x=feature,
            hue="label",
            hue_order=["target", "decoy"],
            stat="count",
            bins=100,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("")

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()

    if args.output_path is None:
        output_path = args.features_path.replace(".parquet", "_histograms.pdf")
    else:
        output_path = args.output_path

    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved figure to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
