#!/usr/bin/env python3
"""Minimal SHAP analysis - generates only the complete summary plot with all features."""

import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run minimal SHAP analysis."""
    parser = argparse.ArgumentParser(description="Minimal SHAP analysis")
    parser.add_argument(
        "diagnosis_features", help="Path to diagnosis_features.parquet file"
    )
    parser.add_argument(
        "--output-file",
        default="shap_summary_plot_all_features.pdf",
        help="Output file path",
    )
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading diagnosis features from: {args.diagnosis_features}")
    df = pd.read_parquet(args.diagnosis_features)
    logger.info(f"Loaded {len(df):,} diagnosis features")

    # Prepare features - only numeric columns
    exclude_columns = [
        "decoy",
        "precursor_idx",
        "rank",
        "elution_group_idx",
        "proteins",
        "channel",
        "precursor_idx_rank",
    ]
    feature_columns = [
        col
        for col in df.columns
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    logger.info(f"Using {len(feature_columns)} features")

    X = df[feature_columns].values
    y = df["decoy"].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # Train Random Forest
    logger.info("Training Random Forest")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    logger.info(f"Test accuracy: {rf.score(X_test, y_test):.3f}")

    # Compute SHAP values
    logger.info("Computing SHAP values")
    explainer = shap.TreeExplainer(rf)
    max_samples = min(1000, len(X_test))
    X_shap = X_test[:max_samples]
    shap_values = explainer.shap_values(X_shap)

    # Handle binary classification format
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values_decoy = shap_values[:, :, 1]  # Extract decoy class values
    elif isinstance(shap_values, list):
        shap_values_decoy = shap_values[1]
    else:
        shap_values_decoy = shap_values

    logger.info(f"SHAP values shape: {shap_values_decoy.shape}")

    # Create complete SHAP summary plot
    logger.info("Creating SHAP summary plot")
    plt.figure(figsize=(14, 12))

    # Sort features by importance
    feature_importance = np.abs(shap_values_decoy).mean(axis=0)
    feature_order = np.argsort(feature_importance)[::-1]  # Descending order

    # Create plot with all features
    shap.summary_plot(
        shap_values_decoy[:, feature_order],
        X_shap[:, feature_order],
        feature_names=[feature_columns[i] for i in feature_order],
        show=False,
        max_display=len(feature_columns),
    )
    plt.title("SHAP Feature Impact - All Features")
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SHAP summary plot to: {args.output_file}")
    logger.info("Analysis completed!")


if __name__ == "__main__":
    main()
