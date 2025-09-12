#!/usr/bin/env python3

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_fdr_features(features_path):
    """Load FDR-filtered features from parquet file."""
    logger.info(f"Loading FDR features from: {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df):,} FDR-filtered features")
    return df


def prepare_features_for_shap(df, max_samples=5000):
    """Prepare features for SHAP analysis."""
    logger.info("Preparing features for SHAP analysis")

    # Get all feature columns (excluding metadata columns)
    metadata_cols = [
        "precursor_idx",
        "rank",
        "decoy",
        "elution_group_idx",
        "channel",
        "proteins",
        "rt_calibrated",
        "rt_library",
        "mz_library",
        "qval",
        "precursor_idx_rank",
    ]

    feature_cols = [col for col in df.columns if col not in metadata_cols]
    logger.info(f"Found {len(feature_cols)} feature columns")

    # Handle missing values
    X = df[feature_cols].fillna(0)
    y = df["decoy"].values  # 0 = target, 1 = decoy

    # Sample data if too large
    if len(X) > max_samples:
        logger.info(f"Sampling {max_samples} rows from {len(X)} total")
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X.iloc[idx]
        y = y[idx]

    return X, y, feature_cols


def run_shap_analysis(X, y, feature_cols, output_folder):
    """Run SHAP analysis on the features."""
    logger.info("Running SHAP analysis")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train random forest classifier
    logger.info("Training Random Forest classifier")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # Calculate accuracy
    train_acc = rf.score(X_train_scaled, y_train)
    test_acc = rf.score(X_test_scaled, y_test)
    logger.info(f"Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")

    # Create SHAP explainer
    logger.info("Creating SHAP explainer")
    explainer = shap.TreeExplainer(rf)

    # Calculate SHAP values for test set (limit to 1000 samples for speed)
    n_shap_samples = min(1000, len(X_test_scaled))
    X_shap = X_test_scaled[:n_shap_samples]
    logger.info(f"Calculating SHAP values for {n_shap_samples} samples")

    shap_values = explainer.shap_values(X_shap)

    # SHAP values for binary classification: [class_0_values, class_1_values]
    # We want class_1 (decoy) values to understand what makes something a decoy
    shap_values_decoy = shap_values[1]

    # Create summary plot
    logger.info("Creating SHAP summary plot")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_decoy,
        X_shap,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    summary_path = os.path.join(output_folder, "shap_summary_plot.pdf")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to: {summary_path}")

    # Create feature importance bar plot
    logger.info("Creating SHAP feature importance plot")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_decoy,
        X_shap,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    importance_path = os.path.join(output_folder, "shap_feature_importance.pdf")
    plt.savefig(importance_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP feature importance plot to: {importance_path}")

    # Calculate mean absolute SHAP values for ranking
    mean_shap_values = np.mean(np.abs(shap_values_decoy), axis=0)
    feature_importance_df = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap": mean_shap_values}
    ).sort_values("mean_abs_shap", ascending=False)

    # Save feature importance ranking
    importance_ranking_path = os.path.join(output_folder, "shap_feature_ranking.csv")
    feature_importance_df.to_csv(importance_ranking_path, index=False)
    logger.info(f"Saved SHAP feature ranking to: {importance_ranking_path}")

    # Log top 10 most important features
    logger.info("Top 10 most important features by SHAP values:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        logger.info(f"{i + 1:2d}. {row['feature']}: {row['mean_abs_shap']:.4f}")

    return feature_importance_df


def main():
    parser = argparse.ArgumentParser(
        description="Run minimal SHAP analysis on FDR features"
    )
    parser.add_argument(
        "--features_path",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/astral_lf/candidate_features_fdr.parquet",
        help="Path to FDR-filtered features parquet file",
    )
    parser.add_argument(
        "--output_folder",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/astral_lf",
        help="Output folder for SHAP plots and results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples to use for SHAP analysis",
    )

    args = parser.parse_args()

    # Load data
    df = load_fdr_features(args.features_path)

    # Prepare features
    X, y, feature_cols = prepare_features_for_shap(df, args.max_samples)

    # Run SHAP analysis
    _ = run_shap_analysis(X, y, feature_cols, args.output_folder)

    logger.info("SHAP analysis completed successfully")


if __name__ == "__main__":
    main()
