#!/usr/bin/env python3
"""Calculate precursors at 1% FDR for each feature sorted ascending and descending."""

import argparse
import pandas as pd
import numpy as np
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
    "num_over_0",
    "num_over_0_rank_0_5",
    "num_over_0_rank_6_11",
    "num_over_0_rank_12_17",
    "num_over_0_rank_18_23",
    "num_over_50_rank_0_5",
    "num_over_50_rank_6_11",
    "num_over_50_rank_12_17",
    "num_over_50_rank_18_23",
    "hyperscore_intensity_observation",
    "hyperscore_intensity_library",
    "hyperscore_inverse_mass_error",
    "rt_observed",
    "delta_rt",
    "longest_b_series",
    "longest_y_series",
    "naa",
    "weighted_mass_error",
    "log10_b_ion_intensity",
    "log10_y_ion_intensity",
    "idf_hyperscore",
    "idf_xic_dot_product",
    "idf_intensity_dot_product",
    "median_profile_sum",
    "median_profile_sum_filtered",
    "num_profiles",
    "num_profiles_filtered",
    "num_over_0_top6_idf",
    "num_over_50_top6_idf",
    "hyperscore_v1_sage",
    "hyperscore_v2_apex",
    "hyperscore_v3_corr",
    "hyperscore_v4_strict",
    "hyperscore_v5_combined",
    "xtandem_openms",
    "xtandem_count",
    "xtandem_intensity",
    "xtandem_normalized",
    "xtandem_consecutive",
    "n_b_strict",
    "n_y_strict",
    "match_fraction",
    "b_y_balance",
    "match_quality_ratio",
    "sequence_coverage",
    "matched_intensity_fraction",
    "intensity_corr_strict",
    "apex_intensity",
    "center_to_apex_offset",
    "n_scans_above_halfmax",
    "peak_symmetry",
    "peak_sharpness",
    "peak_concentration",
    "idf_weighted_count_strict",
    "idf_weighted_count_corr30",
    "idf_corr_weighted_strict",
    "n_mass_strict_3ppm",
    "n_mass_strict_5ppm",
    "idf_mass_strict_3ppm",
    "idf_mass_strict_5ppm",
    "idf_corr_mass_gaussian",
    "composite_mult",
    "idf_corr_top6",
]


def count_at_fdr(scores, is_target, fdr_threshold=0.01):
    """Count target precursors at a given FDR threshold.

    Assumes scores are already sorted descending (best first).

    Parameters
    ----------
    scores : np.ndarray
        Sorted scores (descending, best first)
    is_target : np.ndarray
        Boolean array, True for targets
    fdr_threshold : float
        FDR threshold

    Returns
    -------
    int
        Number of target precursors passing FDR
    """
    targets_cumsum = np.cumsum(is_target)
    decoys_cumsum = np.cumsum(~is_target)

    # FDR = decoys / targets (with protection against division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = decoys_cumsum / np.maximum(targets_cumsum, 1)

    # Find the last index where FDR <= threshold
    passing = fdr <= fdr_threshold
    if not np.any(passing):
        return 0

    last_passing_idx = np.where(passing)[0][-1]
    return int(targets_cumsum[last_passing_idx])


def main():
    parser = argparse.ArgumentParser(
        description="Calculate precursors at 1%% FDR for each feature"
    )
    parser.add_argument(
        "--features_path",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia_rust_fdr/quant/20231017_OA2_TiHe_ADIAMA_HeLa_200ng_Evo011_21min_F-40_05/features_pre_fdr.parquet",
    )
    parser.add_argument("--fdr", type=float, default=0.01)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.features_path)
    logger.info(f"Loaded {len(df):,} rows")

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    logger.info(f"Evaluating {len(available_features)} features")

    is_target = (df["decoy"] == 0).values

    results = []
    for feature in available_features:
        values = df[feature].values

        # Descending: higher values are better
        order_desc = np.argsort(-values, kind="stable")
        n_desc = count_at_fdr(values[order_desc], is_target[order_desc], args.fdr)

        # Ascending: lower values are better
        order_asc = np.argsort(values, kind="stable")
        n_asc = count_at_fdr(values[order_asc], is_target[order_asc], args.fdr)

        best = max(n_desc, n_asc)
        best_dir = "desc" if n_desc >= n_asc else "asc"

        results.append(
            {
                "feature": feature,
                "precursors_1pct_fdr_desc": n_desc,
                "precursors_1pct_fdr_asc": n_asc,
                "best": best,
                "best_direction": best_dir,
            }
        )
        logger.info(
            f"  {feature:40s}  desc={n_desc:>8,}  asc={n_asc:>8,}  best={best:>8,} ({best_dir})"
        )

    result_df = pd.DataFrame(results).sort_values("best", ascending=False)

    if args.output_path is None:
        output_path = args.features_path.replace(".parquet", "_fdr_ranking.tsv")
    else:
        output_path = args.output_path

    result_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved to {output_path}")

    print("\n" + result_df.to_string(index=False))


if __name__ == "__main__":
    main()
