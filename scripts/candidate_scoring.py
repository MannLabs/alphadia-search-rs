#!/usr/bin/env python3
"""Script for scoring peptide candidates using MS data and spectral libraries."""

from alphadia_search_rs import (
    SpecLibFlat,
    PeakGroupScoring,
    DIAData,
    ScoringParameters,
    CandidateCollection,
    PeakGroupQuantification,
    QuantificationParameters,
    CandidateFeatureCollection,
)
import os
import pandas as pd
import numpy as np
import logging
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from alpharaw.ms_data_base import MSData_Base
from alphabase.spectral_library.flat import SpecLibFlat as AlphaBaseSpecLibFlat
from alphadia.fdr.fdr import perform_fdr
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching

FEATURE_COLUMNS = CandidateFeatureCollection.get_feature_names()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_candidates_from_parquet(candidates_path, top_n=None):
    """
    Load candidates from parquet file and return filtered DataFrame.

    Parameters
    ----------
    candidates_path : str
        Path to the candidates parquet file
    top_n : int, optional
        Number of top candidates by score to keep

    Returns
    -------
    pd.DataFrame
        Candidates loaded as DataFrame
    """
    logger.info(f"Loading candidates from: {candidates_path}")
    candidates_df = pd.read_parquet(candidates_path)

    logger.info(f"Loaded {len(candidates_df):,} candidates")

    # Filter top N candidates by highest score if specified
    if top_n is not None:
        candidates_df = candidates_df.nlargest(top_n, "score")
        logger.info(f"Filtered to top {len(candidates_df):,} candidates by score")

    # The function load_candidates_from_parquet returns a DataFrame, not a CandidateCollection
    return candidates_df


def create_dia_data_next_gen(ms_data):
    """
    Create DIAData from alpharaw MSData_Base object.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data

    Returns
    -------
    DIAData
        DIAData object created from the MS data
    """
    logger.info("Creating DIAData from MSData_Base")

    # Create a dummy cycle array - this appears to be mobility data which is not available in this dataset
    cycle_len = ms_data.spectrum_df["delta_scan_idx"].max() + 1
    cycle_array = np.zeros((cycle_len, 1, 1, 1), dtype=np.float32)

    start_time = time.perf_counter()
    rs_data_next_gen = DIAData.from_arrays(
        spectrum_delta_scan_idx=ms_data.spectrum_df["delta_scan_idx"].values,
        isolation_lower_mz=ms_data.spectrum_df["isolation_lower_mz"].values.astype(
            np.float32
        ),
        isolation_upper_mz=ms_data.spectrum_df["isolation_upper_mz"].values.astype(
            np.float32
        ),
        spectrum_peak_start_idx=ms_data.spectrum_df["peak_start_idx"].values,
        spectrum_peak_stop_idx=ms_data.spectrum_df["peak_stop_idx"].values,
        spectrum_cycle_idx=ms_data.spectrum_df["cycle_idx"].values,
        spectrum_rt=ms_data.spectrum_df["rt"].values.astype(np.float32) * 60.0,
        peak_mz=ms_data.peak_df["mz"].values.astype(np.float32),
        peak_intensity=ms_data.peak_df["intensity"].values.astype(np.float32),
        cycle=cycle_array,
    )
    end_time = time.perf_counter()
    creation_time = end_time - start_time
    logger.info(f"DIAData creation time: {creation_time:.4f} seconds")

    return rs_data_next_gen


def create_spec_lib_flat(alphabase_speclib_flat):
    """
    Create SpecLibFlat from alphabase SpecLibFlat object.

    Parameters
    ----------
    alphabase_speclib_flat : AlphaBaseSpecLibFlat
        Alphabase spectral library in flat format

    Returns
    -------
    SpecLibFlat
        SpecLibFlat object for alphadia-search-rs
    """
    logger.info("Creating SpecLibFlat from alphabase SpecLibFlat")

    spec_lib_flat = SpecLibFlat.from_arrays(
        alphabase_speclib_flat.precursor_df["precursor_idx"].values.astype(np.uint64),
        alphabase_speclib_flat.precursor_df["mz_library"].values.astype(np.float32),
        alphabase_speclib_flat.precursor_df["mz_calibrated"].values.astype(np.float32),
        alphabase_speclib_flat.precursor_df["rt_library"].values.astype(np.float32),
        alphabase_speclib_flat.precursor_df["rt_calibrated"].values.astype(np.float32),
        alphabase_speclib_flat.precursor_df["nAA"].values.astype(np.uint8),
        alphabase_speclib_flat.precursor_df["flat_frag_start_idx"].values.astype(
            np.uint64
        ),
        alphabase_speclib_flat.precursor_df["flat_frag_stop_idx"].values.astype(
            np.uint64
        ),
        alphabase_speclib_flat.fragment_df["mz_library"].values.astype(np.float32),
        alphabase_speclib_flat.fragment_df["mz_calibrated"].values.astype(np.float32),
        alphabase_speclib_flat.fragment_df["intensity"].values.astype(np.float32),
        alphabase_speclib_flat.fragment_df["cardinality"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["charge"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["loss_type"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["number"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["position"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["type"].values.astype(np.uint8),
    )

    return spec_lib_flat


def run_candidate_scoring(ms_data, alphabase_speclib_flat, candidates_df):
    """
    Run candidate scoring using alphaRaw MSData_Base object, SpecLibFlat, and candidates.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data
    alphabase_speclib_flat : SpecLibFlat
        Spectral library in flat format
    candidates_df : pd.DataFrame
        Candidates DataFrame to score

    Returns
    -------
    pd.DataFrame
        Scored candidates DataFrame with features
    """
    rs_data_next_gen = create_dia_data_next_gen(ms_data)
    spec_lib_flat = create_spec_lib_flat(alphabase_speclib_flat)

    cycle_len = ms_data.spectrum_df["cycle_idx"].max() + 1

    # Convert DataFrame to CandidateCollection
    candidates = CandidateCollection.from_arrays(
        candidates_df["precursor_idx"].values.astype(np.uint64),
        candidates_df["rank"].values.astype(np.uint64),
        candidates_df["score"].values.astype(np.float32),
        candidates_df["scan_center"].values.astype(np.uint64),
        candidates_df["scan_start"].values.astype(np.uint64),
        candidates_df["scan_stop"].values.astype(np.uint64),
        candidates_df["frame_center"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_start"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_stop"].values.astype(np.uint64) // cycle_len,
    )

    scoring_params = ScoringParameters()
    scoring_params.update(
        {
            "top_k_fragments": 99,
            "mass_tolerance": 7.0,
        }
    )

    peak_group_scoring = PeakGroupScoring(scoring_params)

    logger.info(f"Scoring {len(candidates_df):,} candidates")

    # Get candidate features
    candidate_features = peak_group_scoring.score(
        rs_data_next_gen, spec_lib_flat, candidates
    )

    # Convert features to dictionary of arrays
    features_dict = candidate_features.to_dict_arrays()

    # Create DataFrame from features
    features_df = pd.DataFrame(features_dict)

    features_df = features_df.merge(
        alphabase_speclib_flat.precursor_df[
            [
                "precursor_idx",
                "decoy",
                "elution_group_idx",
                "channel",
                "proteins",
                "rt_calibrated",
                "rt_library",
                "mz_library",
            ]
        ],
        on="precursor_idx",
        how="left",
    )

    return features_df


def run_fdr_filtering(psm_scored_df, candidates_df, output_folder):
    """
    Run FDR filtering on scored candidates.

    Parameters
    ----------
    psm_scored_df : pd.DataFrame
        DataFrame with scored candidates including decoy column
    candidates_df : pd.DataFrame
        Original candidates DataFrame
    output_folder : str
        Path to output folder for saving FDR results

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        FDR-filtered PSMs with q-value <= 0.01 and corresponding candidates_filtered_df
    """
    logger.info("Running FDR filtering")

    classifier = BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        batch_size=5000,
        learning_rate=0.001,
        epochs=10,
        experimental_hyperparameter_tuning=True,
    )

    logger.info(f"Performing NN based FDR with {len(FEATURE_COLUMNS)} features")

    # Create composite index for proper matching
    psm_scored_df["precursor_idx_rank"] = (
        psm_scored_df["precursor_idx"].astype(str)
        + "_"
        + psm_scored_df["rank"].astype(str)
    )
    candidates_df["precursor_idx_rank"] = (
        candidates_df["precursor_idx"].astype(str)
        + "_"
        + candidates_df["rank"].astype(str)
    )

    psm_df = perform_fdr(
        classifier,
        FEATURE_COLUMNS,
        psm_scored_df[psm_scored_df["decoy"] == 0].copy(),
        psm_scored_df[psm_scored_df["decoy"] == 1].copy(),
        competetive=True,
    )

    psm_df = psm_df[psm_df["qval"] <= 0.01]
    logger.info(f"After FDR filtering (q-value <= 0.01): {len(psm_df):,} PSMs")

    # Create candidates_filtered_df using precursor_idx_rank index
    candidates_filtered_df = candidates_df[
        candidates_df["precursor_idx_rank"].isin(psm_df["precursor_idx_rank"])
    ].copy()
    logger.info(
        f"Created candidates_filtered_df with {len(candidates_filtered_df):,} candidates that passed 1% FDR"
    )

    # Save FDR results
    fdr_output_path = os.path.join(output_folder, "candidate_features_fdr.parquet")
    psm_df.to_parquet(fdr_output_path)
    logger.info(f"Saved FDR-filtered features to: {fdr_output_path}")

    candidates_filtered_path = os.path.join(
        output_folder, "candidates_filtered.parquet"
    )
    candidates_filtered_df.to_parquet(candidates_filtered_path)
    logger.info(f"Saved candidates_filtered_df to: {candidates_filtered_path}")

    return psm_df, candidates_filtered_df


def get_diagnosis_features(psm_scored_df, psm_fdr_passed_df):
    """Get best scoring target and decoy for each unique elution group from FDR-filtered results.

    Get best scoring target and decoy for each unique elution group from FDR-filtered results.
    Uses the original psm_scored_df to get paired decoys, not just FDR-filtered decoys.

    Parameters
    ----------
    psm_scored_df : pd.DataFrame
        DataFrame with scored candidates including decoy column and elution_group_idx
    psm_fdr_passed_df : pd.DataFrame
        DataFrame with FDR-filtered results (q-value <= 0.01)

    Returns
    -------
    pd.DataFrame
        DataFrame with best scoring target and decoy for each unique elution group
    """
    logger.info("Getting diagnosis features for unique elution groups")

    # Get unique elution groups from FDR-filtered results
    unique_elution_groups = psm_fdr_passed_df["elution_group_idx"].unique()
    logger.info(
        f"Found {len(unique_elution_groups):,} unique elution groups with FDR < 0.01"
    )

    # For each unique elution group, get the best scoring target and decoy
    diagnosis_features_list = []

    for elution_group_idx in unique_elution_groups:
        # Get all candidates for this elution group from the original psm_scored_df (not FDR-filtered)
        group_candidates = psm_scored_df[
            psm_scored_df["elution_group_idx"] == elution_group_idx
        ]

        # Get best scoring target
        target_candidates = group_candidates[group_candidates["decoy"] == 0]
        if len(target_candidates) > 0:
            best_target = target_candidates.loc[target_candidates["score"].idxmax()]
            diagnosis_features_list.append(best_target)

        # Get best scoring decoy from the original psm_scored_df (paired decoy)
        decoy_candidates = group_candidates[group_candidates["decoy"] == 1]
        if len(decoy_candidates) > 0:
            best_decoy = decoy_candidates.loc[decoy_candidates["score"].idxmax()]
            diagnosis_features_list.append(best_decoy)

    diagnosis_features_df = pd.DataFrame(diagnosis_features_list)

    logger.info(
        f"Created diagnosis features DataFrame with {len(diagnosis_features_df):,} rows"
    )
    logger.info(
        f"Target candidates: {len(diagnosis_features_df[diagnosis_features_df['decoy'] == 0]):,}"
    )
    logger.info(
        f"Decoy candidates: {len(diagnosis_features_df[diagnosis_features_df['decoy'] == 1]):,}"
    )

    return diagnosis_features_df


def save_diagnosis_features(diagnosis_features_df, output_folder):
    """
    Save diagnosis features DataFrame to parquet file.

    Parameters
    ----------
    diagnosis_features_df : pd.DataFrame
        DataFrame with diagnosis features for targets and decoys
    output_folder : str
        Output folder path
    """
    output_path = os.path.join(output_folder, "diagnosis_features.parquet")
    diagnosis_features_df.to_parquet(output_path)
    logger.info(f"Saved diagnosis features to: {output_path}")


def plot_diagnosis_feature_histograms(diagnosis_features_df, output_folder):
    """
    Plot histograms of all features from diagnosis features DataFrame colored by decoy and target using seaborn.

    Parameters
    ----------
    diagnosis_features_df : pd.DataFrame
        DataFrame with best scoring target and decoy for each unique elution group
    output_folder : str
        Path to output folder for saving plots
    """
    logger.info("Creating diagnosis feature histograms using seaborn")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Define features to plot (excluding non-numeric columns)

    # Filter to only include columns that exist in the DataFrame
    available_features = [
        col for col in FEATURE_COLUMNS if col in diagnosis_features_df.columns
    ]

    if not available_features:
        logger.warning("No feature columns found in DataFrame")
        return

    logger.info(f"Plotting histograms for {len(available_features)} features")

    # Calculate number of rows and columns for subplot layout
    n_features = len(available_features)
    n_cols = 4  # 4 columns
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle(
        "Diagnosis Feature Distributions: Target vs Decoy (Best per Elution Group)",
        fontsize=16,
        fontweight="bold",
    )

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Plot each feature
    for idx, feature in enumerate(available_features):
        ax = axes[idx]

        # Filter data for this feature (remove NaN values)
        feature_data = diagnosis_features_df[["decoy", feature]].dropna()

        if len(feature_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(f"{feature}", fontweight="bold")
            continue

        # Create a long-form DataFrame for seaborn
        plot_data = feature_data.copy()
        plot_data["Type"] = plot_data["decoy"].map({0: "Target", 1: "Decoy"})

        # Calculate shared bins across all data for this feature
        all_values = plot_data[feature].values
        bins = np.linspace(all_values.min(), all_values.max(), 51)  # 50 bins

        # Plot histograms using seaborn with shared bins
        sns.histplot(
            data=plot_data,
            x=feature,
            hue="Type",
            bins=bins,
            stat="density",
            alpha=0.7,
            ax=ax,
            palette={"Target": "blue", "Decoy": "red"},
        )

        # Customize plot
        ax.set_title(f"{feature}", fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        # Add statistics text
        target_data = plot_data[plot_data["Type"] == "Target"][feature]
        decoy_data = plot_data[plot_data["Type"] == "Decoy"][feature]

        target_mean = target_data.mean() if len(target_data) > 0 else 0
        decoy_mean = decoy_data.mean() if len(decoy_data) > 0 else 0
        target_count = len(target_data)
        decoy_count = len(decoy_data)
        stats_text = f"Target: {target_count} (mean: {target_mean:.3f})\nDecoy: {decoy_count} (mean: {decoy_mean:.3f})"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=8,
        )

    # Hide empty subplots
    for idx in range(len(available_features), len(axes)):
        axes[idx].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_folder, "diagnosis_feature_histograms.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved diagnosis feature histograms to: {plot_path}")

    # Close the figure to free memory
    plt.close()


def run_peak_group_quantification(ms_data, spec_lib_flat, candidates_filtered_df):
    """
    Run peak group quantification on FDR-filtered candidates.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data
    spec_lib_flat : AlphaBaseSpecLibFlat
        Alphabase spectral library in flat format
    candidates_filtered_df : pd.DataFrame
        FDR-filtered candidates DataFrame

    Returns
    -------
    SpecLibFlatQuantified
        Quantified spectral library
    """
    logger.info(
        f"Running peak group quantification on {len(candidates_filtered_df):,} FDR-filtered candidates"
    )

    # Create DIAData and SpecLibFlat objects
    rs_data_next_gen = create_dia_data_next_gen(ms_data)
    spec_lib_ng = create_spec_lib_flat(spec_lib_flat)

    # Convert DataFrame to CandidateCollection
    cycle_len = ms_data.spectrum_df["cycle_idx"].max() + 1

    candidates_collection = CandidateCollection.from_arrays(
        candidates_filtered_df["precursor_idx"].values.astype(np.uint64),
        candidates_filtered_df["rank"].values.astype(np.uint64),
        candidates_filtered_df["score"].values.astype(np.float32),
        candidates_filtered_df["scan_center"].values.astype(np.uint64),
        candidates_filtered_df["scan_start"].values.astype(np.uint64),
        candidates_filtered_df["scan_stop"].values.astype(np.uint64),
        candidates_filtered_df["frame_center"].values.astype(np.uint64) // cycle_len,
        candidates_filtered_df["frame_start"].values.astype(np.uint64) // cycle_len,
        candidates_filtered_df["frame_stop"].values.astype(np.uint64) // cycle_len,
    )

    # Create quantification parameters
    quant_params = QuantificationParameters()

    # Create and run quantification
    peak_group_quantification = PeakGroupQuantification(quant_params)
    quantified_lib = peak_group_quantification.quantify(
        rs_data_next_gen, spec_lib_ng, candidates_collection
    )

    # Convert quantified library to DataFrames using the new tuple structure
    precursor_dict, fragment_dict = quantified_lib.to_dict_arrays()
    precursor_df = pd.DataFrame(precursor_dict)
    fragment_df = pd.DataFrame(fragment_dict)

    logger.info(
        f"Created precursor_df with {len(precursor_df):,} rows and fragment_df with {len(fragment_df):,} rows"
    )

    return precursor_df, fragment_df


def main():
    """Run candidate scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Run candidate scoring with MS data, spectral library, and candidates"
    )
    parser.add_argument(
        "--ms_data_path",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib/CPD_NE_000057_08.hdf",
        help="Path to the MS data file (HDF format)",
    )
    parser.add_argument(
        "--spec_lib_path",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib/speclib_flat_calibrated.hdf",
        help="Path to the spectral library file (HDF format)",
    )
    parser.add_argument(
        "--candidates_path",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib/candidates.parquet",
        help="Path to the candidates file (parquet format)",
    )
    parser.add_argument(
        "--output_folder",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--top-n", type=int, default=10000, help="Top N candidates to score"
    )
    parser.add_argument(
        "--fdr", action="store_true", help="Run FDR filtering on scored candidates"
    )
    parser.add_argument(
        "--diagnosis",
        action="store_true",
        help="Generate diagnosis features (best target/decoy per elution group with FDR < 0.01)",
    )
    parser.add_argument(
        "--quantify",
        action="store_true",
        help="Run peak group quantification on FDR-filtered candidates",
    )
    args = parser.parse_args()

    logger.info(f"Loading MS data from: {args.ms_data_path}")
    # Load MS data using alpharaw
    ms_data = MSData_Base()
    ms_data.load_hdf(args.ms_data_path)

    logger.info(f"Loading spectral library from: {args.spec_lib_path}")
    spec_lib_flat = AlphaBaseSpecLibFlat()
    spec_lib_flat.load_hdf(args.spec_lib_path)

    # Load candidates
    candidates = load_candidates_from_parquet(args.candidates_path, args.top_n)

    # Run scoring and get features
    psm_scored_df = run_candidate_scoring(ms_data, spec_lib_flat, candidates)

    # Run FDR filtering
    psm_fdr_passed_df = None
    candidates_filtered_df = None
    if args.fdr or args.diagnosis or args.quantify:
        psm_fdr_passed_df, candidates_filtered_df = run_fdr_filtering(
            psm_scored_df, candidates, args.output_folder
        )

    # Generate diagnosis features if requested
    if args.diagnosis and psm_fdr_passed_df is not None:
        diagnosis_features_df = get_diagnosis_features(psm_scored_df, psm_fdr_passed_df)
        save_diagnosis_features(diagnosis_features_df, args.output_folder)
        plot_diagnosis_feature_histograms(diagnosis_features_df, args.output_folder)

    # Run peak group quantification if requested
    if args.quantify and candidates_filtered_df is not None:
        precursor_quantified_df, fragment_quantified_df = run_peak_group_quantification(
            ms_data, spec_lib_flat, candidates_filtered_df
        )
        logger.info(
            f"Peak group quantification completed for {len(candidates_filtered_df):,} candidates"
        )

        precursor_quantified_path = os.path.join(
            args.output_folder, "precursor_quantified_df.parquet"
        )
        fragment_quantified_path = os.path.join(
            args.output_folder, "fragment_quantified_df.parquet"
        )

        precursor_quantified_df.to_parquet(precursor_quantified_path)
        fragment_quantified_df.to_parquet(fragment_quantified_path)

    # Save results
    if psm_fdr_passed_df is not None:
        output_path = os.path.join(args.output_folder, "candidate_features.parquet")
        psm_fdr_passed_df.to_parquet(output_path)
        logger.info(
            f"Saved {len(psm_fdr_passed_df):,} candidate features to: {output_path}"
        )


if __name__ == "__main__":
    main()
