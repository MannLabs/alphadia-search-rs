#!/usr/bin/env python3

from alphadia_ng import (
    SpecLibFlat,
    PeakGroupScoring,
    DIAData,
    ScoringParameters,
    CandidateCollection,
    PeakGroupQuantification,
    QuantificationParameters,
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
import alphadia_ng

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
]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def convert_delta_rt_to_abs(target_df, decoy_df, feature_columns):
    """
    Convert delta_rt feature to absolute value while keeping other features unchanged.

    Parameters
    ----------
    target_df : pd.DataFrame
        Target examples with features
    decoy_df : pd.DataFrame
        Decoy examples with features
    feature_columns : list
        List of feature column names to use

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Modified feature matrices for targets and decoys
    """
    # Copy dataframes to avoid modifying originals
    target_modified = target_df.copy()
    decoy_modified = decoy_df.copy()

    # Convert delta_rt to absolute value if it exists
    if "delta_rt" in feature_columns:
        target_modified["delta_rt"] = target_modified["delta_rt"].abs()
        decoy_modified["delta_rt"] = decoy_modified["delta_rt"].abs()

    # Extract feature matrices
    target_features = target_modified[feature_columns].fillna(0).values
    decoy_features = decoy_modified[feature_columns].fillna(0).values

    return target_features, decoy_features


def perform_svm_fdr(target_df, decoy_df, feature_columns, competitive=True):
    """
    SVM-based semi-supervised FDR calculation using Rust SVM implementation.

    Parameters
    ----------
    target_df : pd.DataFrame
        Target examples with features
    decoy_df : pd.DataFrame
        Decoy examples with features
    feature_columns : list
        List of feature column names to use
    competitive : bool
        Whether to use competitive target-decoy approach

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with SVM scores and q-values
    """
    logger.info("Running SVM-based semi-supervised FDR calculation")

    # Prepare feature matrices with delta_rt converted to absolute value
    target_features, decoy_features = convert_delta_rt_to_abs(
        target_df, decoy_df, feature_columns
    )

    # Add bias term (constant 1.0) to features
    target_features_with_bias = np.column_stack(
        [target_features, np.ones(len(target_features))]
    )
    decoy_features_with_bias = np.column_stack(
        [decoy_features, np.ones(len(decoy_features))]
    )

    # Combine features and create labels
    all_features = np.vstack([target_features_with_bias, decoy_features_with_bias])
    labels = np.hstack([np.ones(len(target_features)), -np.ones(len(decoy_features))])

    logger.info(
        f"Training SVM on {len(target_features)} targets and {len(decoy_features)} decoys"
    )
    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")

    # Train SVM using Rust implementation
    svm_result = alphadia_ng.train_svm_python_interface(
        features=all_features.tolist(),
        labels=labels.tolist(),
        lambda_reg=0.01,  # L2 regularization
        epsilon=1e-6,  # Convergence tolerance
        max_iter=100,  # Maximum iterations
    )

    _weights = np.array(svm_result["weights"])
    scores = np.array(svm_result["scores"])
    converged = svm_result["converged"]

    if not converged:
        logger.warning("SVM training did not converge within maximum iterations")
    else:
        logger.info("SVM training converged successfully")

    # Split scores back to targets and decoys
    target_scores = scores[: len(target_features)]
    decoy_scores = scores[len(target_features) :]

    # Calculate q-values using target-decoy competition
    logger.info("Calculating q-values using target-decoy FDR estimation")

    # Combine scores and labels for FDR calculation
    all_svm_scores = np.hstack([target_scores, decoy_scores])
    all_labels = np.hstack(
        [np.ones(len(target_scores)), np.zeros(len(decoy_scores))]
    )  # 1=target, 0=decoy

    q_values = alphadia_ng.compute_q_values_python_interface(
        scores=all_svm_scores.tolist(), labels=all_labels.tolist()
    )
    q_values = np.array(q_values)

    # Split q-values back to targets and decoys
    target_qvals = q_values[: len(target_scores)]
    decoy_qvals = q_values[len(target_scores) :]

    # Create result DataFrames
    target_result = target_df.copy()
    target_result["svm_score"] = target_scores
    target_result["qval"] = target_qvals

    decoy_result = decoy_df.copy()
    decoy_result["svm_score"] = decoy_scores
    decoy_result["qval"] = decoy_qvals

    # Combine results
    combined_result = pd.concat([target_result, decoy_result], ignore_index=True)

    logger.info("SVM FDR calculation completed")
    logger.info(f"Targets with q < 0.01: {np.sum(target_qvals < 0.01)}")
    logger.info(f"Targets with q < 0.05: {np.sum(target_qvals < 0.05)}")
    logger.info(f"Targets with q < 0.1: {np.sum(target_qvals < 0.1)}")
    logger.info(f"Decoys with q < 0.01: {np.sum(decoy_qvals < 0.01)}")
    logger.info(f"Decoys with q < 0.05: {np.sum(decoy_qvals < 0.05)}")
    logger.info(f"Decoys with q < 0.1: {np.sum(decoy_qvals < 0.1)}")

    return combined_result


def perform_svm_nn_hybrid_fdr(target_df, decoy_df, feature_columns):
    """
    Hybrid SVM-NN FDR calculation:
    1. Perform SVM-based FDR calculation
    2. Filter at 10% FDR to get unique elution_group_idx + rank combinations
    3. Filter original SVM results (100% FDR) to only include these combinations
    4. Apply NN-based FDR on the filtered results

    Parameters
    ----------
    target_df : pd.DataFrame
        Target examples with features
    decoy_df : pd.DataFrame
        Decoy examples with features
    feature_columns : list
        List of feature column names to use

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with final NN-based scores and q-values
    """
    logger.info("Running hybrid SVM-NN FDR calculation")

    # Step 1: Perform SVM-based FDR calculation
    logger.info("Step 1: Performing SVM-based FDR calculation")
    svm_results = perform_svm_fdr(
        target_df, decoy_df, feature_columns, competitive=True
    )

    logger.info(f"SVM results: {len(svm_results):,}")

    # Step 2: Filter at 10% FDR and get unique elution_group_idx + rank combinations
    logger.info(
        "Step 2: Filtering at 50% FDR and getting unique elution_group_idx + rank combinations"
    )

    # Filter SVM results to 10% FDR
    svm_10_fdr = svm_results[svm_results["qval"] <= 0.50].copy()
    logger.info(f"SVM results passing 10% FDR: {len(svm_10_fdr):,}")

    # Create elution_group_idx + rank combination column
    svm_10_fdr["elution_group_rank"] = (
        svm_10_fdr["elution_group_idx"].astype(str)
        + "_"
        + svm_10_fdr["rank"].astype(str)
    )

    # Get unique elution_group_idx + rank combinations that passed 10% FDR
    unique_combinations = svm_10_fdr["elution_group_rank"].unique()
    logger.info(
        f"Unique elution_group_idx + rank combinations at 10% FDR: {len(unique_combinations):,}"
    )

    # Step 3: Filter original SVM results (100% FDR) to only include these combinations
    logger.info("Step 3: Filtering original SVM results by unique combinations")

    # Create combination column in original results
    svm_results["elution_group_rank"] = (
        svm_results["elution_group_idx"].astype(str)
        + "_"
        + svm_results["rank"].astype(str)
    )

    # Filter to only combinations that passed 10% FDR
    filtered_df = svm_results[
        svm_results["elution_group_rank"].isin(unique_combinations)
    ].copy()

    # Remove the temporary combination column
    filtered_df = filtered_df.drop(columns=["elution_group_rank"])
    logger.info(
        f"Selected {len(filtered_df):,} candidates (elution_group_idx + rank combinations): "
        f"{len(filtered_df[filtered_df['decoy'] == 0]):,} targets, "
        f"{len(filtered_df[filtered_df['decoy'] == 1]):,} decoys"
    )

    # Step 4: Apply NN-based FDR on filtered results
    logger.info("Step 4: Applying neural network-based FDR on filtered results")

    if len(filtered_df) == 0:
        logger.warning(
            "No candidates remaining after filtering - returning empty results"
        )
        return pd.DataFrame()

    # Separate targets and decoys for NN FDR
    filtered_targets = filtered_df[filtered_df["decoy"] == 0].copy()
    filtered_decoys = filtered_df[filtered_df["decoy"] == 1].copy()

    if len(filtered_targets) == 0 or len(filtered_decoys) == 0:
        logger.warning("No targets or decoys remaining after filtering")
        return filtered_df

    # Apply neural network FDR
    from alphadia.fdr.fdr import perform_fdr
    from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching

    classifier = BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        batch_size=min(
            5000, len(filtered_df)
        ),  # Adjust batch size for smaller datasets
        learning_rate=0.001,
        epochs=10,
        experimental_hyperparameter_tuning=True,
    )

    final_results = perform_fdr(
        classifier,
        feature_columns,
        filtered_targets,
        filtered_decoys,
        competetive=True,
    )

    logger.info("Hybrid SVM-NN FDR calculation completed")
    logger.info(
        f"Final results with NN q < 0.01: {np.sum(final_results['qval'] < 0.01):,}"
    )
    logger.info(
        f"Final results with NN q < 0.05: {np.sum(final_results['qval'] < 0.05):,}"
    )
    logger.info(
        f"Final results with NN q < 0.1: {np.sum(final_results['qval'] < 0.1):,}"
    )

    return final_results


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

    spectrum_arrays = (
        ms_data.spectrum_df["delta_scan_idx"].values,
        ms_data.spectrum_df["isolation_lower_mz"].values.astype(np.float32),
        ms_data.spectrum_df["isolation_upper_mz"].values.astype(np.float32),
        ms_data.spectrum_df["peak_start_idx"].values,
        ms_data.spectrum_df["peak_stop_idx"].values,
        ms_data.spectrum_df["cycle_idx"].values,
        ms_data.spectrum_df["rt"].values.astype(np.float32) * 60.0,
    )
    peak_arrays = (
        ms_data.peak_df["mz"].values.astype(np.float32),
        ms_data.peak_df["intensity"].values.astype(np.float32),
    )

    # Create a dummy cycle array - this appears to be mobility data which is not available in this dataset
    cycle_len = ms_data.spectrum_df["delta_scan_idx"].max() + 1
    cycle_array = np.zeros((cycle_len, 1, 1, 1), dtype=np.float32)

    start_time = time.perf_counter()
    rs_data_next_gen = DIAData.from_arrays(*spectrum_arrays, *peak_arrays, cycle_array)
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
        SpecLibFlat object for alphadia-ng
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
                "charge",
                "sequence",
            ]
        ],
        on="precursor_idx",
        how="left",
    )

    return features_df


def run_fdr_filtering(
    psm_scored_df, candidates_df, output_folder, use_svm=False, use_svm_nn=False
):
    """
    Run FDR filtering on scored candidates using neural network, SVM, or hybrid SVM-NN method.

    Parameters
    ----------
    psm_scored_df : pd.DataFrame
        DataFrame with scored candidates including decoy column
    candidates_df : pd.DataFrame
        Original candidates DataFrame
    output_folder : str
        Path to output folder for saving FDR results
    use_svm : bool
        Whether to use SVM-based semi-supervised FDR method
    use_svm_nn : bool
        Whether to use hybrid SVM-NN method (SVM at 10% FDR, best per group, then NN FDR)

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        FDR-filtered PSMs with q-value <= 0.01 and corresponding candidates_filtered_df
    """
    method_name = (
        "Hybrid SVM-NN"
        if use_svm_nn
        else ("SVM-based" if use_svm else "Neural Network")
    )
    logger.info(f"Running FDR filtering ({method_name})")

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

    features_raw_output_path = os.path.join(
        output_folder, "candidate_features_raw.parquet"
    )

    psm_scored_df.to_parquet(features_raw_output_path)
    logger.info(f"Saved raw features to: {features_raw_output_path}")

    target_df = psm_scored_df[psm_scored_df["decoy"] == 0].copy()
    decoy_df = psm_scored_df[psm_scored_df["decoy"] == 1].copy()

    if use_svm_nn:
        # Use hybrid SVM-NN method
        psm_df = perform_svm_nn_hybrid_fdr(target_df, decoy_df, FEATURE_COLUMNS)
    elif use_svm:
        # Use SVM-based semi-supervised FDR calculation
        psm_df = perform_svm_fdr(target_df, decoy_df, FEATURE_COLUMNS, competitive=True)
    else:
        # Use neural network-based FDR calculation
        classifier = BinaryClassifierLegacyNewBatching(
            test_size=0.001,
            batch_size=5000,
            learning_rate=0.001,
            epochs=10,
            experimental_hyperparameter_tuning=True,
        )

        psm_df = perform_fdr(
            classifier,
            FEATURE_COLUMNS,
            target_df,
            decoy_df,
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
    """
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
    parser.add_argument(
        "--svm",
        action="store_true",
        help="Use SVM-based semi-supervised FDR method instead of neural network",
    )
    parser.add_argument(
        "--svm-nn",
        action="store_true",
        help="Use hybrid SVM-NN method: SVM at 10% FDR, best per elution group, then NN FDR",
    )
    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.svm and args.svm_nn:
        parser.error("--svm and --svm-nn are mutually exclusive options")

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
            psm_scored_df,
            candidates,
            args.output_folder,
            use_svm=args.svm,
            use_svm_nn=args.svm_nn,
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
