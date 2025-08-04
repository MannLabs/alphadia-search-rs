#!/usr/bin/env python3

from alphadia_ng import SpecLibFlat, PeakGroupScoring, DIADataNextGen, ScoringParameters, CandidateCollection
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
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
        candidates_df = candidates_df.nlargest(top_n, 'score')
        logger.info(f"Filtered to top {len(candidates_df):,} candidates by score")

    # The function load_candidates_from_parquet returns a DataFrame, not a CandidateCollection
    return candidates_df

def create_dia_data_next_gen(ms_data):
    """
    Create DIADataNextGen from alpharaw MSData_Base object.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data

    Returns
    -------
    DIADataNextGen
        DIADataNextGen object created from the MS data
    """
    logger.info("Creating DIADataNextGen from MSData_Base")

    spectrum_arrays = (
        ms_data.spectrum_df['delta_scan_idx'].values,
        ms_data.spectrum_df['isolation_lower_mz'].values.astype(np.float32),
        ms_data.spectrum_df['isolation_upper_mz'].values.astype(np.float32),
        ms_data.spectrum_df['peak_start_idx'].values,
        ms_data.spectrum_df['peak_stop_idx'].values,
        ms_data.spectrum_df['cycle_idx'].values,
        ms_data.spectrum_df['rt'].values.astype(np.float32),
    )
    peak_arrays = (
        ms_data.peak_df['mz'].values.astype(np.float32),
        ms_data.peak_df['intensity'].values.astype(np.float32)
    )

    start_time = time.perf_counter()
    rs_data_next_gen = DIADataNextGen.from_arrays(
        *spectrum_arrays,
        *peak_arrays
    )
    end_time = time.perf_counter()
    creation_time = end_time - start_time
    logger.info(f"DIADataNextGen creation time: {creation_time:.4f} seconds")

    return rs_data_next_gen

def create_spec_lib_flat(alpha_base_spec_lib_flat):
    """
    Create SpecLibFlat from alphabase SpecLibFlat object.

    Parameters
    ----------
    alpha_base_spec_lib_flat : AlphaBaseSpecLibFlat
        Alphabase spectral library in flat format

    Returns
    -------
    SpecLibFlat
        SpecLibFlat object for alphadia-ng
    """
    logger.info("Creating SpecLibFlat from alphabase SpecLibFlat")

    spec_lib_flat = SpecLibFlat.from_arrays(
        alpha_base_spec_lib_flat.precursor_df['precursor_idx'].values.astype(np.uint64),
        alpha_base_spec_lib_flat.precursor_df['mz_calibrated'].values.astype(np.float32),
        alpha_base_spec_lib_flat.precursor_df['rt_calibrated'].values.astype(np.float32),
        alpha_base_spec_lib_flat.precursor_df['flat_frag_start_idx'].values.astype(np.uint64),
        alpha_base_spec_lib_flat.precursor_df['flat_frag_stop_idx'].values.astype(np.uint64),
        alpha_base_spec_lib_flat.fragment_df['mz_calibrated'].values.astype(np.float32),
        alpha_base_spec_lib_flat.fragment_df['intensity'].values.astype(np.float32),
        alpha_base_spec_lib_flat.fragment_df['cardinality'].values.astype(np.uint8),
        alpha_base_spec_lib_flat.fragment_df['charge'].values.astype(np.uint8),
        alpha_base_spec_lib_flat.fragment_df['loss_type'].values.astype(np.uint8),
        alpha_base_spec_lib_flat.fragment_df['number'].values.astype(np.uint8),
        alpha_base_spec_lib_flat.fragment_df['position'].values.astype(np.uint8),
        alpha_base_spec_lib_flat.fragment_df['type'].values.astype(np.uint8)
    )

    plt.hist(alpha_base_spec_lib_flat.precursor_df['rt_calibrated'], bins=100)
    plt.savefig('rt_calibrated.pdf')
    plt.close()

    return spec_lib_flat

def run_candidate_scoring(ms_data, alpha_base_spec_lib_flat, candidates_df):
    """
    Run candidate scoring using alphaRaw MSData_Base object, SpecLibFlat, and candidates.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data
    alpha_base_spec_lib_flat : SpecLibFlat
        Spectral library in flat format
    candidates_df : pd.DataFrame
        Candidates DataFrame to score

    Returns
    -------
    pd.DataFrame
        Scored candidates DataFrame with features
    """
    rs_data_next_gen = create_dia_data_next_gen(ms_data)
    spec_lib_flat = create_spec_lib_flat(alpha_base_spec_lib_flat)

    cycle_len = ms_data.spectrum_df['cycle_idx'].max() + 1

    # Convert DataFrame to CandidateCollection
    candidates = CandidateCollection.from_arrays(
        candidates_df['precursor_idx'].values.astype(np.uint64),
        candidates_df['rank'].values.astype(np.uint64),
        candidates_df['score'].values.astype(np.float32),
        candidates_df['scan_center'].values.astype(np.uint64),
        candidates_df['scan_start'].values.astype(np.uint64),
        candidates_df['scan_stop'].values.astype(np.uint64),
        candidates_df['frame_center'].values.astype(np.uint64) // cycle_len,
        candidates_df['frame_start'].values.astype(np.uint64) // cycle_len,
        candidates_df['frame_stop'].values.astype(np.uint64) // cycle_len,
    )

    scoring_params = ScoringParameters()
    scoring_params.update({
        'top_k_fragments': 99,
        'mass_tolerance': 7.0,
    })

    peak_group_scoring = PeakGroupScoring(scoring_params)

    logger.info(f"Scoring {len(candidates_df):,} candidates")

    # Get candidate features
    candidate_features = peak_group_scoring.score(rs_data_next_gen, spec_lib_flat, candidates)

    # Convert features to dictionary of arrays
    features_dict = candidate_features.to_dict_arrays()

    # Create DataFrame from features
    features_df = pd.DataFrame(features_dict)


    features_df = features_df.merge(
        alpha_base_spec_lib_flat.precursor_df[['precursor_idx', 'decoy','elution_group_idx','channel']],
        on='precursor_idx',
        how='left'
    )

    return features_df

def run_fdr_filtering(result_df, output_folder):
    """
    Run FDR filtering on scored candidates.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame with scored candidates including decoy column
    output_folder : str
        Path to output folder for saving FDR results

    Returns
    -------
    pd.DataFrame
        FDR-filtered PSMs with q-value <= 0.01
    """
    logger.info("Running FDR filtering")

    classifier = BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        batch_size=5000,
        learning_rate=0.001,
        epochs=10,
        experimental_hyperparameter_tuning=True,
    )

    available_columns = ['score', 'mean_correlation',
           'median_correlation', 'correlation_std', 'intensity_correlation',
           'num_fragments', 'num_scans', 'num_over_95', 'num_over_90',
           'num_over_80', 'num_over_50', 'hyperscore_intensity_observation',
           'hyperscore_intensity_library', 'rt_observed', 'delta_rt']

    logger.info(f"Performing NN based FDR with {len(available_columns)} features")

    psm_df = perform_fdr(
        classifier,
        available_columns,
        result_df[result_df["decoy"] == 0].copy(),
        result_df[result_df["decoy"] == 1].copy(),
        competetive=True,
    )

    psm_df = psm_df[psm_df["qval"] <= 0.01]
    logger.info(f"After FDR filtering (q-value <= 0.01): {len(psm_df):,} PSMs")

    # Save FDR results
    fdr_output_path = os.path.join(output_folder, "candidate_features_fdr.parquet")
    psm_df.to_parquet(fdr_output_path)
    logger.info(f"Saved FDR-filtered features to: {fdr_output_path}")

    return psm_df

def plot_feature_histograms(result_df, output_folder):
    """
    Plot histograms of all features colored by decoy and target, and save as PDF.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame with scored candidates including decoy column
    output_folder : str
        Path to output folder for saving plots
    """
    logger.info("Creating feature histograms")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Define features to plot (excluding non-numeric columns)
    feature_columns = ['score', 'mean_correlation', 'median_correlation', 'correlation_std',
                      'intensity_correlation', 'num_fragments', 'num_scans', 'num_over_95',
                      'num_over_90', 'num_over_80', 'num_over_50', 'hyperscore_intensity_observation',
                      'hyperscore_intensity_library', 'rt_observed', 'delta_rt']

    # Filter to only include columns that exist in the DataFrame
    available_features = [col for col in feature_columns if col in result_df.columns]

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
    fig.suptitle('Feature Distributions: Target vs Decoy', fontsize=16, fontweight='bold')

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Plot each feature
    for idx, feature in enumerate(available_features):
        ax = axes[idx]

        # Separate target and decoy data
        target_data = result_df[result_df['decoy'] == 0][feature]
        decoy_data = result_df[result_df['decoy'] == 1][feature]

        # Plot histograms
        if len(target_data) > 0:
            ax.hist(target_data, bins=50, alpha=0.7, label='Target', color='blue', density=True)
        if len(decoy_data) > 0:
            ax.hist(decoy_data, bins=50, alpha=0.7, label='Decoy', color='red', density=True)

        # Customize plot
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        target_mean = target_data.mean() if len(target_data) > 0 else 0
        decoy_mean = decoy_data.mean() if len(decoy_data) > 0 else 0
        stats_text = f'Target mean: {target_mean:.3f}\nDecoy mean: {decoy_mean:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)

    # Hide empty subplots
    for idx in range(len(available_features), len(axes)):
        axes[idx].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_folder, "feature_histograms.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature histograms to: {plot_path}")

    # Close the figure to free memory
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run candidate scoring with MS data, spectral library, and candidates")
    parser.add_argument("--ms_data_path",
                       default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib/CPD_NE_000057_08.hdf",
                       help="Path to the MS data file (HDF format)")
    parser.add_argument("--spec_lib_path",
                       default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib/speclib_flat_calibrated.hdf",
                       help="Path to the spectral library file (HDF format)")
    parser.add_argument("--candidates_path",
                       default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib/candidates.parquet",
                       help="Path to the candidates file (parquet format)")
    parser.add_argument("--output_folder",
                       default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib",
                       help="Path to the output folder")
    parser.add_argument("--top-n",
                       type=int,
                       default=10000,
                       help="Top N candidates to score")
    parser.add_argument("--fdr",
                       action="store_true",
                       help="Run FDR filtering on scored candidates")
    parser.add_argument("--plot",
                       action="store_true",
                       help="Generate feature histogram plots")
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
    result_df = run_candidate_scoring(ms_data, spec_lib_flat, candidates)

    # Generate plots if requested (before FDR filtering)
    if args.plot:
        plot_feature_histograms(result_df, args.output_folder)

    # Run FDR if requested
    if args.fdr:
        result_df = run_fdr_filtering(result_df, args.output_folder)

    # Save results
    output_path = os.path.join(args.output_folder, "candidate_features.parquet")
    result_df.to_parquet(output_path)
    logger.info(f"Saved {len(result_df):,} candidate features to: {output_path}")

if __name__ == "__main__":
    main()