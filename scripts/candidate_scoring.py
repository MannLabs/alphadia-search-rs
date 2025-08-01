#!/usr/bin/env python3

from alphadia_ng import SpecLibFlat, PeakGroupScoring, DIADataNextGen, ScoringParameters, CandidateCollection
import os
import pandas as pd
import numpy as np
import logging
import time
import argparse
from alpharaw.ms_data_base import MSData_Base
from alphabase.spectral_library.flat import SpecLibFlat as AlphaBaseSpecLibFlat

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

    logger.info(f"Loaded {len(candidates_df)} candidates")

    # Filter top N candidates by highest score if specified
    if top_n is not None:
        candidates_df = candidates_df.nlargest(top_n, 'score')
        logger.info(f"Filtered to top {len(candidates_df)} candidates by score")

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
        alpha_base_spec_lib_flat.fragment_df['intensity'].values.astype(np.float32)
    )

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
        Scored candidates DataFrame
    """
    rs_data_next_gen = create_dia_data_next_gen(ms_data)


    spec_lib_flat = create_spec_lib_flat(alpha_base_spec_lib_flat)

    # scoring will be done in the next step

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
                       default=100000,
                       help="Top N candidates to score")
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

    # Run scoring
    run_candidate_scoring(ms_data, spec_lib_flat, candidates)

if __name__ == "__main__":
    main()