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

def load_candidates_from_parquet(candidates_path):
    """
    Load candidates from parquet file and convert to CandidateCollection.

    Parameters
    ----------
    candidates_path : str
        Path to the candidates parquet file

    Returns
    -------
    CandidateCollection
        Candidates loaded into CandidateCollection object
    """
    logger.info(f"Loading candidates from: {candidates_path}")
    candidates_df = pd.read_parquet(candidates_path)

    logger.info(f"Loaded {len(candidates_df)} candidates")

    # Convert DataFrame to arrays for CandidateCollection.from_arrays
    precursor_idxs = candidates_df['precursor_idx'].values.astype(np.uint64)
    ranks = candidates_df['rank'].values.astype(np.uint64)
    scores = candidates_df['score'].values.astype(np.float32)
    scan_center = candidates_df['scan_center'].values.astype(np.uint64)
    scan_start = candidates_df['scan_start'].values.astype(np.uint64)
    scan_stop = candidates_df['scan_stop'].values.astype(np.uint64)
    frame_center = candidates_df['frame_center'].values.astype(np.uint64)
    frame_start = candidates_df['frame_start'].values.astype(np.uint64)
    frame_stop = candidates_df['frame_stop'].values.astype(np.uint64)

    # Create CandidateCollection from arrays
    candidates = CandidateCollection.from_arrays(
        precursor_idxs.tolist(),
        ranks.tolist(),
        scores.tolist(),
        scan_center.tolist(),
        scan_start.tolist(),
        scan_stop.tolist(),
        frame_center.tolist(),
        frame_start.tolist(),
        frame_stop.tolist()
    )

    return candidates

def run_candidate_scoring(ms_data, alpha_base_spec_lib_flat, candidates):
    """
    Run candidate scoring using alphaRaw MSData_Base object, SpecLibFlat, and candidates.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data
    alpha_base_spec_lib_flat : SpecLibFlat
        Spectral library in flat format
    candidates : CandidateCollection
        Candidate collection to score

    Returns
    -------
    CandidateCollection
        Scored candidates
    """
    logger.info("Creating DIADataNextGen from MSData_Base")

    # Prepare arrays for DIADataNextGen
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

    logger.info("Setting up scoring parameters")
    scoring_params = ScoringParameters()

    spec_lib_flat = SpecLibFlat.from_arrays(
        alpha_base_spec_lib_flat.precursor_df['precursor_idx'].values.astype(np.uint64),
        alpha_base_spec_lib_flat.precursor_df['mz_calibrated'].values.astype(np.float32),
        alpha_base_spec_lib_flat.precursor_df['rt_calibrated'].values.astype(np.float32),
        alpha_base_spec_lib_flat.precursor_df['flat_frag_start_idx'].values.astype(np.uint64),
        alpha_base_spec_lib_flat.precursor_df['flat_frag_stop_idx'].values.astype(np.uint64),
        alpha_base_spec_lib_flat.fragment_df['mz_calibrated'].values.astype(np.float32),
        alpha_base_spec_lib_flat.fragment_df['intensity'].values.astype(np.float32)
    )

    # Default parameters
    config_dict = {
        'fwhm_rt': 3.0,
        'kernel_size': 20,
        'peak_length': 5,
        'mass_tolerance': 7.0,
        'rt_tolerance': 200.0,
        'candidate_count': 5
    }
    scoring_params.update(config_dict)

    logger.info(f"Using parameters: {config_dict}")

    # Create peak group scoring
    peak_group_scoring = PeakGroupScoring(scoring_params)

    # Measure scoring time
    logger.info("Running candidate scoring...")
    start_time = time.perf_counter()
    scored_candidates = peak_group_scoring.score_candidates_next_gen(
        rs_data_next_gen,
        spec_lib_flat,
        candidates
    )
    end_time = time.perf_counter()
    scoring_time = end_time - start_time
    logger.info(f"Candidate scoring time: {scoring_time:.4f} seconds")
    logger.info(f"Scored {scored_candidates.len()} candidates")

    return scored_candidates

def parse_scored_candidates(scored_candidates, ms_data, alpha_base_spec_lib_flat):
    """
    Parse scored candidates back to DataFrame format.

    Parameters
    ----------
    scored_candidates : CandidateCollection
        Scored candidates from peak group scoring
    ms_data : MSData_Base
        AlphaRaw MSData_Base object
    alpha_base_spec_lib_flat : SpecLibFlat
        Spectral library in flat format

    Returns
    -------
    pd.DataFrame
        DataFrame containing scored candidates
    """
    result = scored_candidates.to_arrays()

    precursor_idx = result[0]
    rank = result[1]
    score = result[2]
    scan_center = result[3]
    scan_start = result[4]
    scan_stop = result[5]
    frame_center = result[6]
    frame_start = result[7]
    frame_stop = result[8]

    scored_candidates_df = pd.DataFrame({
        'precursor_idx': precursor_idx,
        'rank': rank,
        'score': score,
        'scan_center': scan_center,
        'scan_start': scan_start,
        'scan_stop': scan_stop,
        'frame_center': frame_center,
        'frame_start': frame_start,
        'frame_stop': frame_stop
    })

    # Merge with precursor information
    scored_candidates_df = scored_candidates_df.merge(
        alpha_base_spec_lib_flat.precursor_df[['precursor_idx', 'elution_group_idx', 'decoy']],
        on='precursor_idx',
        how='left'
    )

    # Convert frame indices back to cycle indices
    cycle_len = ms_data.spectrum_df['cycle_idx'].max() + 1
    scored_candidates_df['cycle_start'] = scored_candidates_df['frame_start'] // cycle_len
    scored_candidates_df['cycle_stop'] = scored_candidates_df['frame_stop'] // cycle_len
    scored_candidates_df['cycle_center'] = scored_candidates_df['frame_center'] // cycle_len

    return scored_candidates_df

def main():
    parser = argparse.ArgumentParser(description="Run candidate scoring with MS data, spectral library, and candidates")
    parser.add_argument("ms_data_path", help="Path to the MS data file (HDF format)")
    parser.add_argument("spec_lib_path", help="Path to the spectral library file (HDF format)")
    parser.add_argument("candidates_path", help="Path to the candidates file (parquet format)")
    parser.add_argument("output_folder", help="Path to the output folder")
    args = parser.parse_args()

    logger.info(f"Loading MS data from: {args.ms_data_path}")
    # Load MS data using alpharaw
    ms_data = MSData_Base()
    ms_data.load_hdf(args.ms_data_path)

    logger.info(f"Loading spectral library from: {args.spec_lib_path}")
    spec_lib_flat = AlphaBaseSpecLibFlat()
    spec_lib_flat.load_hdf(args.spec_lib_path)

    # Load candidates
    candidates = load_candidates_from_parquet(args.candidates_path)

    # Run scoring
    scored_candidates = run_candidate_scoring(ms_data, spec_lib_flat, candidates)

if __name__ == "__main__":
    main()