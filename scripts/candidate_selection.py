#!/usr/bin/env python3
"""Script for selecting peptide candidates from MS data and spectral libraries."""

from alphadia_search_rs import (
    SpecLibFlat,
    PeakGroupSelection,
    DIAData,
    SelectionParameters,
)
import pandas as pd
import numpy as np
import logging
import time
import argparse
from alpharaw.ms_data_base import MSData_Base
from alphabase.spectral_library.flat import SpecLibFlat as AlphaBaseSpecLibFlat

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_candidate_selection(ms_data, alphabase_speclib_flat):
    """
    Run candidate selection using alphaRaw MSData_Base object and SpecLibFlat.

    Parameters
    ----------
    ms_data : MSData_Base
        AlphaRaw MSData_Base object containing spectrum data
    spec_lib_flat : SpecLibFlat
        Spectral library in flat format

    Returns
    -------
    candidates
        Search results from candidate selection
    """
    logger = logging.getLogger(__name__)

    logger.info("Creating DIAData from MSData_Base")

    # Prepare arrays for DIAData
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
    start_time = time.perf_counter()
    rs_data_next_gen = DIAData.from_arrays(*spectrum_arrays, *peak_arrays)
    end_time = time.perf_counter()
    creation_time = end_time - start_time
    logger.info(f"DIAData creation time: {creation_time:.4f} seconds")

    logger.info("Setting up scoring parameters")
    selection_params = SelectionParameters()

    logger.info(
        f"Creating SpecLibFlat with {alphabase_speclib_flat.precursor_df.shape[0]} precursors"
    )

    spec_lib_flat = SpecLibFlat.from_arrays(
        alphabase_speclib_flat.precursor_df["precursor_idx"].values.astype(np.uint64),
        alphabase_speclib_flat.precursor_df["mz_calibrated"].values.astype(np.float32),
        alphabase_speclib_flat.precursor_df["rt_calibrated"].values.astype(np.float32),
        alphabase_speclib_flat.precursor_df["nAA"].values.astype(np.uint8),
        alphabase_speclib_flat.precursor_df["flat_frag_start_idx"].values.astype(
            np.uint64
        ),
        alphabase_speclib_flat.precursor_df["flat_frag_stop_idx"].values.astype(
            np.uint64
        ),
        alphabase_speclib_flat.fragment_df["mz_calibrated"].values.astype(np.float32),
        alphabase_speclib_flat.fragment_df["intensity"].values.astype(np.float32),
        alphabase_speclib_flat.fragment_df["cardinality"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["charge"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["loss_type"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["number"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["position"].values.astype(np.uint8),
        alphabase_speclib_flat.fragment_df["type"].values.astype(np.uint8),
    )

    # Default parameters
    config_dict = {
        "fwhm_rt": 3.0,
        "kernel_size": 20,
        "peak_length": 3,
        "mass_tolerance": 7.0,
        "rt_tolerance": 500.0,
        "candidate_count": 3,
    }
    selection_params.update(config_dict)

    logger.info(f"Using parameters: {config_dict}")

    # Create peak group selection
    peak_group_selection = PeakGroupSelection(selection_params)

    # Measure search time
    logger.info("Running candidate selection...")
    start_time = time.perf_counter()
    candidates = peak_group_selection.search(rs_data_next_gen, spec_lib_flat)
    end_time = time.perf_counter()
    search_time = end_time - start_time
    logger.info(f"Candidate selection time: {search_time:.4f} seconds")
    logger.info(f"Found {candidates.len()} candidates")

    return candidates


def parse_candidates(candidates, ms_data, alphabase_speclib_flat):
    """Parse candidates and create DataFrame with MS data and spectral library information."""
    result = candidates.to_arrays()

    precursor_idx = result[0]
    rank = result[1]
    score = result[2]
    scan_center = result[3]
    scan_start = result[4]
    scan_stop = result[5]
    frame_center = result[6]
    frame_start = result[7]
    frame_stop = result[8]

    candidates_df = pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "rank": rank,
            "score": score,
            "scan_center": scan_center,
            "scan_start": scan_start,
            "scan_stop": scan_stop,
            "frame_center": frame_center,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )

    candidates_df = candidates_df.merge(
        alphabase_speclib_flat.precursor_df[
            ["precursor_idx", "elution_group_idx", "decoy"]
        ],
        on="precursor_idx",
        how="left",
    )

    cycle_len = ms_data.spectrum_df["cycle_idx"].max() + 1

    candidates_df["frame_start"] = candidates_df["frame_start"] * cycle_len
    candidates_df["frame_stop"] = candidates_df["frame_stop"] * cycle_len
    candidates_df["frame_center"] = candidates_df["frame_center"] * cycle_len

    candidates_df["scan_start"] = 0
    candidates_df["scan_stop"] = 1
    candidates_df["scan_center"] = 0

    return candidates_df


def main():
    """Run candidate selection pipeline."""
    parser = argparse.ArgumentParser(
        description="Run candidate selection with alphaRaw MSData_Base and SpecLibFlat"
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
        "--output_folder",
        default="/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/ibrutinib",
        help="Path to the output folder",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    logger.info(f"Loading MS data from: {args.ms_data_path}")
    # Load MS data using alpharaw
    ms_data = MSData_Base()
    ms_data.load_hdf(args.ms_data_path)

    logger.info(f"Loading spectral library from: {args.spec_lib_path}")
    spec_lib_flat = AlphaBaseSpecLibFlat()
    spec_lib_flat.load_hdf(args.spec_lib_path)

    candidates = run_candidate_selection(ms_data, spec_lib_flat)
    candidates_df = parse_candidates(candidates, ms_data, spec_lib_flat)

    output_path = f"{args.output_folder}/candidates.parquet"

    logger.info(f"Saving candidates to: {output_path}")
    candidates_df.to_parquet(output_path)

    return


if __name__ == "__main__":
    main()
