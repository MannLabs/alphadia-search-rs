"""Test script for peak group selection functionality."""

from alphadia_search_rs import (
    SpecLibFlat,
    PeakGroupSelection,
    DIAData,
    SelectionParameters,
)
import os
import pandas as pd
import numpy as np
import tempfile
import logging
import time
import argparse
from alphabase.tools.data_downloader import DataShareDownloader

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DIAData benchmark")
    parser.add_argument(
        "--path",
        type=str,
        default="/Users/georgwallmann/Documents/data/alphadia-search-rs",
        help="Path to data folder (default: /Users/georgwallmann/Documents/data/alphadia-search-rs)",
    )
    args = parser.parse_args()

    # Use provided path if it exists, otherwise create temp directory
    if os.path.exists(args.path):
        tmp_folder = args.path
    else:
        logger.warning(f"Path {args.path} does not exist, creating temporary directory")
        tmp_folder = tempfile.mkdtemp()

    logger.info(f"Using folder: {tmp_folder}")

    # Download required files (DataShareDownloader skips if files already exist)
    logger.info("Ensuring test data is available...")
    data_url = "https://datashare.biochem.mpg.de/s/gxfAcvJO7Ja6H4V"
    required_files = [
        "spectrum_df.parquet",
        "peak_df.parquet",
        "precursor_df.parquet",
        "fragment_df.parquet",
    ]

    for file_name in required_files:
        file_url = f"{data_url}/download?files={file_name}"
        DataShareDownloader(file_url, tmp_folder).download()

    logger.info("Loading data")
    spectrum_df = pd.read_parquet(os.path.join(tmp_folder, "spectrum_df.parquet"))
    peak_df = pd.read_parquet(os.path.join(tmp_folder, "peak_df.parquet"))

    precursor_df = pd.read_parquet(os.path.join(tmp_folder, "precursor_df.parquet"))
    fragment_df = pd.read_parquet(os.path.join(tmp_folder, "fragment_df.parquet"))

    logger.info("Creating spec lib")

    fragment_df["cardinality"] = 0  # TODO add this to fragment_df.parquet

    speclib = SpecLibFlat.from_arrays(
        precursor_df["precursor_idx"].values.astype(np.uint64),
        precursor_df["precursor_mz"].values.astype(np.float32),  # library
        precursor_df["precursor_mz"].values.astype(
            np.float32
        ),  # calibrated (same as library as this is the first round)
        precursor_df["rt_pred"].values.astype(np.float32),  # library
        precursor_df["rt_pred"].values.astype(np.float32),  # calibrated
        precursor_df["nAA"].values.astype(np.uint8),
        precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
        precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
        fragment_df["mz"].values.astype(np.float32),  # library
        fragment_df["mz"].values.astype(np.float32),  # calibrated
        fragment_df["intensity"].values.astype(np.float32),
        fragment_df["cardinality"].values.astype(np.uint8),
        fragment_df["charge"].values.astype(np.uint8),
        fragment_df["loss_type"].values.astype(np.uint8),
        fragment_df["number"].values.astype(np.uint8),
        fragment_df["position"].values.astype(np.uint8),
        fragment_df["type"].values.astype(np.uint8),
    )

    # Prepare arrays for DIAData
    spectrum_arrays = (
        spectrum_df["delta_scan_idx"].values,
        spectrum_df["isolation_lower_mz"].values.astype(np.float32),
        spectrum_df["isolation_upper_mz"].values.astype(np.float32),
        spectrum_df["peak_start_idx"].values,
        spectrum_df["peak_stop_idx"].values,
        spectrum_df["cycle_idx"].values,
        spectrum_df["rt"].values.astype(np.float32),
    )
    peak_arrays = (
        peak_df["mz"].values.astype(np.float32),
        peak_df["intensity"].values.astype(np.float32),
    )

    logger.info("Setting up selection parameters")
    selection_params = SelectionParameters()

    # Update parameters using dictionary
    config_dict = {
        "fwhm_rt": 3.0,
        "kernel_size": 20,
        "peak_length": 5,
        "mass_tolerance": 7.0,
        "rt_tolerance": 1000.0,
        "candidate_count": 3,
    }
    selection_params.update(config_dict)

    logger.info(f"Using parameters: {config_dict}")

    # =============================================================================
    # BENCHMARK DIAData
    # =============================================================================
    logger.info("=" * 60)
    logger.info("BENCHMARKING DIAData")
    logger.info("=" * 60)

    # Measure creation time
    logger.info("Creating DIAData...")
    start_time = time.perf_counter()

    cycle_len = spectrum_df["delta_scan_idx"].max() + 1
    cycle_array = np.zeros((cycle_len, 1, 1, 1), dtype=np.float32)

    rs_data_next_gen = DIAData.from_arrays(*spectrum_arrays, *peak_arrays, cycle_array)

    end_time = time.perf_counter()
    creation_time_next_gen = end_time - start_time
    logger.info(f"DIAData creation time: {creation_time_next_gen:.4f} seconds")

    # Log memory footprint
    memory_mb = rs_data_next_gen.memory_footprint_mb()
    memory_bytes = rs_data_next_gen.memory_footprint_bytes()
    logger.info(
        f"DIAData memory footprint: {memory_mb:.2f} MB ({memory_bytes:,} bytes)"
    )
    logger.info(
        f"DIAData contains {rs_data_next_gen.num_observations} quadrupole observations"
    )

    # Create peak group selection
    peak_group_selection = PeakGroupSelection(selection_params)

    # Measure search time
    logger.info("Searching with DIAData...")
    start_time = time.perf_counter()
    candidates_next_gen = peak_group_selection.search(rs_data_next_gen, speclib)
    end_time = time.perf_counter()
    search_time_next_gen = end_time - start_time
    logger.info(f"DIAData search time: {search_time_next_gen:.4f} seconds")
    logger.info(f"Found {candidates_next_gen.len()} candidates with DIAData")

    # =============================================================================
    # PERFORMANCE SUMMARY
    # =============================================================================
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Creation Time:     {creation_time_next_gen:.4f} seconds")
    logger.info(f"Search Time:       {search_time_next_gen:.4f} seconds")
    logger.info(f"Memory Usage:      {memory_mb:.2f} MB ({memory_bytes:,} bytes)")
    logger.info(f"Candidates Found:  {candidates_next_gen.len()}")
