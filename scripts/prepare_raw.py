#!/usr/bin/env python3
"""Script for processing raw MS files and saving as parquet files."""

import numpy as np
import argparse
import os
from alphadia.raw_data.alpharaw_wrapper import Thermo
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def process_raw_file(raw_file_path, output_folder):
    """
    Process a raw file and save as parquet files.

    Args:
        raw_file_path: Path to the raw file
        output_folder: Path to the output folder
    """
    logger.info(f"Processing raw file: {raw_file_path}")
    logger.info(f"Output folder: {output_folder}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the raw data
    dia_data = Thermo(raw_file_path)

    # Try to detect cycle length from spectrum data patterns
    # Look at the MS level pattern to find cycle boundaries
    ms_levels = dia_data.spectrum_df["ms_level"].values
    logger.info(f"MS levels present: {np.unique(ms_levels)}")

    cycle_len = dia_data.cycle.shape[1]

    logger.info(f"Detected cycle length: {cycle_len}")

    # Calculate cycle indices
    delta_scan_idx = np.tile(
        np.arange(cycle_len), int(len(dia_data.spectrum_df) / cycle_len + 1)
    )
    cycle_idx = np.repeat(
        np.arange(int(len(dia_data.spectrum_df) / cycle_len + 1)), cycle_len
    )

    # Add indices to spectrum dataframe
    dia_data.spectrum_df["delta_scan_idx"] = delta_scan_idx[: len(dia_data.spectrum_df)]
    dia_data.spectrum_df["cycle_idx"] = cycle_idx[: len(dia_data.spectrum_df)]

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(raw_file_path))[0]

    output_path = os.path.join(output_folder, f"{base_name}.hdf")
    logger.info(f"Saving HDF file to {output_path}")

    dia_data.save_hdf(output_path)

    return dia_data


def main():
    """Process raw files."""
    parser = argparse.ArgumentParser(
        description="Process a raw file and save as parquet files"
    )
    parser.add_argument("raw_file", help="Path to the input raw file")
    parser.add_argument("output_folder", help="Path to the output folder")

    args = parser.parse_args()

    process_raw_file(args.raw_file, args.output_folder)


if __name__ == "__main__":
    main()
