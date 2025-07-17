#!/usr/bin/env python3

import numpy as np
import argparse
from alpharaw.ms_data_base import MSData_Base
import alpha_rs
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def process_raw_file(hdf_file_path, cycle_len=301):
    """
    Process a raw file and optionally save as HDF.

    Args:
        raw_file_path: Path to the raw file
        hdf_file_path: Optional path to save HDF file
    """
    logger.info(f"Processing HDF file: {hdf_file_path}")
    # Load the raw data
    dia_data = MSData_Base()
    dia_data.load_hdf(hdf_file_path)

    # Calculate cycle indices
    delta_scan_idx = np.tile(np.arange(cycle_len), int(len(dia_data.spectrum_df) / cycle_len + 1))
    cycle_idx = np.repeat(np.arange(int(len(dia_data.spectrum_df) / cycle_len + 1)), cycle_len)

    # Add indices to spectrum dataframe
    dia_data.spectrum_df['delta_scan_idx'] = delta_scan_idx[:len(dia_data.spectrum_df)]
    dia_data.spectrum_df['cycle_idx'] = cycle_idx[:len(dia_data.spectrum_df)]

    logger.info(f"Processing with alpha_rs")
    # Process with alpha_rs
    hist = alpha_rs.test_xic_index(
        dia_data.spectrum_df['delta_scan_idx'].values,
        dia_data.spectrum_df['isolation_lower_mz'].values.astype(np.float32),
        dia_data.spectrum_df['isolation_upper_mz'].values.astype(np.float32),
        dia_data.spectrum_df['peak_start_idx'].values,
        dia_data.spectrum_df['peak_stop_idx'].values,
        dia_data.spectrum_df['cycle_idx'].values,
        dia_data.peak_df['mz'].values.astype(np.float32),
        dia_data.peak_df['intensity'].values.astype(np.float32)
    )

    logger.info(f"Processing complete")
    return dia_data

def main():
    parser = argparse.ArgumentParser(description='Process a raw file and generate XIC index histogram')
    parser.add_argument('hdf_file', help='Path to the HDF file')
    parser.add_argument('--cycle_len', type=int, default=301, help='Cycle length')

    args = parser.parse_args()

    process_raw_file(args.hdf_file, args.cycle_len)

if __name__ == "__main__":
    main()
