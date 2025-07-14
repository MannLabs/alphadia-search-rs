from alpha_rs import SpecLibFlat, PeakGroupScoring, DIAData
import os
import pandas as pd
import numpy as np
tmp_folder = "/Users/georgwallmann/Documents/data/alpha-rs"
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    logger.info("Loading data")
    spectrum_df = pd.read_parquet(os.path.join(tmp_folder, 'spectrum_df.parquet'))
    peak_df = pd.read_parquet(os.path.join(tmp_folder, 'peak_df.parquet'))

    precursor_df = pd.read_parquet(os.path.join(tmp_folder, 'precursor_df.parquet'))
    fragment_df = pd.read_parquet(os.path.join(tmp_folder, 'fragment_df.parquet'))

    logger.info("Creating spec lib")

    speclib = SpecLibFlat.from_arrays(
        precursor_df['precursor_idx'].values.astype(np.uint64),
        precursor_df['precursor_mz'].values.astype(np.float32),
        precursor_df['rt_pred'].values.astype(np.float32),
        precursor_df['flat_frag_start_idx'].values.astype(np.uint64),
        precursor_df['flat_frag_stop_idx'].values.astype(np.uint64),
        fragment_df['mz'].values.astype(np.float32),
        fragment_df['intensity'].values.astype(np.float32)
    )

    logger.info("Creating rs data")
    rs_data = DIAData.from_arrays(
        spectrum_df['delta_scan_idx'].values,
        spectrum_df['isolation_lower_mz'].values.astype(np.float32),
        spectrum_df['isolation_upper_mz'].values.astype(np.float32),
        spectrum_df['peak_start_idx'].values,
        spectrum_df['peak_stop_idx'].values,
        spectrum_df['cycle_idx'].values,
        spectrum_df['rt'].values.astype(np.float32),
        peak_df['mz'].values.astype(np.float32),
        peak_df['intensity'].values.astype(np.float32)
    )

    logger.info("Creating peak group scoring")
    fwhm_rt = 3
    kernel_size = 20
    peak_length = 5

    peak_group_scoring = PeakGroupScoring(fwhm_rt, kernel_size, peak_length)

    logger.info("Searching")    
    mass_tolerance = 7
    rt_tolerance = 200
    candidates = peak_group_scoring.search(rs_data, speclib, mass_tolerance, rt_tolerance)

    logger.info(f"Found {candidates.len()} candidates")
