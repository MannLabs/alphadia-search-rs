from alpha_rs import SpecLibFlat, PeakGroupScoring, DIAData, ScoringParameters
import os
import pandas as pd
import numpy as np
import tempfile
import logging
from alphabase.tools.data_downloader import DataShareDownloader

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    # Use preferred folder if it exists, otherwise create temp directory
    preferred_folder = "/Users/georgwallmann/Documents/data/alpha-rs"
    if os.path.exists(preferred_folder):
        tmp_folder = preferred_folder
    else:
        tmp_folder = tempfile.mkdtemp()
    
    logger.info(f"Using folder: {tmp_folder}")
    
    # Download required files (DataShareDownloader skips if files already exist)
    logger.info("Ensuring test data is available...")
    data_url = "https://datashare.biochem.mpg.de/s/gxfAcvJO7Ja6H4V"
    required_files = ['spectrum_df.parquet', 'peak_df.parquet', 'precursor_df.parquet', 'fragment_df.parquet']
    
    for file_name in required_files:
        file_url = f"{data_url}/download?files={file_name}"
        DataShareDownloader(file_url, tmp_folder).download()

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

    logger.info("Creating peak group scoring with default parameters")
    # Using default parameters (no arguments needed)
    scoring_params = ScoringParameters()
    
    # Update parameters using dictionary
    config_dict = {
        'fwhm_rt': 3.0,
        'kernel_size': 20,
        'peak_length': 5,
        'mass_tolerance': 7.0,
        'rt_tolerance': 200.0
    }
    scoring_params.update(config_dict)
    
    # You can also update parameters after creation
    update_dict = {'mass_tolerance': 10.0, 'rt_tolerance': 150.0}
    scoring_params.update(update_dict)
    
    logger.info(f"Using parameters: fwhm_rt={scoring_params.fwhm_rt}, "
                f"kernel_size={scoring_params.kernel_size}, "
                f"peak_length={scoring_params.peak_length}, "
                f"mass_tolerance={scoring_params.mass_tolerance}, "
                f"rt_tolerance={scoring_params.rt_tolerance}")
    
    # Using the parameters object
    peak_group_scoring = PeakGroupScoring(scoring_params)

    logger.info("Searching")    
    candidates = peak_group_scoring.search(rs_data, speclib)

    logger.info(f"Found {candidates.len()} candidates")
