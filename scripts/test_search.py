from alpha_rs import SpecLibFlat, PeakGroupScoring, DIAData, DIADataNextGen, ScoringParameters
import os
import pandas as pd
import numpy as np
import tempfile
import logging
import time
import gc
from alphabase.tools.data_downloader import DataShareDownloader

logging.basicConfig(level=logging.INFO)

def measure_destruction_time(obj, name):
    """Measure time to destroy an object and free its memory"""
    start_time = time.perf_counter()
    del obj
    gc.collect()  # Force garbage collection
    end_time = time.perf_counter()
    destruction_time = end_time - start_time
    logging.info(f"{name} destruction time: {destruction_time:.4f} seconds")
    return destruction_time

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    # Use preferred folder if it exists, otherwise create temp directory
    preferred_folder = "/Users/mschwoerer/alpha-rsdata/alpha-rs"
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

    # Prepare arrays for both implementations
    spectrum_arrays = (
        spectrum_df['delta_scan_idx'].values,
        spectrum_df['isolation_lower_mz'].values.astype(np.float32),
        spectrum_df['isolation_upper_mz'].values.astype(np.float32),
        spectrum_df['peak_start_idx'].values,
        spectrum_df['peak_stop_idx'].values,
        spectrum_df['cycle_idx'].values,
        spectrum_df['rt'].values.astype(np.float32),
    )
    peak_arrays = (
        peak_df['mz'].values.astype(np.float32),
        peak_df['intensity'].values.astype(np.float32)
    )

    logger.info("Setting up scoring parameters")
    scoring_params = ScoringParameters()
    
    # Update parameters using dictionary
    config_dict = {
        'fwhm_rt': 3.0,
        'kernel_size': 20,
        'peak_length': 5,
        'mass_tolerance': 7.0,
        'rt_tolerance': 1000.0,
        'candidate_count': 3
    }
    scoring_params.update(config_dict)
    
    
    logger.info(f"Using parameters: fwhm_rt={scoring_params.fwhm_rt}, "
                f"kernel_size={scoring_params.kernel_size}, "
                f"peak_length={scoring_params.peak_length}, "
                f"mass_tolerance={scoring_params.mass_tolerance}, "
                f"rt_tolerance={scoring_params.rt_tolerance}, "
                f"candidate_count={scoring_params.candidate_count}")

    # =============================================================================
    # BENCHMARK DIADataNextGen
    # =============================================================================
    logger.info("=" * 60)
    logger.info("BENCHMARKING DIADataNextGen")
    logger.info("=" * 60)
    
    # Measure creation time
    logger.info("Creating DIADataNextGen...")
    start_time = time.perf_counter()
    rs_data_next_gen = DIADataNextGen.from_arrays(
        *spectrum_arrays,
        *peak_arrays
    )
    end_time = time.perf_counter()
    creation_time_next_gen = end_time - start_time
    logger.info(f"DIADataNextGen creation time: {creation_time_next_gen:.4f} seconds")

    # Log memory footprint
    memory_mb = rs_data_next_gen.memory_footprint_mb()
    memory_bytes = rs_data_next_gen.memory_footprint_bytes()
    logger.info(f"DIADataNextGen memory footprint: {memory_mb:.2f} MB ({memory_bytes:,} bytes)")
    logger.info(f"DIADataNextGen contains {rs_data_next_gen.num_observations} quadrupole observations")

    # Create peak group scoring
    peak_group_scoring = PeakGroupScoring(scoring_params)

    # Measure search time
    logger.info("Searching with DIADataNextGen...")
    start_time = time.perf_counter()
    candidates_next_gen = peak_group_scoring.search_next_gen(rs_data_next_gen, speclib)
    end_time = time.perf_counter()
    search_time_next_gen = end_time - start_time
    logger.info(f"DIADataNextGen search time: {search_time_next_gen:.4f} seconds")
    logger.info(f"Found {candidates_next_gen.len()} candidates with DIADataNextGen")

    # Measure destruction time
    destruction_time_next_gen = measure_destruction_time(rs_data_next_gen, "DIADataNextGen")
    destruction_time_candidates_next_gen = measure_destruction_time(candidates_next_gen, "DIADataNextGen candidates")

    # =============================================================================
    # BENCHMARK DIAData (Original)
    # =============================================================================
    logger.info("=" * 60)
    logger.info("BENCHMARKING DIAData (Original)")
    logger.info("=" * 60)
    
    # Measure creation time
    logger.info("Creating DIAData...")
    start_time = time.perf_counter()
    rs_data = DIAData.from_arrays(
        *spectrum_arrays,
        *peak_arrays
    )
    end_time = time.perf_counter()
    creation_time_original = end_time - start_time
    logger.info(f"DIAData creation time: {creation_time_original:.4f} seconds")

    # Log memory footprint
    memory_mb_original = rs_data.memory_footprint_mb()
    memory_bytes_original = rs_data.memory_footprint_bytes()
    logger.info(f"DIAData memory footprint: {memory_mb_original:.2f} MB ({memory_bytes_original:,} bytes)")

    # Measure search time
    logger.info("Searching with DIAData...")
    start_time = time.perf_counter()
    candidates_original = peak_group_scoring.search(rs_data, speclib)
    end_time = time.perf_counter()
    search_time_original = end_time - start_time
    logger.info(f"DIAData search time: {search_time_original:.4f} seconds")
    logger.info(f"Found {candidates_original.len()} candidates with DIAData")

    # Measure destruction time
    destruction_time_original = measure_destruction_time(rs_data, "DIAData")
    destruction_time_candidates_original = measure_destruction_time(candidates_original, "DIAData candidates")

    # =============================================================================
    # SUMMARY COMPARISON
    # =============================================================================
    logger.info("=" * 60)
    logger.info("PERFORMANCE COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Creation Time:")
    logger.info(f"  DIAData:        {creation_time_original:.4f}s")
    logger.info(f"  DIADataNextGen: {creation_time_next_gen:.4f}s")
    logger.info(f"  Speedup:        {creation_time_original/creation_time_next_gen:.2f}x")
    
    logger.info(f"Search Time:")
    logger.info(f"  DIAData:        {search_time_original:.4f}s")
    logger.info(f"  DIADataNextGen: {search_time_next_gen:.4f}s")
    logger.info(f"  Speedup:        {search_time_original/search_time_next_gen:.2f}x")
    
    logger.info(f"Destruction Time:")
    logger.info(f"  DIAData:        {destruction_time_original:.4f}s")
    logger.info(f"  DIADataNextGen: {destruction_time_next_gen:.4f}s")
    
    logger.info(f"Memory Usage:")
    logger.info(f"  DIAData:        {memory_mb_original:.2f} MB")
    logger.info(f"  DIADataNextGen: {memory_mb:.2f} MB")
    logger.info(f"  Memory ratio:   {memory_mb_original/memory_mb:.2f}x")
    
    logger.info(f"Results:")
    logger.info(f"  DIAData candidates:        {candidates_original.len()}")
    logger.info(f"  DIADataNextGen candidates: {candidates_next_gen.len()}")
