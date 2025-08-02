#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate alphadia-ng && maturin develop --release && python ./scripts/candidate_scoring.py \
    --ms_data_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/CPD_NE_000057_08.hdf \
    --spec_lib_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/speclib_flat_calibrated.hdf \
    --candidates_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/candidates.parquet \
    --output_folder /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib \
    --top-n 100000000