#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate alphadia-ng && maturin develop --release && python ./scripts/candidate_scoring.py \
    --ms_data_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/astral_lf/20231017_OA2_TiHe_ADIAMA_HeLa_200ng_Evo011_21min_F-40_07.hdf \
    --spec_lib_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/astral_lf/speclib_flat_calibrated_decoy.hdf \
    --candidates_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/astral_lf/candidates.parquet \
    --output_folder /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/astral_lf \
    --top-n 100000000 --fdr --quantify --diagnosis