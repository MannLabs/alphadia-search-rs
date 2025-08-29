#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate alphadia-ng && maturin develop --release && python ./scripts/candidate_selection.py \
    --ms_data_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/CPS_CO_000009_01.hdf \
    --spec_lib_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/speclib_flat_calibrated_decoy.hdf \
    --output_folder /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate alphadia-ng && maturin develop --release && python ./scripts/candidate_scoring.py \
    --ms_data_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/CPS_CO_000009_01.hdf \
    --spec_lib_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/speclib_flat_calibrated_decoy.hdf \
    --candidates_path /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib/candidates.parquet \
    --output_folder /Users/georgwallmann/Documents/data/alphadia_performance_tests/output/alphadia-ng-scoring/ibrutinib \
    --top-n 100000000 --fdr --quantify --svm-nn