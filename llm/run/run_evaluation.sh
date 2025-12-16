#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LLM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$LLM_DIR")"

db_root_path="${PROJECT_ROOT}/data/dev_databases/"
data_mode='dev'
diff_json_path="${PROJECT_ROOT}/data/dev.json"
predicted_sql_path_kg="${LLM_DIR}/exp_result/turbo_output_kg/"
predicted_sql_path="${LLM_DIR}/exp_result/turbo_output/"
ground_truth_path="${LLM_DIR}/data/"
num_cpus=4
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'

echo "Project root: ${PROJECT_ROOT}"
echo "DB root path: ${db_root_path}"
echo "Ground truth path: ${ground_truth_path}"

echo '''starting to compare with knowledge for ex'''
python3 -u "${LLM_DIR}/src/evaluation.py" --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare without knowledge for ex'''
python3 -u "${LLM_DIR}/src/evaluation.py" --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare with knowledge for ves'''
python3 -u "${LLM_DIR}/src/evaluation_ves.py" --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare without knowledge for ves'''
python3 -u "${LLM_DIR}/src/evaluation_ves.py" --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}
