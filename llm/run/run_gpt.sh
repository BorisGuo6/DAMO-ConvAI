#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LLM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$LLM_DIR")"

eval_path="${PROJECT_ROOT}/data/dev.json"
db_root_path="${PROJECT_ROOT}/data/dev_databases/"
use_knowledge='True'
not_use_knowledge='False'
mode='dev'
cot='True'
no_cot='False'

# Use OPENAI_API_KEY from environment variable
YOUR_API_KEY="${OPENAI_API_KEY}"

if [ -z "$YOUR_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

engine='gpt-3.5-turbo'

data_output_path="${LLM_DIR}/exp_result/turbo_output/"
data_kg_output_path="${LLM_DIR}/exp_result/turbo_output_kg/"

# Number of samples to process (set to 0 for all samples, or a small number for testing)
NUM_SAMPLES=${NUM_SAMPLES:-5}

echo "Project root: ${PROJECT_ROOT}"
echo "Eval path: ${eval_path}"
echo "DB root path: ${db_root_path}"
echo "Processing ${NUM_SAMPLES} samples (set NUM_SAMPLES=0 to process all)"

echo 'generate GPT3.5 batch with knowledge'
python3 -u "${LLM_DIR}/src/gpt_request.py" --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine} --eval_path ${eval_path} --data_output_path ${data_kg_output_path} --use_knowledge ${use_knowledge} \
--chain_of_thought ${no_cot} --num_samples ${NUM_SAMPLES}

echo 'generate GPT3.5 batch without knowledge'
python3 -u "${LLM_DIR}/src/gpt_request.py" --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine} --eval_path ${eval_path} --data_output_path ${data_output_path} --use_knowledge ${not_use_knowledge} \
--chain_of_thought ${no_cot} --num_samples ${NUM_SAMPLES}
