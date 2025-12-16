#!/bin/bash

run_name='T5_large_bird_no_knowledge'
output_dir='output/T5_large_bird'

# Disable wandb for local testing (optional)
export WANDB_MODE=disabled

echo '''flying'''

# For macOS/CPU training, we need to adjust parameters
# Remove CUDA_VISIBLE_DEVICES since we're on macOS
python train_bird.py --seed 1 --cfg experiment/T5_large_finetune_bird.cfg \
--run_name ${run_name} --logging_strategy steps --logging_first_step true --logging_steps 4 \
--eval_strategy steps --eval_steps 2000 --metric_for_best_model avr --greater_is_better true \
--save_strategy steps --save_steps 2000 --save_total_limit 1 --load_best_model_at_end \
--gradient_accumulation_steps 8 --num_train_epochs 200 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --do_predict false --predict_with_generate true \
--output_dir ${output_dir} --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 1024 \
--report_to none --overwrite_output_dir true
