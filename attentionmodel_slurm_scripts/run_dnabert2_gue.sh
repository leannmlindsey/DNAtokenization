#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --job-name=gue_dnabert2
#SBATCH --gpus=v100-32:1
#SBATCH -o /path/to/output/log/directory/%x%j.outerror

# Load Modules 
module load anaconda3/2024.10-1
nvidia-smi

*********************************************************************************************
# modify these paths for your own system
data_path="/full/path/to/data"
script_dir="/full/path/to/DNABERT_2/finetune"
output_path="/full/path/to/RESULTS/DNABERT/GB"
# activate the conda environment you created for the attention models
source activate dna
********************************************************************************************
echo "starting DNABERT env on conda"
mkdir $output_path


lr=3e-5
seed=$1
dataset=$2
vocab=bpe
m=5
echo "The provided data_path is $data_path"
echo "The learning rate is $lr"
echo "The seed is $seed"
echo "The output path is $output_path"


echo "The provided data_path is $data_path"

for seed in $seed
do
    #for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    #do
    #    python $script_dir/train_orig.py \
    #        --model_name_or_path zhihan1996/DNABERT-2-117M \
    #        --data_path  $data_path/GUE/EMP/$data \
    #        --kmer -1 \
    #        --run_name DNABERT2_${vocab}_${lr}_EMP_${data}_seed${seed} \
    #        --model_max_length 128 \
    #        --per_device_train_batch_size 8 \
    #        --per_device_eval_batch_size 16 \
    #        --gradient_accumulation_steps 1 \
    #        --learning_rate ${lr} \
    #        --num_train_epochs 3 \
    #        --fp16 \
    #        --save_steps 200 \
    #        --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
    #        --evaluation_strategy steps \
    #        --eval_steps 200 \
    #        --warmup_steps 50 \
    #        --logging_steps 100000 \
    #        --overwrite_output_dir True \
    #        --log_level info \
    #        --find_unused_parameters False
    #done



    for data in prom_core_all prom_core_notata
    do
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in prom_core_tata
    do
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done



    for data in prom_300_tata
    do 
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done 


    for data in reconstructed
    do
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/splice/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_splice_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done



    for data in covid
    do
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/virus/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_virus_${data}_seed${seed} \
            --model_max_length 256 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 8 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done

    for data in 0 1 2 3 4
    do 
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/mouse/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_mouse_${data}_seed${seed} \
            --model_max_length 30 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --max_steps 1000 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_mouse_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python $script_dir/train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/tf/$data \
            --kmer -1 \
            --run_name gue_DNABERT2_${vocab}_${lr}_tf_${data}_seed${seed} \
            --model_max_length 30 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/dnabert2/gue_${run_name}_${lr}_tf_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done
done
