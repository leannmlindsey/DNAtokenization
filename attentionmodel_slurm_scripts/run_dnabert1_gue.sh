#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --mail-user=leann.lindsey@utah.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name=gue_dnabert1
#SBATCH --gpus=v100-32:1
#SBATCH -o /ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/outerror/%x%j.outerror

# Load Modules 
module load anaconda3/2024.10-1
nvidia-smi

echo "starting DNABERT env on conda"
source activate dna_sandbox
conda list
script_dir="/ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/MODELS/DNABERT_2/finetune"
output_path="/ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/RESULTS/DNABERT_1"

#data_path=$1
#lr=3e-5
lr=3e-5
seed=$1
kmer=6
vocab=bpe
data_path="/ocean/projects/bio230026p/ahabib/FINETUNE_DATA"
m=5
echo "The provided data_path is $data_path"
echo "The learning rate is $lr"
echo "The seed is $seed"
echo "The output path is $output_path"


echo "The provided data_path is $data_path"

echo "The provided kmer is: $kmer, data_path is $data_path"

# sh scripts/run_dna1.sh 3 ; sh scripts/run_dna1.sh 4 ; sh scripts/run_dna1.sh 5 ; sh scripts/run_dna1.sh 6

for seed in $seed
do
    #for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    #do
    #    python $script_dir/train.py \
    #        --model_name_or_path zhihan1996/DNA_bert_${kmer} \
    #        --data_path  ${data_path}/GUE/EMP/$data \
    #        --kmer ${kmer} \
    #        --run_name DNABERT1_${kmer}_EMP_${data}_seed${seed} \
    #        --model_max_length 512 \
    #        --per_device_train_batch_size 8 \
    #        --per_device_eval_batch_size 16 \
    #        --gradient_accumulation_steps 1 \
    #        --learning_rate 3e-5 \
    #        --num_train_epochs 3 \
    #        --fp16 \
    #        --save_steps 200 \
    #        --output_dir $output_path/gue_${run_name} \
    #        --evaluation_strategy steps \
    #        --eval_steps 200 \
    #        --warmup_steps 50 \
    #        --logging_steps 100000 \
    #        --overwrite_output_dir True \
    #        --log_level info \
    #        --seed ${seed} \
    #        --find_unused_parameters False
    #done


    for data in prom_core_all prom_core_notata
    do
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in prom_core_tata
    do
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 310 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in prom_300_tata
    do
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 310 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in reconstructed
    do
        python python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/splice/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_splice_${data}_seed${seed} \
            --model_max_length 410 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in covid
    do
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/virus/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_virus_${data}_seed${seed} \
            --model_max_length 512 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-5 \
            --num_train_epochs 9 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/mouse/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_mouse_${data}_seed${seed} \
            --model_max_length 110 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 5 \
            --max_steps 1000 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python $script_dir/train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/tf/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_tf_${data}_seed${seed} \
            --model_max_length 110 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/gue_DNABERT1_${kmer}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done
