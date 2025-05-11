#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --job-name=dnabert2_ntv2
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
data_path="/ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/DATA"
echo "The provided data_path is $data_path"
echo "The learning rate is $lr"
echo "The seed is $seed"
echo "The output path is $output_path"

# Define a function to get the appropriate max_length for each dataset
get_max_length() {
    local dataset=$1

    case $dataset in
        # Promoter datasets - 300bp
        promoter_all|promoter_tata|promoter_no_tata)
            echo 300
            ;;
        # Enhancer datasets - 200bp
        enhancers|enhancers_types)
            echo 200
            ;;
        # Splice sites datasets
        splice_sites_all)
            echo 400
            ;;
        splice_sites_acceptors|splice_sites_donors)
            echo 600
            ;;
        # Histone modification datasets - 500bp
        H3K*|H4K*|H2AFZ)
            echo 500
            ;;
        # Default case if dataset not recognized
        *)
            echo 128  # Default to original value
            ;;
    esac
}
cd $script_dir

for seed in $seed
do
    # Uncomment below if you prefer to run the full GB benchmark
    #for data in H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1 enhancers enhancers_types promoter_all promoter_no_tata promoter_tata splice_sites_acceptors splice_sites_all splice_sites_donors
    for data in $dataset
    do
        # Get the appropriate max_length for this dataset
        max_length=$(get_max_length $data)

        echo "Training on dataset: $data with max_length: $max_length"

        python train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_${data}_seed${seed} \
            --model_max_length $max_length \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $output_path/ntv2_DNABERT2_${vocab}_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done
done
