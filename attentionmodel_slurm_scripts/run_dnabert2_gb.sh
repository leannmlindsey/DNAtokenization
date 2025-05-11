#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --job-name=dnabert2_gb
#SBATCH --gpus=v100-32:1
#SBATCH -o /path/to/output/log/directory/%x%j.outerror

# Load Modules 
module load anaconda3/2024.10-1
nvidia-smi

*********************************************************************************************
# modify these paths for your own system
data_path="/full/path/to/data"
script_dir="/full/path/to/DNABERT_2/finetune"
output_path="/full/path/to/RESULTS/DNABERT2/GB"
# activate the conda environment you created for the attention models
source activate dna_sandbox
********************************************************************************************
echo "starting DNABERT env on conda"
mkdir $output_path

lr=3e-5
seed=$1
data=$2
echo "The provided data_path is $data_path"
echo "The learning rate is $lr"
echo "The seed is $seed"
echo "The output path is $output_path"
cd $script_dir

get_max_length() {
    local dataset=$1

    case $dataset in
        # 200 length datasets
        demo_coding_vs_intergenomic_seqs|demo_human_or_worm)
            echo 200
            ;;
        # 251 length datasets
        human_nontata_promoters)
            echo 251
            ;;
        # 500 length datasets
        human_enhancers_cohn)
            echo 500
            ;;
        # 512 length datasets
        human_enhancers_ensembl|human_ensembl_regulatory|human_ocr_ensembl)
            echo 512
            ;;
        # 1024 length datasets
        dummy_mouse_enhancers_ensembl)
            echo 1024
            ;;
        # Default case if dataset not recognized
        *)
            echo 128  # Default to original value
            ;;
    esac
}

for seed in $seed
do
    # Uncomment below if you prefer to run the full GB benchmark
    #for data in dummy_mouse_enhancers human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory, human_ocr_ensembl human_nontata_promoters demo_coding_vs_intergenomic demo_coding_vs_intergenomic demo_human_or_worm drosophilia_enhancers 
    do
	max_length=$(get_max_length $data)

        echo "Training on dataset: $data with max_length: $max_length"

        python train_orig.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  ${data_path}/$data \
            --kmer -1 \
            --run_name DNABERT2_${lr}_${data}_seed${seed} \
            --model_max_length $max_length \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $output_path/gb_DNABERT2_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done
done
