#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 20:00:00
#SBATCH --job-name=ntv2_gue
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

lr=1e-4
seed=$1
dataset=$2
data_path="/ocean/projects/bio230026p/ahabib/FINETUNE_DATA"
m=5
echo "The provided data_path is $data_path"
echo "The learning rate is $lr"
echo "The seed is $seed"
echo "The output path is $output_path"
cd $script_dir

if [ "$m" -eq 0 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-1000g
    run_name=NT_500_1000g
elif [ "$m" -eq 1 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-human-ref
    run_name=NT_500_human
elif [ "$m" -eq 2 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-1000g
    run_name=NT_2500_1000g
elif [ "$m" -eq 3 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    run_name=NT_2500_multi
elif [ "$m" -eq 4 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
    run_name=NTv2-250m-ms
elif [ "$m" -eq 5 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    run_name=NTv2-500m-ms
else
    echo "Wrong argument"
    exit 1
fi
echo "Use: $model"

# Define a function to get the appropriate max_length for each dataset
get_max_length() {
    local dataset=$1

    case $dataset in
        # Promoter core datasets - 20bp
        prom_core_all|prom_core_notata|prom_core_tata)
            echo 80
            ;;
        # Promoter 300 datasets - 70bp
        prom_300_all|prom_300_notata|prom_300_tata)
            echo 310
            ;;
        # Reconstructed dataset - 80bp
        reconstructed)
            echo 410
            ;;
        # COVID dataset - 256bp
        covid)
            echo 1024
            ;;
        # Numeric datasets - 30bp
        m0|m1|m2|m3|m4|tf0|tf1|tf2|tf3|tf4)
            echo 110
            ;;
        *)
            echo 128  # Default to original value
            ;;
    esac
}
get_datapath() {
    local dataset=$1

    case $dataset in
        # Promoter core datasets - 20bp
        prom_core_all|prom_core_notata|prom_core_tata)
            echo "GUE/prom/$dataset"
            ;;
        # Promoter 300 datasets - 70bp
        prom_300_all|prom_300_notata|prom_300_tata)
            echo "GUE/prom/$dataset"
            ;;
        # Reconstructed dataset - 80bp
        reconstructed)
            echo "GUE/splice/$dataset"
            ;;
        # COVID dataset - 256bp
        covid)
            echo "GUE/virus/$dataset"
            ;;
        # Numeric datasets - 30bp
        m0|m1|m2|m3|m4)
            num=${dataset#m} 
            echo "GUE/mouse/$num"
            ;;
	tf0|tf1|tf2|tf3|tf4)
	    num=${dataset#tf} 
            echo "GUE/tf/$num"
            ;;

        *)
            echo "This dataset does not exist" 
            ;;
    esac
}

for seed in $seed
do
    for data in $dataset
     do
	max_length=$(get_max_length $data)
	data_path_suffix=$(get_datapath $data)

        echo "Training on dataset: $data with max_length: $max_length with model: ${model} with datapath: $data_path_suffix"

        python train_orig.py \
            --model_name_or_path ${model} \
            --data_path  $data_path/$data_path_suffix \
            --kmer -1 \
            --run_name NTv2_${lr}_${data}_seed${seed} \
            --model_max_length $max_length \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 6 \
            --fp16 \
            --save_steps 400 \
            --output_dir $output_path/gue_NTv2_${lr}_${data}_seed${seed} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done
done
