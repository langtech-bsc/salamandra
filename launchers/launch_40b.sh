#!/bin/bash
#SBATCH --job-name=gpt_40b
#SBATCH --error=logs/job_40b_%j.err
#SBATCH --output=logs/job_40b_%j.out
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00:00
#SBATCH --partition=acc
#SBATCH --qos acc_xehpc
#SBATCH --account bsc88
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --nodes=256

export WORKING_DIR=$(dirname $PWD)

# CONFIG FILE
export CONFIG_PATH=$WORKING_DIR/configs
export CONFIG_NAME=bsc_40b.yaml
export MODEL_NAME="${CONFIG_NAME%.*}"

# PATHS
export PATH_SINGULARITY=$WORKING_DIR/singularity_images/nemo_2403
export PATH_TOKENIZER=$WORKING_DIR/tokenizer
export PATH_DATA=$WORKING_DIR/data
export PATH_RESULTS=$WORKING_DIR/results/${MODEL_NAME}/output
export PATH_LOGS=$WORKING_DIR/results/${MODEL_NAME}/logs
export PATH_CACHE=$WORKING_DIR/results/${MODEL_NAME}/cache

# CREATE OUTPUT DIRS
mkdir -p $PATH_RESULTS
mkdir -p $PATH_LOGS
mkdir -p $PATH_CACHE

# ENVIRONMENT VARIABLES
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRANSFORMERS_OFFLINE=1
export SLURM_NTASKS_PER_NODE=4
export SLURM_CPU_BIND=none

### NCCL
export NCCL_NVLS_ENABLE=0
export NCCL_IB_TIMEOUT=70
export NCCL_IB_HCA=mlx5
export NCCL_IB_RETRY_CNT=40
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# WANDB
export WANDB_MODE=offline
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export WANDB_DIR=$PATH_RESULTS/wandb
export WANDB_CONFIG_DIR=$WANDB_DIR/config
export WANDB_HTTP_TIMEOUT=8000
export WANDB_INIT_TIMEOUT=8000

# CACHE
export HF_HOME=$PATH_CACHE
export HUGGINGFACE_HOME=$PATH_CACHE
export NUMBA_CACHE_DIR=$PATH_CACHE
export WANDB_CACHE_DIR=$PATH_CACHE

# PRINTS
echo ">>> SINGULARITY: $( basename $PATH_SINGULARITY )"
echo ">>> PATH_CONFIG: $CONFIG_PATH/$CONFIG_NAME"
echo ">>> PATH_TOKENIZER: $PATH_TOKENIZER"
echo ">>> PATH_RESULTS: $PATH_RESULTS"
echo ">>> PATH_DATA: $PATH_DATA"

# SRUN
srun ../run_singularity.sh
