#!/bin/bash -l

#SBATCH --job-name=test-stllm
#SBATCH --comment="user03"

#SBATCH --partition=defq
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=dgx02
##SBATCH --time=14-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate stllm-test1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd /home/user03/VARDiff-test/newtest/AIC-LLM/code/src


# bash '../scripts/pems03.sh'

bash '../scripts/pems04.sh'

# bash '../scripts/pems08.sh'

# bash '../scripts/pems07.sh'

