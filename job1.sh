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
conda activate stllm_test

cd /home/user03/VARDiff-test/final-test1/AIC-LLM/code/src
bash '../scripts/pems031.sh'
