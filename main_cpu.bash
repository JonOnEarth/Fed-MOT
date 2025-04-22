#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=cpu
#SBATCH --mem=48GB
#SBATCH --ntasks=2
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
# module load anaconda3/2022.05 cuda/11.8
# conda create --name pytorch_env python=3.10 -y
source activate pytorch_env
# pip install ujson

python main.py --dataset $1 --model_name $2 --K $3 --n_assign $4 --warm_up $5 --client_group $6