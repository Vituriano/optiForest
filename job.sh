#!/bin/bash
#SBATCH --job-name=optiforest
#SBATCH --output=results/slurm_%j.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

cd ~/optiForest

.venv/bin/python scripts/run_optiforest_study.py \
  --datasets mnist,cover,arrhythmia \
  --runs 15 \
  --workers 3
