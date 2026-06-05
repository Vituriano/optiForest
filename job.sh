#!/bin/bash
#SBATCH --job-name=optiforest
#SBATCH --output=/home/CIN/vsmo/optiForest/results/slurm_%j.log
#SBATCH --chdir=/home/CIN/vsmo/optiForest
#SBATCH --partition=long-simple
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G


.venv/bin/python scripts/run_optiforest_study.py \
  --datasets mnist,cover,arrhythmia \
  --runs 15 \
  --workers 3
