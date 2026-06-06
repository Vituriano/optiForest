#!/bin/bash
#SBATCH --job-name=optiforest
#SBATCH --output=/home/CIN/vsmo/optiForest/results/slurm_%j.log
#SBATCH --chdir=/home/CIN/vsmo/optiForest
#SBATCH --partition=long-simple
#SBATCH --time=48:00:00
#SBATCH --nodelist=cluster-node5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

.venv/bin/python scripts/run_optiforest_study.py \
  --datasets arrhythmia,mnist,cover \
  --runs 15 \
  --flat-workers 8
