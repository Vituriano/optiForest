#!/bin/bash
#SBATCH --job-name=optiforest
#SBATCH --output=/home/CIN/vsmo/optiForest/results/slurm_%j.log
#SBATCH --chdir=/home/CIN/vsmo/optiForest
#SBATCH --partition=long-simple
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --nodelist=cluster-node5
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

.venv/bin/python scripts/run_optiforest_study.py \
  --datasets mnist,cover,arrhythmia \
  --runs 15 \
  --flat-workers 16
