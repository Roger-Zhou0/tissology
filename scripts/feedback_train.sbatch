#!/bin/bash
#SBATCH --job-name=tissue-feedback
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=logs/feedback-%j.out
#SBATCH --error=logs/feedback-%j.err
set -euo pipefail
cd "${TISSUE_SOURCE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the source checkout}}"
module load pytorch-conda/2.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m studio.feedback_train train "$1" "$2"
