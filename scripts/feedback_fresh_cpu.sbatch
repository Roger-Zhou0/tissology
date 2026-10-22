#!/bin/bash
#SBATCH --job-name=feedback-fresh-cpu
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/feedback-fresh-cpu-%j.out
#SBATCH --error=logs/feedback-fresh-cpu-%j.err
set -euo pipefail
cd "${TISSUE_SOURCE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the source checkout}}"
module load pytorch-conda/2.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m studio.feedback_evaluation "$3" "$1" "$2"
