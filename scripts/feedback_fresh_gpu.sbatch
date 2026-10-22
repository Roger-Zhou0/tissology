#!/bin/bash
#SBATCH --job-name=feedback-fresh-gpu
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/feedback-fresh-gpu-%j.out
#SBATCH --error=logs/feedback-fresh-gpu-%j.err
set -euo pipefail
cd "${TISSUE_SOURCE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the source checkout}}"
module load pytorch-conda/2.8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -n "${TISSUE_INFERENCE_DEPS:-}" ]]; then
  export PYTHONPATH="$TISSUE_INFERENCE_DEPS${PYTHONPATH:+:$PYTHONPATH}"
fi
python -m studio.feedback_evaluation encode "$1" "$2"
