#!/bin/bash
#SBATCH --job-name=tissue-collection-scan
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/collection-scan-%j.out
#SBATCH --error=logs/collection-scan-%j.err
set -euo pipefail
if [[ ( $# != 2 && $# != 3 ) || ! "${2:-}" =~ ^[0-9a-f]{32}$ || "${1:-}" != /* || ( $# == 3 && ! "${3:-}" =~ ^[0-9a-f]{64}$ ) ]]; then
  echo 'Usage: collection_scan.sbatch ABSOLUTE_COLLECTION_STORE COLLECTION_ID [CONFIGURATION_SHA256]' >&2
  exit 2
fi
cd "${TISSUE_SOURCE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the source checkout}}"
module load pytorch-conda/2.8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n "${TISSUE_INFERENCE_DEPS:-}" ]]; then
  export PYTHONPATH="$TISSUE_INFERENCE_DEPS${PYTHONPATH:+:$PYTHONPATH}"
fi
arguments=(run --root "$1" --collection-id "$2")
if [[ $# == 3 ]]; then
  arguments+=(--configuration-sha="$3")
fi
python -m studio.collection_scan "${arguments[@]}"
