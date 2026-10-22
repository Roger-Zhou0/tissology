#!/bin/bash
#SBATCH --job-name=tissue-collection-scan
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/collection-scan-cpu-%j.out
#SBATCH --error=logs/collection-scan-cpu-%j.err
set -euo pipefail
if [[ $# != 3 || ! "${2:-}" =~ ^[0-9a-f]{32}$ || "${1:-}" != /* || ! "${3:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Usage: collection_scan_cpu.sbatch ABSOLUTE_COLLECTION_STORE COLLECTION_ID CONFIGURATION_SHA256' >&2
  exit 2
fi
if [[ "${SLURM_JOB_ACCOUNT:-}" != bfqi-delta-cpu || -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-}" != 8 ]]; then
  echo 'Collection scans require an eight-CPU bfqi-delta-cpu Slurm allocation.' >&2
  exit 2
fi
for name in SLURM_GPUS SLURM_GPUS_ON_NODE; do
  value="${!name:-}"
  if [[ -n "$value" && "$value" != 0 && "$value" != N/A && "$value" != '(null)' ]]; then
    echo 'CPU collection scans cannot use a GPU allocation.' >&2
    exit 2
  fi
done
for name in SLURM_JOB_GPUS SLURM_STEP_GPUS; do
  value="${!name:-}"
  if [[ -n "$value" && "$value" != N/A && "$value" != '(null)' ]]; then
    echo 'CPU collection scans cannot use allocated GPU identifiers.' >&2
    exit 2
  fi
done
cd "${TISSUE_SOURCE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the source checkout}}"
module load pytorch-conda/2.8
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -n "${TISSUE_INFERENCE_DEPS:-}" ]]; then
  export PYTHONPATH="$TISSUE_INFERENCE_DEPS${PYTHONPATH:+:$PYTHONPATH}"
fi
python -m studio.collection_scan run --root "$1" --collection-id "$2" --configuration-sha="$3"
