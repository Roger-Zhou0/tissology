"""Explicit CPU execution policy shared by scheduled and portable deployments.

The default remains Delta Slurm. A standalone host must explicitly select
``TISSUE_EXECUTION_BACKEND=local``; Slurm variables are never fabricated.
"""
import os
import re
import socket

CPU_ACCOUNT = "bfqi-delta-cpu"
MAX_THREADS = 8


def execution_backend():
    backend = os.environ.get("TISSUE_EXECUTION_BACKEND", "slurm").strip().lower()
    if backend not in {"slurm", "local"}:
        raise RuntimeError("TISSUE_EXECUTION_BACKEND must be slurm or local")
    return backend


def _local_capacity():
    hostname = socket.gethostname().lower()
    short = hostname.split(".", 1)[0]
    if re.fullmatch(r"(?:dt-)?login\d*", short):
        raise RuntimeError("Local execution is disabled on Delta login hosts")
    scheduled_capacity = None
    if (any(key.startswith("SLURM_") for key in os.environ) or "delta.ncsa" in hostname or
            ".delta.internal" in hostname or re.fullmatch(r"cn\d+", short)):
        # A local child worker may run within a genuinely authorized CPU job.
        # Keep, verify and report that context; never erase/fabricate Slurm vars.
        scheduled_capacity = _slurm_capacity()
    try:
        requested = int(os.environ.get("TISSUE_CPU_THREADS", "2"))
    except ValueError as error:
        raise RuntimeError("TISSUE_CPU_THREADS must be an integer from 1 to 8") from error
    if not 1 <= requested <= MAX_THREADS:
        raise RuntimeError("TISSUE_CPU_THREADS must be an integer from 1 to 8")
    # Container CPU affinity is useful when configured, and never increases the
    # explicit deployment cap. Fractional CPU quotas need a one-thread setting.
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return min(requested, available, MAX_THREADS, scheduled_capacity or MAX_THREADS)


def _slurm_capacity():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU work requires an allocated CPU Slurm allocation")
    if os.environ.get("SLURM_JOB_ACCOUNT") != CPU_ACCOUNT:
        raise RuntimeError(f"CPU work requires account {CPU_ACCOUNT}")
    for name in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        if os.environ.get(name, "").strip() not in ("", "0", "N/A", "(null)"):
            raise RuntimeError("CPU work must not use a GPU allocation")
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if os.environ.get(name, "").strip() not in ("", "N/A", "(null)"):
            raise RuntimeError("CPU work must not use a GPU allocation")
    try:
        allocated = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("SLURM_CPUS_PER_TASK must identify the allocated CPUs") from error
    if allocated < 1:
        raise RuntimeError("At least one allocated CPU is required")
    return allocated


def require_cpu_execution(threads=None):
    """Validate execution context before importing/loading any model."""
    if execution_backend() == "local":
        allocated = _local_capacity()
    else:
        allocated = _slurm_capacity()
    selected = min(allocated, MAX_THREADS) if threads is None else threads
    if type(selected) is not int or not 1 <= selected <= min(allocated, MAX_THREADS):
        raise ValueError("Thread count must fit the allocated CPUs and the eight-thread cap")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(selected)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    return selected


def execution_provenance():
    """Record real scheduler context, using nulls for standalone CPU work."""
    backend = execution_backend()
    return {"execution_backend": backend,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "local_job_id": os.environ.get("TISSUE_LOCAL_JOB_ID") if backend == "local" else None}
