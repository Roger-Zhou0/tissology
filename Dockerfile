# syntax=docker/dockerfile:1
# The default target contains the application, not models or scientific data.
FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TISSUE_DATA_DIR=/data \
    TISSUE_EXECUTION_BACKEND=local \
    TISSUE_CPU_THREADS=2 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=7860
RUN useradd --create-home --uid 1000 studio \
    && mkdir -p /app /data \
    && chown studio:studio /app /data
WORKDIR /app
COPY deploy/requirements-base.txt /app/deploy/requirements-base.txt
RUN python -m pip install --no-cache-dir -r deploy/requirements-base.txt
COPY --chown=studio:studio pyproject.toml /app/
COPY --chown=studio:studio studio /app/studio
COPY --chown=studio:studio scripts /app/scripts
COPY --chown=studio:studio deploy /app/deploy
RUN python -m pip install --no-cache-dir --no-deps .
USER studio
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python /app/deploy/healthcheck.py
ENTRYPOINT ["/bin/sh", "/app/deploy/start.sh"]

# Optional CPU dependencies for an independently provisioned approved model.
# No CONCH code, gated weights, datasets, or credentials are downloaded here.
FROM base AS cpu-inference
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
      torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r deploy/requirements-inference.txt
USER studio

# Keep the small workspace image as the default build, including on hosts
# that do not allow selecting a Docker build target.
FROM base AS workspace
