#!/bin/sh
set -eu

# Keep model execution serialized in one application process. Initialization
# only creates missing storage; it must not replace an existing collection.
python - <<'PY'
import os
if os.environ.get('TISSUE_PUBLIC_READONLY', '0') != '1' and len(os.environ.get('TISSUE_ACCESS_PASSWORD', '')) < 16:
    raise SystemExit('Set TISSUE_ACCESS_PASSWORD to at least 16 characters, or explicitly select TISSUE_PUBLIC_READONLY=1')
PY
python -m studio.portable init --data-dir "${TISSUE_DATA_DIR:-/data}"
if [ -n "${TISSUE_MODEL_BUNDLE:-}" ] && [ "${TISSUE_PUBLIC_READONLY:-0}" != "1" ]; then
  python -m studio.portable configure --data-dir "${TISSUE_DATA_DIR:-/data}" \
    --bundle "$TISSUE_MODEL_BUNDLE"
fi
exec python -m uvicorn studio.portable:from_environment \
  --factory --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
