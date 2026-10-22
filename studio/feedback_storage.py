"""Publish complete immutable feedback JSON files without replacing old versions."""
import json
import os
from pathlib import Path
import tempfile

from .storage import digest


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {**value, "sha256": digest(value)}
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(artifact, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic exclusive publication. Readers see either no file or all bytes.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return artifact
