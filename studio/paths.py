"""Cache repeated canonical-path checks only within one validation transaction.

Overlapping exclusion manifests contain the same images thousands of times.
Resolving those paths repeatedly generates avoidable shared-filesystem metadata
traffic. A transaction still resolves every distinct path normally; later calls
outside it or in a new transaction check the filesystem again.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path


_cache = ContextVar("tissue_validation_path_cache", default=None)


@contextmanager
def _transaction():
    # Nested validations belong to the outer transaction and share its cache.
    if _cache.get() is not None:
        yield
        return
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def cached_paths(function=None):
    """Use as @cached_paths, @cached_paths(), or with cached_paths()."""
    if function is None:
        return _transaction()

    @wraps(function)
    def wrapped(*args, **kwargs):
        with _transaction():
            return function(*args, **kwargs)
    return wrapped


def resolve_path(value):
    path = Path(value)
    cache = _cache.get()
    if cache is None:
        return path.resolve()
    # Preserve '..' until actual resolution: lexical normalization before
    # resolving symlinks could identify a different filesystem object.
    key = (None if path.is_absolute() else os.getcwd(), str(path))
    if key not in cache:
        cache[key] = path.resolve()
    return cache[key]


resolved = resolve_path
