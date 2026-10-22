"""Optional directory-prefix acceleration for existing path validation code.

The original ``studio.paths`` module and its frozen hashes remain unchanged.
Within this explicit transaction, parent directories are assumed stable, just as
the original transaction cache assumes already-resolved paths remain stable.
Every distinct leaf is inspected; a later transaction re-resolves every parent.
Symlink leaves and unusual paths use ordinary pathlib resolution. Image byte
hashes, manifest checks and exclusion comparisons remain the caller's job.

CLI: python -m studio.paths_prefix scripts.some_report [arguments ...]
"""
from contextlib import contextmanager
import os
from pathlib import Path
import runpy
import stat
import sys

from . import paths


class _PrefixCache(dict):
    """Adapter for the original resolver's membership-then-lookup protocol.

    Membership fills a missing entry. This is intentionally private: it allows
    existing imported resolver functions to use the faster transaction without
    monkeypatching them or changing their frozen source.
    """

    def __init__(self, previous=None):
        super().__init__(previous or {})
        self.parents = {}

    def __contains__(self, key):
        if not dict.__contains__(self, key):
            cwd, value = key
            path = Path(value)
            if not path.name or path.name == "..":
                resolved = path.resolve()
            else:
                parent_key = (cwd, str(path.parent))
                if parent_key not in self.parents:
                    parent = path.parent.resolve()
                    self.parents[parent_key] = parent if parent.is_dir() else None
                parent = self.parents[parent_key]
                if parent is None:
                    resolved = path.resolve()
                else:
                    try:
                        mode = os.lstat(path).st_mode
                    except FileNotFoundError:
                        resolved = parent / path.name
                    except OSError:
                        resolved = path.resolve()
                    else:
                        resolved = path.resolve() if stat.S_ISLNK(mode) else parent / path.name
            dict.__setitem__(self, key, resolved)
        return True


@contextmanager
def prefix_paths():
    """Scope directory and leaf caches to one immutable-input transaction."""
    previous = paths._cache.get()
    if isinstance(previous, _PrefixCache):
        yield previous
        return
    cache = _PrefixCache(previous)
    token = paths._cache.set(cache)
    try:
        yield cache
    finally:
        paths._cache.reset(token)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m studio.paths_prefix MODULE [ARGUMENT ...]")
    module, arguments = sys.argv[1], sys.argv[2:]
    original = sys.argv
    try:
        sys.argv = [module, *arguments]
        with prefix_paths():
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
