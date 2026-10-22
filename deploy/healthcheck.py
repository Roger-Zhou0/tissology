"""Probe the minimal unauthenticated health route, not a user's data."""
import os
import urllib.request

port = int(os.environ.get("PORT", "7860"))
with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=4) as response:
    if response.status != 200:
        raise SystemExit(1)
