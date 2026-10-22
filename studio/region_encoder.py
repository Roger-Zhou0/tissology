"""Lazy, bounded CONCH crop encoding on the application's bfqi CPU allocation.

The exact rectangular crop remains the evidence. For retrieval only, center it
on a white square before official CONCH preprocessing so center cropping cannot
silently omit the edges the researcher selected. No classifier or labels load.
"""
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, file_hash
from .cpu_features import load_cpu_encoder, require_cpu_allocation
from .storage import read
from .study import verify
from .execution import execution_provenance

REGION_ENCODER_SOURCE_SHA256 = file_hash(__file__)


class RegionEncoder:
    def __init__(self, configuration, *, threads=None):
        path = Path(configuration)
        if path.is_symlink() or not path.is_file():
            raise ValueError("A regular, verified scan configuration is required for region search")
        config = read(path)
        verify(config)
        if (config.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                config.get("code_revision") != CODE_REVISION):
            raise ValueError("Region search requires the same pinned CONCH encoder as the collection")
        self.checkpoint = config["checkpoint"]
        self.code_directory = config["code_directory"]
        self.configuration_sha256 = config["sha256"]
        self.threads = threads
        self._encoder = None
        self._lock = threading.Lock()

    def encode_region(self, image):
        # Check even after warmup: an inherited model is never permission to
        # run on a login node, a different allocation, or a GPU job.
        threads = require_cpu_allocation(self.threads)
        if (not isinstance(image, Image.Image) or image.mode != "RGB" or
                min(image.size) < 32 or max(image.size) > 4096 or
                image.width * image.height > 16_000_000):
            raise ValueError("Region encoding requires an RGB crop with each side between 32 and 4096 pixels")
        with self._lock:
            started = time.monotonic()
            cold = self._encoder is None
            if cold:
                self._encoder = load_cpu_encoder(self.checkpoint, self.code_directory, threads=threads)
            model, preprocess, metadata = self._encoder
            side = max(image.size)
            left, top = (side - image.width) // 2, (side - image.height) // 2
            square = Image.new("RGB", (side, side), (255, 255, 255))
            square.paste(image, (left, top))
            import torch
            with torch.inference_mode():
                tensor = preprocess(square).unsqueeze(0).to(device="cpu", dtype=torch.float32)
                result = model.encode_image(tensor, proj_contrast=True, normalize=True)
                vector = result[0].detach().cpu().numpy().astype(np.float32, copy=True)
            if (vector.shape != (512,) or not np.isfinite(vector).all() or
                    not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)):
                raise ValueError("CONCH returned an invalid normalized region embedding")
            return vector, {
                **metadata,
                "crop_input_transform": "centered white square padding; full rectangle retained; no geometric stretch",
                "crop_size_px": [image.width, image.height],
                "padded_size_px": [side, side],
                "padding_px": [left, top, side - image.width - left, side - image.height - top],
                "padding_rgb": [255, 255, 255],
                "configuration_sha256": self.configuration_sha256,
                "region_encoder_source_sha256": REGION_ENCODER_SOURCE_SHA256,
                **execution_provenance(),
                "cold_start": cold,
                "elapsed_seconds": time.monotonic() - started,
                "encoder_training_performed": False,
                "scope": "Crop-to-whole-image appearance ranking; no region localization or segmentation",
            }
