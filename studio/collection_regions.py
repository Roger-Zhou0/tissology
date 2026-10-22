"""Immutable rectangular image evidence and lazily cached CPU crop queries.

Coordinates address the normalized PNG in a frozen collection, with a top-left
origin and half-open bounds. A crop query ranks whole images; it is not a
localization or segmentation result. No model is loaded when evidence is saved.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading

import numpy as np
from PIL import Image

from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, MODEL, REPRESENTATION
from .feedback_storage import freeze
from .storage import digest, now
from .study import verify

MIN_SIDE = 32
MAX_SIDE = 4096
MAX_REGIONS = 256
MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 16_000_000
COORDINATES = "original normalized PNG pixels; top-left origin; [x0,y0,x1,y1), right and bottom exclusive"
SCOPE = "Researcher-selected rectangle; no automatic tissue boundary, localization, or physical scale is inferred."
INPUT_TRANSFORM = "centered white square padding; full rectangle retained; no geometric stretch"
EMBEDDING_POLICY = {"version": "collection-region-conch-white-square-v1", "model": MODEL,
                    "checkpoint_sha256": CHECKPOINT_SHA256, "code_revision": CODE_REVISION,
                    "representation": REPRESENTATION, "crop_input_transform": INPUT_TRANSFORM,
                    "padding_rgb": [255, 255, 255], "official_preprocess_size_px": 448}
EMBEDDING_POLICY_SHA256 = digest(EMBEDDING_POLICY)
_CREATE_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()


class RegionBusy(ValueError):
    """Bounded work admission refuses a concurrent operation instead of queuing."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path, bound=MAX_BYTES):
    """Read a regular file without following its final symlink, with a byte cap."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        import stat
        initial = os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > bound:
            raise ValueError("Region artifact is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(bound + 1)
        final = os.fstat(fd)
        if (len(data) > bound or (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) !=
                (final.st_size, final.st_mtime_ns, final.st_ctime_ns)):
            raise ValueError("Region artifact changed while being read")
        return data
    finally:
        os.close(fd)


@contextmanager
def _admission(lock, lock_path):
    if not lock.acquire(blocking=False):
        raise RegionBusy("Another region operation is running. Try again shortly.")
    fd = None
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RegionBusy("Another region operation is running. Try again shortly.") from error
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock.release()


def _bounds(bounds, size):
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or
            any(type(value) is not int for value in bounds)):
        raise ValueError("Region bounds must be four integer pixel coordinates")
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError("Region bounds must stay inside the source image")
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        raise ValueError(f"Select a region at least {MIN_SIDE} by {MIN_SIDE} pixels")
    if x1 - x0 > MAX_SIDE or y1 - y0 > MAX_SIDE:
        raise ValueError(f"Select a region no larger than {MAX_SIDE} pixels on each side")
    return list(bounds)


class CollectionRegions:
    def __init__(self, semantic, encoder=None):
        self.semantic = semantic
        self.encoder = encoder
        self.directory = semantic.directory / "regions"

    def _safe(self, path=None):
        self.semantic.assert_current()
        root = self.semantic.collections.directory
        selected = Path(path) if path is not None else self.directory
        if selected != root and root not in selected.parents:
            raise ValueError("Region storage must remain inside its collection")
        current = root
        for part in selected.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Region storage cannot contain symbolic links")
        if selected.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("Region storage must remain inside its collection")
        return selected

    def _directory(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise ValueError("Invalid region identifier")
        return self._safe(self.directory / identifier)

    def _source(self, identifier):
        row = self.semantic.row(identifier)
        path = self._safe(self.semantic.root / "images" / (identifier + ".png"))
        data = _bytes(path)
        if _sha(data) != row["image_sha256"]:
            raise ValueError("Collection image bytes differ from the frozen region source")
        with Image.open(io.BytesIO(data)) as original:
            if (original.width * original.height > MAX_PIXELS or
                    getattr(original, "n_frames", 1) != 1 or
                    list(original.size) != row.get("size_px", [row.get("width"), row.get("height")])):
                raise ValueError("Region source dimensions differ from the frozen image")
            image = original.convert("RGB")
        return row, image

    def _binding(self, image_id, bounds, size, version):
        return {"collection_id": self.semantic.collection_id,
                "catalog_sha256": self.semantic.catalog["sha256"],
                "image_id": image_id, "image_sha256": self.semantic.row(image_id)["image_sha256"],
                "bounds_px": list(bounds), "source_size_px": list(size),
                "prediction_version": version, "coordinate_system": COORDINATES,
                "crop_format": "RGB PNG; exact integer pixel rectangle; no resizing"}

    def create(self, image_id, bounds_px, prediction_version=None):
        self._safe()
        row, source = self._source(image_id)
        bounds = _bounds(bounds_px, source.size)
        _, version = self.semantic.collections.prediction_view(self.semantic.collection_id, prediction_version)
        binding = self._binding(image_id, bounds, source.size, version)
        identifier = digest(binding)
        directory = self._directory(identifier)
        if directory.exists():
            return self.load(identifier)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._safe()
        with _admission(_CREATE_LOCK, self.directory / ".create.lock"):
            if directory.exists():
                return self.load(identifier)
            if sum(1 for path in self.directory.iterdir() if re.fullmatch(r"[a-f0-9]{64}", path.name)) >= MAX_REGIONS:
                raise ValueError(f"This collection already has the {MAX_REGIONS} saved-region limit")
            crop = source.crop(tuple(bounds))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            data = buffer.getvalue()
            if len(data) > MAX_BYTES:
                raise ValueError("The selected crop exceeds the saved image size limit")
            # Re-read the frozen source after decoding, before publishing evidence.
            self._source(image_id)
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
            try:
                (temporary / "crop.png").write_bytes(data)
                record = freeze(temporary / "region.json", {
                    "schema": "collection-region-v1", "id": identifier, "created_at": now(),
                    **binding, "filename": row["filename"], "crop_sha256": _sha(data),
                    "crop_size_px": list(crop.size), "image_url": self.image_url(identifier),
                    "source_image_url": self.semantic.image_url(image_id), "scope": SCOPE,
                    "annotation_extent": "selected rectangular region", "selection_method": "researcher rectangle"})
                os.rename(temporary, directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return record

    def image_url(self, identifier):
        return f"api/collections/{self.semantic.collection_id}/semantic/regions/{identifier}/image"

    def load(self, identifier):
        directory = self._directory(identifier)
        record = json.loads(_bytes(self._safe(directory / "region.json"), 128 * 1024))
        verify(record)
        row, source = self._source(record["image_id"])
        bounds = _bounds(record["bounds_px"], source.size)
        _, version = self.semantic.collections.prediction_view(self.semantic.collection_id, record["prediction_version"]["id"])
        binding = self._binding(row["id"], bounds, source.size, version)
        if (record.get("schema") != "collection-region-v1" or record.get("id") != identifier or
                digest(binding) != identifier or any(record.get(key) != value for key, value in binding.items()) or
                record.get("filename") != row["filename"] or record.get("image_url") != self.image_url(identifier) or
                record.get("source_image_url") != self.semantic.image_url(row["id"]) or
                record.get("annotation_extent") != "selected rectangular region"):
            raise ValueError("Region source, bounds, or prediction version changed")
        data = _bytes(self._safe(directory / "crop.png"))
        if _sha(data) != record.get("crop_sha256"):
            raise ValueError("Saved region crop bytes changed")
        with Image.open(io.BytesIO(data)) as saved:
            crop = source.crop(tuple(bounds))
            if (saved.mode != "RGB" or saved.size != crop.size or list(saved.size) != record.get("crop_size_px") or
                    saved.tobytes() != crop.tobytes()):
                raise ValueError("Saved region pixels differ from the source rectangle")
        return record

    def image_bytes(self, identifier):
        record = self.load(identifier)
        data = _bytes(self._safe(self._directory(identifier) / "crop.png"))
        if _sha(data) != record["crop_sha256"]:
            raise ValueError("Saved region crop changed while being read")
        return data

    @staticmethod
    def _provenance(provenance, crop_size):
        if (not isinstance(provenance, dict) or provenance.get("model") != MODEL or
                provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                provenance.get("code_revision") != CODE_REVISION or
                provenance.get("representation") != REPRESENTATION or provenance.get("device") != "cpu" or
                not isinstance(provenance.get("preprocessing"), dict) or not provenance["preprocessing"] or
                provenance.get("crop_input_transform") != INPUT_TRANSFORM):
            raise ValueError("Region embedding requires the pinned projected CPU CONCH encoder and preprocessing")
        width, height = crop_size
        side = max(crop_size)
        left, top = (side - width) // 2, (side - height) // 2
        if (provenance.get("crop_size_px") != crop_size or provenance.get("padded_size_px") != [side, side] or
                provenance.get("padding_rgb") != [255, 255, 255] or
                provenance.get("padding_px") != [left, top, side - width - left, side - height - top] or
                provenance["preprocessing"].get("image_size") != 448):
            raise ValueError("Region embedding padding or preprocessing differs from the selected rectangle")

    def embedding_directory(self, identifier):
        return self._safe(self._directory(identifier) / ("embedding-" + EMBEDDING_POLICY_SHA256))

    def _cached_embedding(self, record):
        directory = self.embedding_directory(record["id"])
        metadata = json.loads(_bytes(self._safe(directory / "embedding.json"), 256 * 1024))
        verify(metadata)
        self._provenance(metadata.get("encoder_provenance"), record["crop_size_px"])
        data = _bytes(self._safe(directory / "embedding.npy"), 8192)
        if (metadata.get("schema") != "collection-region-embedding-v1" or
                metadata.get("region_id") != record["id"] or metadata.get("region_sha256") != record["sha256"] or
                metadata.get("embedding_policy_sha256") != EMBEDDING_POLICY_SHA256 or
                metadata.get("embedding_policy") != EMBEDDING_POLICY or
                metadata.get("crop_sha256") != record["crop_sha256"] or metadata.get("file_sha256") != _sha(data)):
            raise ValueError("Region embedding cache differs from its frozen evidence")
        vector = np.load(io.BytesIO(data), allow_pickle=False)
        if (vector.dtype != np.float32 or vector.shape != (512,) or not np.isfinite(vector).all() or
                not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)):
            raise ValueError("Region embedding must be a normalized 512-dimensional projected vector")
        return vector, metadata

    def embedding(self, identifier):
        record = self.load(identifier)
        directory = self.embedding_directory(identifier)
        if directory.exists():
            return self._cached_embedding(record)
        if self.encoder is None:
            raise ValueError("Region search is unavailable; the selected region can still be saved as evidence")
        # The single lock is shared across collections and worker processes.
        self._safe(self.semantic.collections.directory / ".region-encoder.lock")
        with _admission(_ENCODE_LOCK, self.semantic.collections.directory / ".region-encoder.lock"):
            if directory.exists():
                return self._cached_embedding(record)
            data = self.image_bytes(identifier)
            with Image.open(io.BytesIO(data)) as image:
                vector, provenance = self.encoder.encode_region(image.convert("RGB"))
            self._provenance(provenance, record["crop_size_px"])
            vector = np.asarray(vector)
            if (vector.dtype != np.float32 or vector.shape != (512,) or not np.isfinite(vector).all() or
                    not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)):
                raise ValueError("Region encoder returned invalid normalized projected features")
            if self.load(identifier) != record:
                raise ValueError("Region source changed during encoding")
            buffer = io.BytesIO()
            np.save(buffer, vector, allow_pickle=False)
            data = buffer.getvalue()
            temporary = Path(tempfile.mkdtemp(prefix=".pending-embedding-", dir=directory.parent))
            try:
                (temporary / "embedding.npy").write_bytes(data)
                metadata = freeze(temporary / "embedding.json", {
                    "schema": "collection-region-embedding-v1", "created_at": now(),
                    "region_id": identifier, "region_sha256": record["sha256"],
                    "embedding_policy": EMBEDDING_POLICY, "embedding_policy_sha256": EMBEDDING_POLICY_SHA256,
                    "crop_sha256": record["crop_sha256"], "file_sha256": _sha(data),
                    "encoder_provenance": provenance, "representation": REPRESENTATION,
                    "feature_dimension": 512, "search_population": "whole images in the same collection",
                    "scope": "Crop appearance query only; retrieved images have no inferred matching coordinates"})
                os.rename(temporary, directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return vector, metadata

    def query(self, identifier, *, encode=True):
        record = self.load(identifier)
        vector, receipt = self.embedding(identifier) if encode else self._cached_embedding(record)
        return vector, {**record, "embedding_sha256": receipt["sha256"],
                        "embedding_file_sha256": receipt["file_sha256"],
                        "encoder_provenance": receipt["encoder_provenance"],
                        "role": "query reference only; not automatically selected finding evidence"}

    def evidence(self, identifier):
        record = self.load(identifier)
        return {"region_id": identifier, "region_sha256": record["sha256"],
                "image_id": record["image_id"], "image_sha256": record["image_sha256"],
                "image_url": record["image_url"], "source_image_url": record["source_image_url"],
                "bounds_px": record["bounds_px"], "source_size_px": record["source_size_px"],
                "crop_sha256": record["crop_sha256"], "prediction_version": record["prediction_version"],
                "coordinate_system": COORDINATES, "annotation_extent": "selected rectangular region"}
