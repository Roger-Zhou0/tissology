"""Bounded, versioned ingestion of user image tiles, separate from study labels.

The web service decodes images on its allocated CPU node. Model inference is an
explicit worker action; importing files never changes existing models or labels.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import tempfile
import time
import uuid
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .collection_index import CollectionIndex, checked, encode
from .collection_reviews import CollectionReviews, DISPOSITIONS, REVIEW_SCOPE
from .feedback_storage import freeze
from .prepare import CLASSES
from .storage import digest, now, read, write

CLASS_NAMES = {
    "ADI": "Adipose tissue", "BACK": "Background", "DEB": "Debris / necrotic tissue",
    "LYM": "Lymphocyte-rich tissue", "MUC": "Mucus", "MUS": "Smooth muscle",
    "NORM": "Normal colorectal mucosa", "STR": "Stroma",
    "TUM": "Colorectal tumor epithelium pattern",
}
LIMITS = {"max_images": 10000, "max_file_bytes": 25 * 1024 * 1024,
          "max_collection_bytes": 5 * 1024 * 1024 * 1024,
          "max_pixels": 4 * 1024 * 1024, "max_side": 4096,
          "min_side": 32, "max_directory_entries": 25000}
SCOPE = ("Preliminary whole-image screening against nine colorectal H&E tissue categories. "
         "Predicted image counts are not measured tissue areas, cell counts, diagnoses, or an audited result. "
         "Scores do not establish accuracy on this collection; unfamiliar tissue can receive confident predictions.")
_IDENTIFIER = re.compile(r"^[a-f0-9]{32,64}$")
_REVIEW_RESERVATION_WAIT_SECONDS = 3.0


class CollectionStore:
    def __init__(self, directory, import_roots=None, limits=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.import_roots = [Path(root).resolve() for root in (import_roots or [])]
        self.limits = {**LIMITS, **(limits or {})}
        self._locks = {}
        self._locks_guard = threading.Lock()
        self._indexes = {}
        self._index_guard = threading.RLock()

    def path(self, identifier):
        if not _IDENTIFIER.fullmatch(str(identifier)):
            raise FileNotFoundError("Unknown collection")
        path = self.directory / identifier
        if not (path / "collection.json").is_file():
            raise FileNotFoundError("Unknown collection")
        return path

    @contextmanager
    def locked(self, identifier):
        directory = self.path(identifier)
        with self._locks_guard:
            lock = self._locks.setdefault(identifier, threading.RLock())
        with lock:
            with (directory / ".lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield directory
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def review_reservation_locked(self):
        """Serialize cross-collection review publication with check reservations.

        Acquire before the collection lock, matching the teaching/check sampler
        order. A separate descriptor also excludes another thread in this process.
        Ordinary simultaneous reviews wait briefly for one another; long fitting
        operations produce a retryable response instead of tying up a web worker.
        """
        with (self.directory / ".teaching-worker.lock").open("a") as handle:
            deadline = time.monotonic() + _REVIEW_RESERVATION_WAIT_SECONDS
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("A teaching or check operation is running; retry this review when it finishes") from error
                    time.sleep(min(.025, remaining))
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _index(self, identifier):
        with self._index_guard:
            if identifier not in self._indexes:
                self._indexes[identifier] = CollectionIndex(self.path(identifier))
            return self._indexes[identifier].ensure()

    @staticmethod
    def _publish_bytes(path, data):
        """Publish a complete file exclusively; reuse a verified crash orphan."""
        if path.exists():
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
                raise ValueError("A stored collection image has changed; restore its original bytes before retrying")
            return
        descriptor, temporary = tempfile.mkstemp(prefix=".image-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    def validate_folder(self, folder):
        if not self.import_roots:
            raise ValueError("Folder import is not configured; upload image files instead")
        selected = Path(folder).expanduser()
        if selected.is_symlink():
            raise ValueError("Choose an ordinary directory, not a symbolic link")
        selected = selected.resolve(strict=True)
        if not selected.is_dir() or not any(selected.is_relative_to(root) for root in self.import_roots):
            raise ValueError("Folder must be inside one of the configured import roots")
        if selected == self.directory or selected.is_relative_to(self.directory):
            raise ValueError("Choose original image files, not the collection store")
        return selected

    def create(self, name, source_kind="upload", folder_path=None, declared_domain="unknown", auto_scan=False, client_request_id=None):
        params = {"name": name, "source_kind": source_kind, "folder_path": folder_path,
                  "declared_domain": declared_domain, "auto_scan": auto_scan}
        if client_request_id is None:
            return self._create(**params)
        if not re.fullmatch(r"[a-f0-9-]{32,64}", client_request_id):
            raise ValueError("Invalid collection creation request identity")
        requests = self.directory / ".create-requests"
        requests.mkdir(exist_ok=True)
        receipt_path = requests / (client_request_id + ".json")
        with (self.directory / ".create.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if receipt_path.exists():
                    receipt = checked(receipt_path)
                    if receipt["request_sha256"] != digest(params):
                        raise ValueError("Creation request identity already belongs to different collection settings")
                    identifier = receipt["collection_id"]
                    if (self.directory / identifier / "scan.json").exists():
                        return self.summary(identifier)
                else:
                    identifier = uuid.uuid4().hex
                    freeze(receipt_path, {"collection_id": identifier, "request_sha256": digest(params)})
                return self._create(**params, identifier=identifier)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _create(self, name, source_kind="upload", folder_path=None, declared_domain="unknown", auto_scan=False, identifier=None):
        if not name.strip() or len(name) > 120:
            raise ValueError("Collection name must contain 1–120 characters")
        if source_kind not in {"upload", "folder"}:
            raise ValueError("Use uploaded image files or an allowed Delta folder")
        if declared_domain not in {"colorectal_he", "unknown"}:
            raise ValueError("Choose colorectal H&E tiles or unknown / other images")
        if auto_scan and declared_domain != "colorectal_he":
            raise ValueError("Automatic screening requires researcher-declared colorectal H&E tiles")
        folder = self.validate_folder(folder_path) if source_kind == "folder" and folder_path else None
        if source_kind == "folder" and folder is None:
            raise ValueError("A folder path is required")
        if source_kind == "upload" and folder_path:
            raise ValueError("Uploaded collections do not accept a folder path")
        identifier = identifier or uuid.uuid4().hex
        directory = self.directory / identifier
        directory.mkdir(exist_ok=True)
        for child in ("originals", "images", "thumbnails"):
            (directory / child).mkdir(exist_ok=True)
        metadata = {
            "schema": "user-image-collection-v1", "id": identifier, "name": name.strip(),
            "created_at": now(), "source_kind": source_kind,
            "source_root": str(folder) if folder else None,
            "declared_domain": declared_domain, "role": "unlabeled_inspection", "auto_scan": bool(auto_scan),
            "scope": SCOPE, "limits": self.limits, "metadata_storage": "indexed-v2",
        }
        if not (directory / "collection.json").exists():
            freeze(directory / "collection.json", metadata)
        if not (directory / "intake.json").exists():
            write(directory / "intake.json", {"images": [], "errors": [], "duplicates": [],
                                          "imported_bytes": 0, "stored_bytes": 0, "attempted_files": 0})
        if not (directory / "scan.json").exists():
            write(directory / "scan.json", {"status": "importing", "processed": 0,
                                            "total": 0, "updated_at": now()})
        self._index(identifier)
        write(directory / "intake.json", {"images": [], "errors": [], "duplicates": [],
            "imported_bytes": 0, "stored_bytes": 0, "attempted_files": 0, "index_initialized": True}, overwrite=True)
        return self.summary(identifier)

    def _attempt(self, identifier, filename, *, row=None, reason=None, duplicate=False,
                 stored_bytes=0, upload_key=None, source_sha256=None, source_bytes=0):
        index = self._index(identifier)
        with index.connection() as connection:
            stats = index.get_state(connection, "stats")
            if stats["attempted_files"] >= self.limits["max_directory_entries"]:
                raise ValueError("Collection import attempt limit reached; create a new collection for remaining files")
            stats["attempted_files"] += 1
            stats["updated_at"] = now()
            if row is not None:
                connection.execute("INSERT INTO images(id,image_sha256,filename,payload) VALUES(?,?,?,?)",
                    (row["id"], row["image_sha256"], row["filename"], encode(row)))
                stats["image_count"] += 1
                stats["imported_bytes"] += row["source_bytes"]
                stats["stored_bytes"] += stored_bytes
                result = {"accepted": True, "duplicate": False, "image": self.public_image(identifier, row)}
                outcome = "accepted"
            else:
                kind = "duplicates" if duplicate else "errors"
                record = {"filename": filename, "reason": reason}
                connection.execute("INSERT INTO issues(kind,filename,payload) VALUES(?,?,?)", (kind, filename, encode(record)))
                stats["duplicate_count" if duplicate else "rejected_count"] += 1
                result = {"accepted": False, "duplicate": duplicate, **record}
                outcome = "duplicate" if duplicate else "rejected"
            index.put_state(connection, "stats", stats)
            if upload_key:
                receipt = {"upload_key": upload_key, "filename": filename, "source_sha256": source_sha256,
                           "source_bytes": source_bytes, "status": outcome, "image_id": row["id"] if row else None,
                           "reason": reason, "result": result}
                connection.execute("INSERT INTO uploads(upload_key,source_sha256,filename,source_bytes,payload) VALUES(?,?,?,?,?)",
                                   (upload_key, source_sha256, filename, source_bytes, encode(receipt)))
                result = {**result, "upload_key": upload_key}
        return result

    def add_file(self, identifier, filename, data, *, upload_key=None, source_sha256=None):
        filename = str(filename).replace("\x00", "")[:500] or "unnamed image"
        if upload_key is not None and not re.fullmatch(r"[a-f0-9]{64}", upload_key):
            raise ValueError("Upload identity must be a lowercase SHA256 key")
        actual_hash = hashlib.sha256(data).hexdigest()
        if source_sha256 is not None and source_sha256 != actual_hash:
            raise ValueError("Uploaded bytes do not match their source SHA256")
        with self.locked(identifier) as directory:
            index = self._index(identifier)
            if upload_key:
                prior = index.receipt(upload_key)
                if prior:
                    if prior["source_sha256"] != actual_hash or prior["filename"] != filename:
                        raise ValueError("Upload identity already belongs to different file bytes or a different filename")
                    return {**prior["result"], "upload_key": upload_key, "idempotent": True}
            status = read(directory / "scan.json")
            if status["status"] != "importing" or (directory / "manifest.json").exists():
                raise ValueError("This collection is frozen; create a new collection to add images")
            stats = index.statistics()
            if stats["attempted_files"] >= self.limits["max_directory_entries"]:
                raise ValueError("Collection import attempt limit reached; create a new collection for remaining files")
            def reject(reason, duplicate=False):
                return self._attempt(identifier, filename, reason=reason, duplicate=duplicate,
                    upload_key=upload_key, source_sha256=actual_hash, source_bytes=len(data))
            if stats["image_count"] >= self.limits["max_images"]:
                return reject("Collection image limit reached")
            if len(data) > self.limits["max_file_bytes"]:
                return reject("File exceeds the per-image byte limit")
            if stats["imported_bytes"] + len(data) > self.limits["max_collection_bytes"]:
                return reject("Collection byte limit reached")
            if Path(filename).suffix.lower() in {".svs", ".ndpi"}:
                return reject("Whole-slide files require a separate tiling workflow")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(data)) as source:
                        original_format, original_mode = source.format, source.mode
                        if original_format not in {"JPEG", "PNG", "TIFF"}:
                            raise ValueError("Only JPEG, PNG and single-page TIFF image tiles are supported")
                        if getattr(source, "n_frames", 1) != 1:
                            raise ValueError("Multipage or whole-slide images need a separate tiling workflow")
                        width, height = source.size
                        if min(width, height) < self.limits["min_side"]:
                            raise ValueError("Image is too small for tissue screening")
                        if max(width, height) > self.limits["max_side"] or width * height > self.limits["max_pixels"]:
                            raise ValueError("Image exceeds tile dimensions; tile whole-slide or large images first")
                        source.load()
                        image = ImageOps.exif_transpose(source).convert("RGB")
                encoded = io.BytesIO()
                image.save(encoded, "PNG")
                canonical = encoded.getvalue()
            except (ValueError, OSError, SyntaxError, UnidentifiedImageError,
                    Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
                reason = str(error) if isinstance(error, ValueError) else "Unreadable or unsupported image bytes"
                return reject(reason)
            image_hash = hashlib.sha256(canonical).hexdigest()
            image_id = image_hash[:32]
            with index.connection() as connection:
                duplicate = connection.execute("SELECT 1 FROM images WHERE image_sha256=?", (image_hash,)).fetchone()
            if duplicate:
                return reject("Duplicate decoded image; counted once", True)
            thumbnail = image.copy()
            thumbnail.thumbnail((320, 320))
            thumbnail_buffer = io.BytesIO()
            thumbnail.save(thumbnail_buffer, "JPEG", quality=88)
            thumbnail_bytes = thumbnail_buffer.getvalue()
            stored_bytes = len(data) + len(canonical) + len(thumbnail_bytes)
            if stats["stored_bytes"] + stored_bytes > self.limits["max_collection_bytes"]:
                return reject("Collection storage limit reached (originals, decoded images and thumbnails)")
            source_path = directory / "originals" / actual_hash
            image_path = directory / "images" / f"{image_id}.png"
            thumbnail_path = directory / "thumbnails" / f"{image_id}.jpg"
            self._publish_bytes(source_path, data)
            self._publish_bytes(image_path, canonical)
            self._publish_bytes(thumbnail_path, thumbnail_bytes)
            width, height = image.size
            row = {"id": image_id, "filename": filename, "original_filename": filename,
                   "path": str(image_path), "source_path": str(source_path),
                   "thumbnail_path": str(thumbnail_path), "sha256": actual_hash,
                   "image_sha256": image_hash, "width": width, "height": height,
                   "size_px": [width, height], "source_format": original_format, "original_mode": original_mode,
                   "source_bytes": len(data), "role": "unlabeled", "imported_at": now()}
            result = self._attempt(identifier, filename, row=row, stored_bytes=stored_bytes,
                upload_key=upload_key, source_sha256=actual_hash, source_bytes=len(data))
            if stats["image_count"] % 25 == 0:
                status.update(total=stats["image_count"] + 1, updated_at=now())
                write(directory / "scan.json", status, overwrite=True)
            return result

    def record_rejection(self, identifier, filename, reason):
        filename = str(filename).replace("\x00", "")[:500] or "unnamed image"
        reason = str(reason).replace("\x00", "")[:2000]
        with self.locked(identifier) as directory:
            if read(directory / "scan.json")["status"] != "importing":
                raise ValueError("Collection is no longer importing")
            return self._attempt(identifier, filename, reason=reason)

    def acquire_admission(self, kind, count):
        """Bound concurrent work across web processes before request buffering."""
        if kind not in {"upload", "folder"}:
            raise ValueError("Invalid admission kind")
        for slot in range(count):
            handle = (self.directory / f".{kind}-slot-{slot}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                handle.close()
        raise ValueError("Import capacity is busy; retry shortly")

    def resume_import(self, identifier):
        directory = self.path(identifier)
        with (directory / ".import-worker.lock").open("a") as worker:
            try:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("This folder import is still running")
            try:
                with self.locked(identifier):
                    status = read(directory / "scan.json")
                    if (directory / "manifest.json").exists() or status["status"] != "failed":
                        raise ValueError("Only an interrupted, unfrozen import can resume")
                    status.pop("import_started_at", None)
                    status.update(status="importing", error=None, import_requested_at=now(), updated_at=now())
                    write(directory / "scan.json", status, overwrite=True)
            finally:
                fcntl.flock(worker, fcntl.LOCK_UN)
        return self.summary(identifier)

    def import_folder(self, identifier, admission=None):
        """Only one folder decoder runs per store, across web processes."""
        admission = admission or self.acquire_admission("folder", 1)
        try:
            self._run_folder_import(identifier)
        finally:
            admission.close()

    def _run_folder_import(self, identifier):
        """Hold an independent lifetime lock so abandoned imports are detectable."""
        directory = self.path(identifier)
        with (directory / ".import-worker.lock").open("a") as worker:
            try:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                with self.locked(identifier):
                    status = read(directory / "scan.json")
                    if status["status"] != "importing":
                        return
                    status.update(import_started_at=now(), updated_at=now())
                    write(directory / "scan.json", status, overwrite=True)
                self._import_folder(identifier)
            finally:
                fcntl.flock(worker, fcntl.LOCK_UN)

    def _import_folder(self, identifier):
        """Read allowed image files in place, copy only this bounded selection."""
        directory = self.path(identifier)
        metadata = read(directory / "collection.json")
        try:
            folder = self.validate_folder(metadata["source_root"])
            entries = 0
            stop = False
            for parent, subdirs, files in os.walk(folder, followlinks=False):
                # Deterministic traversal; never follow directory or file symlinks.
                subdirs[:] = sorted(d for d in subdirs if not (Path(parent) / d).is_symlink()
                                    and not (Path(parent) / d).resolve().is_relative_to(self.directory))
                entries += len(subdirs)
                if entries > self.limits["max_directory_entries"]:
                    self.record_rejection(identifier, "(remaining directories)", "Directory traversal limit reached; use a smaller folder")
                    break
                for name in sorted(files):
                    entries += 1
                    path = Path(parent) / name
                    if entries > self.limits["max_directory_entries"]:
                        self.record_rejection(identifier, "(remaining directory entries)", "Directory traversal limit reached; use a smaller folder")
                        stop = True
                        break
                    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs", ".ndpi"}:
                        continue
                    relative = str(path.relative_to(folder))
                    index = self._index(identifier)
                    if index.seen_filename(relative):
                        continue
                    stats = index.statistics()
                    if stats["image_count"] >= self.limits["max_images"] or stats["stored_bytes"] >= self.limits["max_collection_bytes"]:
                        self.record_rejection(identifier, "(remaining files)", "Collection limit reached; remaining files were not imported")
                        stop = True
                        break
                    try:
                        resolved = path.resolve(strict=True)
                        if path.is_symlink() or not resolved.is_relative_to(folder):
                            raise ValueError("Symbolic links are not imported")
                        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                        with os.fdopen(descriptor, "rb") as handle:
                            info = os.fstat(handle.fileno())
                            if not stat.S_ISREG(info.st_mode):
                                raise ValueError("Only ordinary image files are supported")
                            data = handle.read(self.limits["max_file_bytes"] + 1)
                        self.add_file(identifier, relative, data)
                    except (OSError, ValueError) as error:
                        self.record_rejection(identifier, relative, str(error))
                if stop:
                    break
            self.finalize(identifier)
        except Exception as error:
            self.fail_scan(identifier, "Folder import failed: " + str(error))

    def finalize(self, identifier):
        self.reconcile_publications(identifier)
        with self.locked(identifier) as directory:
            status = read(directory / "scan.json")
            if (directory / "manifest.json").exists():
                return self.summary(identifier)
            if status["status"] != "importing":
                raise ValueError("Only an importing collection can be finalized")
            intake = self._index(identifier).intake()
            if not intake["images"]:
                status.update(status="failed", error="No supported unique images were imported", updated_at=now())
                write(directory / "scan.json", status, overwrite=True)
                return self.summary(identifier)
            metadata = read(directory / "collection.json")
            write(directory / "intake.json", intake, overwrite=True)
            freeze(directory / "manifest.json", {
                "schema": "user-image-manifest-v1", "collection_id": identifier,
                "created_at": now(), "declared_domain": metadata["declared_domain"],
                "source_kind": metadata["source_kind"], "role": "unlabeled_inspection",
                "images": intake["images"], "exclusions": intake["errors"],
                "duplicates": intake["duplicates"], "image_count": len(intake["images"]),
                "selection": "All accepted unique decoded tiles within explicit import limits; no reference labels used",
                "transformation": "Decode single-page image, apply EXIF orientation, convert RGB, freeze lossless PNG; original bytes retained",
            })
            status.update(status="ready", total=len(intake["images"]), updated_at=now())
            write(directory / "scan.json", status, overwrite=True)
        return self.summary(identifier)

    def manifest(self, identifier):
        return read(self.path(identifier) / "manifest.json")

    def request_scan(self, identifier, submitter):
        self.reconcile_publications(identifier)
        if submitter is None:
            raise ValueError("Scheduled scanning is not configured for this server")
        directory = self.path(identifier)
        with (directory / ".scan-submission.lock").open("a") as submission:
            try:
                fcntl.flock(submission, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self.summary(identifier)
            try:
                return self._submit_scan(identifier, submitter)
            finally:
                fcntl.flock(submission, fcntl.LOCK_UN)

    def _submit_scan(self, identifier, submitter):
        from .collection_scheduler import SubmissionUncertain
        with self.locked(identifier) as directory:
            metadata = read(directory / "collection.json")
            if metadata["declared_domain"] != "colorectal_he":
                raise ValueError("The available model supports colorectal H&E tiles only; this collection remains available for browsing")
            status = read(directory / "scan.json")
            if status.get("retry_blocked"):
                raise ValueError("Previous submission is unresolved; check its recorded attempt in the compute queue before retrying")
            if status["status"] in {"queued", "scanning", "complete"}:
                return self.summary(identifier)
            if status["status"] not in {"ready", "failed"} or not (directory / "manifest.json").exists():
                raise ValueError("Finish importing images before starting a scan")
            status.pop("job_id", None)
            status.pop("scheduler", None)
            attempt = uuid.uuid4().hex
            status.update(status="queued", processed=0, queued_at=now(), updated_at=now(), error=None,
                          submission_attempt=attempt, retry_blocked=False)
            write(directory / "scan.json", status, overwrite=True)
        acknowledged = False
        try:
            submitted = submitter.submit_attempt(identifier, attempt) if callable(getattr(submitter, "submit_attempt", None)) else submitter(identifier)
            acknowledged = True
            if isinstance(submitted, str):
                submitted = {"job_id": submitted}
            if not isinstance(submitted, dict) or not str(submitted.get("job_id", "")).isdigit():
                raise SubmissionUncertain("Scheduler did not return a job identifier")
            freeze(directory / "submissions" / (attempt + ".json"), {
                "collection_id": identifier, "submission_attempt": attempt, "job_id": str(submitted["job_id"]), "created_at": now()})
            with self.locked(identifier):
                status = read(directory / "scan.json")
                status.update(job_id=str(submitted["job_id"]), scheduler=submitted, updated_at=now(), retry_blocked=False)
                write(directory / "scan.json", status, overwrite=True)
        except Exception as error:
            with self.locked(identifier):
                status = read(directory / "scan.json")
                if status["status"] not in {"scanning", "complete"}:
                    uncertain = acknowledged or isinstance(error, SubmissionUncertain)
                    status.update(status="failed", error="Could not confirm scan submission: " + str(error),
                                  retry_blocked=uncertain, updated_at=now())
                    write(directory / "scan.json", status, overwrite=True)
        return self.summary(identifier)

    def claim_scan(self, identifier, model=None):
        self.reconcile_publications(identifier)
        with self.locked(identifier) as directory:
            metadata = read(directory / "collection.json")
            if metadata["declared_domain"] != "colorectal_he":
                raise ValueError("The model is restricted to researcher-declared colorectal H&E tiles")
            status = read(directory / "scan.json")
            if status["status"] not in {"ready", "queued", "failed"} or not (directory / "manifest.json").exists():
                raise ValueError("Collection is not available for a new scan")
            if (directory / "predictions.json").exists():
                raise ValueError("A completed prediction artifact already exists")
            actual_job = str((model or {}).get("slurm_job_id", ""))
            if actual_job.isdigit():
                status["job_id"] = actual_job
            status.update(status="scanning", processed=0, model=model or {}, started_at=now(), updated_at=now(), error=None, retry_blocked=False)
            write(directory / "scan.json", status, overwrite=True)
        return self.manifest(identifier)

    def scan_progress(self, identifier, processed, *, failed_images=0, message=None):
        with self.locked(identifier) as directory:
            status = read(directory / "scan.json")
            if status["status"] != "scanning" or not status["processed"] <= processed <= status["total"]:
                raise ValueError("Invalid scan progress")
            status.update(processed=int(processed), failed_images=int(failed_images), message=message, updated_at=now())
            write(directory / "scan.json", status, overwrite=True)

    def finish_scan(self, identifier, predictions, model=None, score_type="Uncalibrated classifier scores"):
        with self.locked(identifier) as directory:
            status = read(directory / "scan.json")
            if status["status"] != "scanning":
                raise ValueError("Collection has no running scan")
            manifest = self.manifest(identifier)
            ids = [row["id"] for row in predictions]
            if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in manifest["images"]}:
                raise ValueError("Predictions must cover every frozen image exactly once")
            counts = {label: 0 for label in CLASSES}
            for row in predictions:
                if row.get("predicted_label") not in CLASSES:
                    raise ValueError("Unknown predicted tissue category")
                top = row.get("top_k", [])
                if not top or top[0].get("label") != row["predicted_label"]:
                    raise ValueError("Top-ranked category does not match the prediction")
                if len({part.get("label") for part in top}) != len(top):
                    raise ValueError("Repeated ranked category")
                for part in top:
                    score = part.get("score")
                    if part.get("label") not in CLASSES or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid prediction score")
                if any(top[i]["score"] < top[i+1]["score"] for i in range(len(top)-1)):
                    raise ValueError("Prediction scores must be in descending order")
                for key in ("confidence", "margin"):
                    value = row.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid confidence or margin")
                if not math.isclose(row["confidence"], top[0]["score"], rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError("Confidence does not match the top classifier score")
                if len(top) >= 2 and not math.isclose(row["margin"], top[0]["score"] - top[1]["score"], rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError("Margin does not match the two leading classifier scores")
                if not isinstance(row.get("uncertain"), bool):
                    raise ValueError("Prediction needs an explicit review-priority flag")
                counts[row["predicted_label"]] += 1
            summary = {"counts": counts, "predicted_count": len(predictions),
                       "uncertain_count": sum(row["uncertain"] for row in predictions),
                       "denominator": "Accepted unique images; each image counts once"}
            artifact = freeze(directory / "predictions.json", {
                "schema": "preliminary-collection-predictions-v1", "collection_id": identifier,
                "manifest_sha256": manifest["sha256"], "created_at": now(),
                "model": model or status.get("model", {}), "score_type": score_type,
                "scope": SCOPE, "predictions": predictions, "summary": summary,
            })
            status.update(status="complete", processed=len(predictions), completed_at=now(), updated_at=now(),
                          prediction_sha256=artifact["sha256"], model=artifact["model"], error=None)
            write(directory / "scan.json", status, overwrite=True)
        return self.summary(identifier)

    def fail_scan(self, identifier, message):
        with self.locked(identifier) as directory:
            status = read(directory / "scan.json")
            if status["status"] == "complete":
                return
            status.update(status="failed", error=str(message)[:2000], updated_at=now())
            write(directory / "scan.json", status, overwrite=True)

    def reconcile_publications(self, identifier):
        """Recover the small window between artifact publication and status save."""
        index = self._index(identifier)
        indexed = index.metadata()
        prediction = indexed["prediction"]
        with self.locked(identifier) as directory:
            status = read(directory / "scan.json")
            if prediction and status["status"] != "complete":
                status.update(status="complete", processed=prediction["summary"]["predicted_count"],
                    completed_at=prediction["created_at"], updated_at=now(),
                    prediction_sha256=prediction["sha256"], model=prediction["model"], error=None)
                write(directory / "scan.json", status, overwrite=True)
            elif "manifest.json" in indexed["bindings"] and (status["status"] == "importing" or
                    (status["status"] == "failed" and not status.get("started_at") and not status.get("queued_at"))):
                status.update(status="ready", total=indexed["stats"]["image_count"], updated_at=now(), error=None)
                write(directory / "scan.json", status, overwrite=True)

    def reconcile_import(self, identifier):
        """Expose interrupted CPU background imports without failing live workers."""
        directory = self.path(identifier)
        metadata = read(directory / "collection.json")
        status = read(directory / "scan.json")
        if metadata["source_kind"] != "folder" or status["status"] != "importing":
            return
        created = datetime.fromisoformat(status.get("import_requested_at", metadata["created_at"]))
        age = (datetime.now(timezone.utc) - created).total_seconds()
        # A new background task has not necessarily acquired its worker lock yet.
        if not status.get("import_started_at") and age < 60:
            return
        with (directory / ".import-worker.lock").open("a") as worker:
            try:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                with self.locked(identifier):
                    current = read(directory / "scan.json")
                    if current["status"] != "importing":
                        return
                    if not current.get("import_started_at") and age < 60:
                        return
                    current.update(status="failed", updated_at=now(),
                                   error="Import interrupted; resume the remaining files. Accepted images remain browsable and their original copies are retained.")
                    write(directory / "scan.json", current, overwrite=True)
            finally:
                fcntl.flock(worker, fcntl.LOCK_UN)

    def reconcile_submission(self, identifier, scheduler):
        directory = self.path(identifier)
        status = read(directory / "scan.json")
        if status.get("job_id") or (status["status"] != "queued" and not status.get("retry_blocked")):
            return
        with (directory / ".scan-submission.lock").open("a") as submission:
            try:
                fcntl.flock(submission, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                attempt = status.get("submission_attempt")
                recovered = None
                if attempt:
                    receipt = directory / "submissions" / (attempt + ".json")
                    if receipt.exists():
                        value = checked(receipt)
                        if value.get("collection_id") != identifier or value.get("submission_attempt") != attempt:
                            raise ValueError("Submission receipt belongs to a different attempt")
                        recovered = str(value["job_id"])
                    elif callable(getattr(scheduler, "recover_attempt", None)):
                        found = scheduler.recover_attempt(identifier, attempt, status["queued_at"])
                        if found.get("known"):
                            recovered = str(found["job_id"])
                with self.locked(identifier):
                    current = read(directory / "scan.json")
                    if current.get("job_id") or current.get("submission_attempt") != attempt or current["status"] in {"scanning", "complete"}:
                        return
                    if recovered and recovered.isdigit():
                        current.update(status="queued", job_id=recovered, retry_blocked=False, error=None, updated_at=now())
                    else:
                        current.update(status="failed", retry_blocked=True, updated_at=now(),
                            error="Submission acknowledgement was interrupted. The job may already exist; automatic retry is blocked. Ask the compute administrator to check attempt " + str(attempt or "legacy unrecorded attempt") + ". Images and completed work are preserved; refresh to reconcile scheduler records.")
                    write(directory / "scan.json", current, overwrite=True)
            finally:
                fcntl.flock(submission, fcntl.LOCK_UN)

    def reconcile_scan(self, identifier, scheduler):
        """A killed Slurm worker cannot write its failure; reflect scheduler truth."""
        self.reconcile_submission(identifier, scheduler)
        if scheduler is None or not callable(getattr(scheduler, "status", None)):
            return
        status = read(self.path(identifier) / "scan.json")
        if status["status"] not in {"queued", "scanning"} or not status.get("job_id"):
            return
        try:
            scheduled = scheduler.status(status["job_id"])
        except Exception:
            # Scheduler unavailability does not prove a job failed.
            return
        if not scheduled.get("known") or not scheduled.get("terminal"):
            return
        with self.locked(identifier) as directory:
            current = read(directory / "scan.json")
            if current["status"] not in {"queued", "scanning"} or current.get("job_id") != status["job_id"]:
                return
            current.update(status="failed", updated_at=now(),
                           error="Scheduled scan ended with state " + str(scheduled.get("state", "unknown")) + " before a complete report was saved. Retry this scan to resume verified work.")
            write(directory / "scan.json", current, overwrite=True)

    def prediction_view(self, identifier, prediction_version=None):
        """Resolve a displayed version without modifying the original scan/index."""
        manager = getattr(self, "teaching", None)
        if prediction_version is None:
            prediction_version = manager.active_version(identifier)["id"] if manager else "original"
        if prediction_version == "original":
            original = self._index(identifier).metadata()["prediction"]
            return None, {"id": "original", "name": "Original scan",
                          "prediction_sha256": original.get("sha256")}
        if manager is None:
            raise ValueError("Collection prediction versions are not configured")
        artifact = manager.predictions(identifier, prediction_version)
        return artifact, {"id": prediction_version,
                          "name": artifact.get("model", {}).get("name") or "Taught version " + prediction_version[:8],
                          "prediction_sha256": artifact["sha256"],
                          "model": artifact.get("model")}

    def summary(self, identifier, prediction_version=None):
        directory = self.path(identifier)
        metadata, status = read(directory / "collection.json"), read(directory / "scan.json")
        index = self._index(identifier)
        indexed = index.metadata()
        stats, prediction = indexed["stats"], indexed["prediction"]
        selected_artifact, selected_version = self.prediction_view(identifier, prediction_version)
        if selected_artifact is not None:
            prediction = selected_artifact
        if status["status"] == "importing":
            status = {**status, "total": stats["image_count"], "updated_at": stats.get("updated_at", status["updated_at"])}
        resumable = "manifest.json" not in indexed["bindings"] and (status["status"] == "failed" or
            (metadata["source_kind"] == "upload" and status["status"] == "importing"))
        return {"id": identifier, "name": metadata["name"], "created_at": metadata["created_at"],
                "source_kind": metadata["source_kind"], "declared_domain": metadata["declared_domain"], "auto_scan": metadata.get("auto_scan", False),
                "status": status["status"], **{key: stats[key] for key in ("image_count", "duplicate_count", "rejected_count", "attempted_files", "imported_bytes", "stored_bytes")},
                "scan": status, "summary": prediction.get("summary"), "model": prediction.get("model", status.get("model")),
                "score_type": prediction.get("score_type"), "scope": SCOPE,
                "errors": index.issues("errors", 100), "duplicates": index.issues("duplicates", 100),
                "manifest_sha256": indexed["bindings"].get("manifest.json"),
                "prediction_sha256": prediction.get("sha256"), "import_resumable": resumable,
                "selected_prediction_version": selected_version,
                "review_summary": CollectionReviews(directory).summary(),
                "metadata_storage": "indexed-v2",
                "resume_note": "Resuming retains previously accepted copies and recorded rejections; changed or previously rejected files require a new collection."}

    def list(self):
        identifiers = [path.name for path in self.directory.iterdir() if _IDENTIFIER.fullmatch(path.name) and (path / "collection.json").exists() and (path / "scan.json").exists()]
        return sorted([self.summary(identifier) for identifier in identifiers], key=lambda row: row["created_at"], reverse=True)

    def public_image(self, identifier, row, prediction=None, review=None):
        return {key: row.get(key) for key in ("id", "filename", "width", "height", "source_format", "original_mode", "source_bytes", "sha256", "image_sha256", "role")} | {
            "image_url": f"api/collections/{identifier}/images/{row['id']}/image",
            "thumbnail_url": f"api/collections/{identifier}/images/{row['id']}/thumbnail",
            **(prediction or {}), "prediction": prediction, "review": review,
        }

    def images(self, identifier, offset=0, limit=40, label=None, uncertain=None, q=None, review="all", prediction_version=None):
        if label is not None and label not in CLASSES:
            raise ValueError("Unknown tissue category")
        if offset < 0 or limit < 1 or limit > self.limits["max_images"]:
            raise ValueError("Invalid image page")
        artifact, selected = self.prediction_view(identifier, prediction_version)
        observations = CollectionReviews(self.path(identifier))
        if artifact is None:
            rows, total = observations.page(self._index(identifier), offset, limit, label, uncertain, q, review)
            images = [self.public_image(identifier, row, prediction, observation) for row, prediction, observation in rows]
        else:
            # Apply class/priority filters to the selected immutable version,
            # while using the existing indexed filename and review filters.
            predictions = {p["id"]: p for p in artifact["predictions"]}
            rows, _ = observations.page(self._index(identifier), 0, self.limits["max_images"], None, None, q, review)
            selected_rows = [(row, original, observation) for row, original, observation in rows
                             if (label is None or predictions[row["id"]]["predicted_label"] == label)
                             and (uncertain is None or predictions[row["id"]]["uncertain"] == uncertain)]
            total = len(selected_rows)
            images = [{**self.public_image(identifier, row, predictions[row["id"]], observation),
                       "original_prediction": original}
                      for row, original, observation in selected_rows[offset:offset + limit]]
        return {"images": images, "total": total, "offset": offset, "limit": limit,
                "selected_prediction_version": selected}

    def image(self, identifier, image_id):
        return self._index(identifier).image(image_id)

    def image_detail(self, identifier, image_id, prediction_version=None):
        index = self._index(identifier)
        row = index.image(image_id)
        latest, history = CollectionReviews(self.path(identifier)).image(image_id)
        artifact, selected = self.prediction_view(identifier, prediction_version)
        original = index.prediction(image_id)
        prediction = next(p for p in artifact["predictions"] if p["id"] == image_id) if artifact else original
        manager = getattr(self, "teaching", None)
        return {**self.public_image(identifier, row, prediction, latest),
                "original_prediction": original, "selected_prediction_version": selected,
                "teaching_check_reserved": bool(manager and manager.is_reserved(identifier, image_id)),
                "review_history": history, "review_scope": REVIEW_SCOPE}

    def save_review(self, identifier, image_id, *, reviewer, disposition, label=None, note="",
                    expected_revision=0, client_request_id, prediction_version="original"):
        reviewer, note = reviewer.strip(), note.strip()
        if not reviewer or len(reviewer) > 120 or len(note) > 2000:
            raise ValueError("Enter a reviewer name (up to 120 characters) and a note up to 2,000 characters")
        if disposition not in DISPOSITIONS or (label is not None and label not in CLASSES):
            raise ValueError("Choose Confirm, Correct label, or Unsure and a supported tissue category")
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool) or expected_revision < 0:
            raise ValueError("Invalid review revision")
        if not re.fullmatch(r"[a-f0-9-]{32,64}", client_request_id):
            raise ValueError("Invalid review save identity")
        with self.review_reservation_locked(), self.locked(identifier) as directory:
            index = self._index(identifier)
            image = index.image(image_id)
            manager = getattr(self, "teaching", None)
            if manager is not None and manager.is_reserved(identifier, image_id):
                raise ValueError("This image is reserved for a model check. Label it in that check, not as a teaching example.")
            original_prediction = index.prediction(image_id)
            metadata = index.metadata()
            artifact, selected = self.prediction_view(identifier, prediction_version)
            prediction = next(p for p in artifact["predictions"] if p["id"] == image_id) if artifact else original_prediction
            if not prediction or not metadata["prediction"].get("sha256"):
                raise ValueError("Wait for a completed scan before reviewing a prediction")
            if disposition == "confirmed":
                if prediction.get("predicted_label") not in CLASSES or prediction.get("error"):
                    raise ValueError("This image has no successful prediction to confirm")
                if label is not None and label != prediction["predicted_label"]:
                    raise ValueError("A confirmed label must match the displayed prediction version")
                label = prediction["predicted_label"]
            elif disposition == "corrected":
                if label not in CLASSES or label == prediction.get("predicted_label"):
                    raise ValueError("Choose a different tissue category for the correction")
            elif label is not None:
                raise ValueError("An unsure observation cannot assign a tissue category")
            request = {"reviewer": reviewer, "disposition": disposition, "label": label, "note": note,
                       "expected_revision": expected_revision, "client_request_id": client_request_id}
            if selected["id"] != "original":
                request["prediction_version"] = selected["id"]
            model = (artifact or metadata["prediction"]).get("model")
            # A large scan can contain hundreds of feature-chunk receipts. Bind
            # each observation to that full record by hash, but store it once in
            # the prediction artifact/export rather than once per reviewed image.
            model_identity = ({key: model[key] for key in (
                "name", "revision", "classifier_sha256", "configuration_sha256",
                "checkpoint_sha256", "coefficients_sha256", "code_revision", "encoder_identity")
                if key in model} if isinstance(model, dict) else model)
            event = CollectionReviews(directory).save(request, {
                "collection_id": identifier, "image_id": image_id, "filename": image["filename"],
                "image_sha256": image["image_sha256"], "source_sha256": image["sha256"],
                "manifest_sha256": metadata["bindings"].get("manifest.json"),
                "prediction_sha256": selected["prediction_sha256"],
                "original_prediction": original_prediction, "reviewed_prediction": prediction,
                "original_prediction_sha256": metadata["prediction"]["sha256"],
                "prediction_version": selected["id"], "model": model_identity,
                "model_provenance_sha256": digest(model),
            })
        return {"review": event, "review_summary": CollectionReviews(directory).summary(), "scope": REVIEW_SCOPE}

    def export_reviews(self, identifier):
        with self.locked(identifier):
            return self._export_reviews(identifier, self.summary(identifier, "original"))

    def _export_reviews(self, identifier, summary):
        result = {"schema": "collection-review-export-v1", "exported_at": now(),
                  "collection_id": identifier, "collection_name": summary["name"],
                  "manifest_sha256": summary["manifest_sha256"], "prediction_sha256": summary["prediction_sha256"],
                  "model": summary["model"], "model_provenance_sha256": digest(summary["model"]),
                  "summary": summary["review_summary"], "scope": REVIEW_SCOPE,
                  **CollectionReviews(self.path(identifier)).export()}
        return {**result, "sha256": digest(result)}

    def uploads(self, identifier, offset=0, limit=500):
        result = self._index(identifier).receipts(offset, limit)
        result["uploads"] = [{key: value for key, value in row.items() if key != "result"} for row in result["uploads"]]
        return result

    def image_path(self, identifier, image_id, thumbnail=False):
        row = self.image(identifier, image_id)
        expected = self.path(identifier) / ("thumbnails" if thumbnail else "images") / (image_id + (".jpg" if thumbnail else ".png"))
        # Serve only our generated, identified files, never a caller-selected path.
        if expected.is_symlink() or expected.resolve().parent != expected.parent or not expected.is_file():
            raise FileNotFoundError("Unknown collection image")
        return expected

    def export(self, identifier, prediction_version=None):
        with self.locked(identifier):
            summary = self.summary(identifier, prediction_version)
            version = summary["selected_prediction_version"]["id"]
            result = {"schema": "preliminary-collection-export-v1", "exported_at": now(),
                    "collection": summary, "images": self.images(identifier, 0, self.limits["max_images"], prediction_version=version)["images"],
                    "scope": SCOPE, "researcher_reviews": self._export_reviews(identifier, summary)}
            if version != "original":
                result["teaching"] = self.teaching.export(identifier, version)
            return result
