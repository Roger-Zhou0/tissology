"""Optional collection-local teaching using frozen CONCH features and fresh checks.

Inspection reviews are never relabelled in place. Opting in creates an immutable
development snapshot; fresh probability-sampled check labels stay out of fitting.
The original scan and every candidate prediction artifact remain immutable.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import uuid
import warnings

import numpy as np

from .collection_reviews import CollectionReviews
from .collection_lineage import (IDENTITY_SCOPE, identities, known_development_receipt,
                                 verify_development_receipt)
from .collection_scan import (ARTIFACT_ROOT, CLASSIFIER_SHA256, CLASSES, SCORE_TYPE,
                              image_binding, prediction_records, score_features)
from .collection_storage import CLASS_NAMES, SCOPE as SCAN_SCOPE
from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, MODEL, file_hash
from .feedback_storage import freeze
from .storage import digest, now, read
from .study import verify
from .execution import execution_provenance, require_cpu_execution

SCOPE = ("Collection-specific nine-category colorectal tile classification. Researcher-selected "
         "development labels train a lightweight classifier on cached CONCH features. A fresh "
         "uniform check estimates accuracy only in its recorded never-reviewed image frame; "
         "it does not establish patient-level generalization or measured tissue area.")
GATE = ("All planned checks must be labelled and the candidate must have at least as many "
        "correct labels as the current version on that check. This operational guard is not "
        "evidence of statistical superiority or noninferiority.")
REPEATED_SCOPE = ("Uncertainty statements apply to this recorded check. They are not adjusted "
                  "for choosing or stopping across repeated candidates; preserve the full version history.")
RECIPE = {"C": 10., "solver": "lbfgs", "max_iter": 2000, "seed": 20261018,
          "source_weight": 5400., "development_weight": 5400., "classes": CLASSES}
_ID = re.compile(r"^[a-f0-9]{32}$")


def checked(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("A required teaching artifact is missing or a symbolic link")
    value = read(path)
    verify(value)
    return value


def reviewer_name(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise ValueError("Enter a reviewer name, up to 120 characters")
    return value.strip()


def require_teaching_cpu():
    try:
        return require_cpu_execution()
    except RuntimeError as error:
        raise ValueError(str(error)) from error


def source_files():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ("collection_teaching.py", "collection_teaching_stats.py", "collection_lineage.py", "collection_scan.py",
             "feedback_storage.py", "fusion_development.py", "execution.py")}


class CollectionFeatures:
    """Load pooled classifier features only after verifying every scan receipt."""
    def __init__(self, collections, collection_id):
        self.root = collections.path(collection_id)
        self.manifest = checked(self.root / "manifest.json")
        self.prediction = checked(self.root / "predictions.json")
        scan = read(self.root / "scan.json")
        if (scan.get("status") != "complete" or
                scan.get("prediction_sha256") != self.prediction["sha256"] or
                self.manifest.get("collection_id") != collection_id or
                self.manifest.get("role") != "unlabeled_inspection" or
                self.prediction.get("collection_id") != collection_id or
                self.prediction.get("manifest_sha256") != self.manifest["sha256"]):
            raise ValueError("Finish a verified collection scan before teaching it")
        self.rows = self.manifest["images"]
        self.ids = [row["id"] for row in self.rows]
        if not self.rows or len(self.rows) > 10000 or len(set(self.ids)) != len(self.ids):
            raise ValueError("Teaching requires a bounded collection of unique images")
        model = self.prediction.get("model", {})
        identity = model.get("encoder_identity", {})
        if (model.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                model.get("code_revision") != CODE_REVISION or
                identity.get("model") != MODEL or identity.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                model.get("manifest_sha256") != self.manifest["sha256"]):
            raise ValueError("Teaching requires the pinned CONCH collection feature cache")
        binding = {"manifest_sha256": self.manifest["sha256"],
                   "configuration_sha256": model.get("configuration_sha256"),
                   "encoder_identity_sha256": digest(identity),
                   "worker_source_sha256": model.get("worker_source_sha256")}
        arrays, offset = [], 0
        self.chunks = model.get("feature_chunks", [])
        for index, receipt in enumerate(self.chunks):
            name = f"batch-{index:06d}.npz"
            path = self.root / "features" / name
            meta = checked(path.with_suffix(".json"))
            count = len(meta.get("images", []))
            rows = self.rows[offset:offset + count]
            if (receipt.get("file") != name or meta.get("file") != name or not count or
                    meta.get("binding") != binding or meta.get("images") != image_binding(rows) or
                    meta["sha256"] != receipt.get("metadata_sha256") or path.is_symlink() or
                    meta.get("file_sha256") != receipt.get("sha256") or file_hash(path) != receipt["sha256"]):
                raise ValueError("Collection feature cache differs from its frozen scan")
            with np.load(path, allow_pickle=False) as cache:
                values = cache["classifier_embeddings"].copy()
                if (set(cache.files) != {"ids", "classifier_embeddings", "embeddings"} or
                        cache["ids"].tolist() != [row["id"] for row in rows] or
                        values.shape != (count, 512) or values.dtype != np.float32 or
                        not np.isfinite(values).all() or
                        not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)):
                    raise ValueError("Invalid pooled classifier feature cache")
            arrays.append(values)
            offset += count
        if offset != len(self.rows):
            raise ValueError("Completed classifier features are unavailable for this collection")
        self.features = np.concatenate(arrays)


class CollectionTeachingManager:
    def __init__(self, collections, replay_directory=None, executor=None):
        self.collections = collections
        self.replay_directory = Path(replay_directory or os.environ.get("TISSUE_REPLAY_DIR") or
                                     ARTIFACT_ROOT / "conch-experiment-001")
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="collection-teaching")
        self._process_lock = threading.Lock()

    def directory(self, collection_id):
        path = self.collections.path(collection_id) / "teaching"
        if path.is_symlink():
            raise ValueError("Teaching records must remain inside their collection")
        return path

    def folder(self, collection_id, candidate_id):
        if not isinstance(candidate_id, str) or not _ID.fullmatch(candidate_id):
            raise ValueError("Unknown teaching version")
        path = self.directory(collection_id) / "candidates" / candidate_id
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Teaching version storage changed")
        return path

    def _plans(self, collection_id):
        return sorted(self.directory(collection_id).glob("candidates/*/plan.json"),
                      key=lambda path: (checked(path)["created_at"], str(path)))

    def _plan(self, collection_id, candidate_id):
        plan = checked(self.folder(collection_id, candidate_id) / "plan.json")
        original = checked(self.collections.path(collection_id) / "predictions.json")
        if (plan.get("id") != candidate_id or plan.get("collection_id") != collection_id or
                plan.get("original_prediction_sha256") != original["sha256"]):
            raise ValueError("Teaching version belongs to a different frozen collection")
        if "known_development_receipt" in plan:
            verify_development_receipt(plan["known_development_receipt"], original["sha256"])
        return plan

    def _events(self, collection_id):
        events = [checked(path) for path in self.directory(collection_id).glob("applications/*.json")]
        events.sort(key=lambda item: item["ordinal"])
        previous = None
        for i, item in enumerate(events, 1):
            if (item.get("collection_id") != collection_id or item.get("ordinal") != i or
                    item.get("previous_application_sha256") != previous):
                raise ValueError("Applied teaching version history changed")
            previous = item["sha256"]
        return events

    def active_version(self, collection_id):
        events = self._events(collection_id)
        return self.version_record(collection_id, events[-1]["candidate_id"] if events else "original")

    def version_record(self, collection_id, version_id="original"):
        if version_id in (None, ""):
            return self.active_version(collection_id)
        if version_id == "original":
            # The index already verifies immutable scan identities and stores
            # the small provenance record without materializing every score.
            artifact = self.collections._index(collection_id).metadata()["prediction"]
            return {"id": "original", "name": "Original scan", "prediction_sha256": artifact.get("sha256"),
                    "model": artifact.get("model"), "applied_at": None, "result": None}
        plan = self._plan(collection_id, version_id)
        folder = self.folder(collection_id, version_id)
        artifact = checked(folder / "predictions.json")
        model = checked(folder / "model.json")
        if artifact.get("model_sha256") != model["sha256"] or model.get("plan_sha256") != plan["sha256"]:
            raise ValueError("Teaching model/prediction bindings changed")
        event = next((row for row in self._events(collection_id) if row["candidate_id"] == version_id), None)
        result = checked(folder / "result.json") if (folder / "result.json").exists() else None
        if file_hash(folder / "coefficients.npz") != model["coefficients_sha256"]:
            raise ValueError("Teaching coefficients differ from their frozen model")
        if result:
            self._validate_result(collection_id, version_id, result)
        if event and (event.get("prediction_sha256") != artifact["sha256"] or not result or
                      event.get("result_sha256") != result["sha256"]):
            raise ValueError("Applied version no longer matches its frozen evidence")
        return {"id": version_id, "name": plan["name"], "prediction_sha256": artifact["sha256"],
                "model": artifact.get("model"), "applied_at": event["created_at"] if event else None,
                "result": result, "application": event, "development_count": len(plan["development_labels"])}

    def predictions(self, collection_id, version_id=None):
        version = self.active_version(collection_id) if version_id is None else self.version_record(collection_id, version_id)
        if version["id"] != "original" and not version.get("application"):
            raise ValueError("A teaching candidate must pass its check and be explicitly applied before inspection")
        path = (self.collections.path(collection_id) / "predictions.json" if version["id"] == "original"
                else self.folder(collection_id, version["id"]) / "predictions.json")
        value = checked(path)
        if value["sha256"] != version["prediction_sha256"]:
            raise ValueError("Selected prediction version changed")
        return value

    def reservations(self):
        """Permanent cross-collection exclusions, including byte/pixel identities."""
        reserved_ids, hashes = set(), set()
        paths = list(self.collections.directory.glob("*/teaching/candidates/*/check.json"))
        paths.extend(self.collections.directory.glob("*/checks/*/plan.json"))
        for path in paths:
            value = checked(path)
            for row in value["sample"]:
                reserved_ids.add((value["collection_id"], row["id"]))
                hashes.update(identities(row))
        return reserved_ids, hashes

    def is_reserved(self, collection_id, image_id):
        ids, hashes = self.reservations()
        row = self.collections.image(collection_id, image_id)
        return (collection_id, image_id) in ids or bool(identities(row) & hashes)

    def _known_development_receipt(self, collection_id):
        original = checked(self.collections.path(collection_id) / "predictions.json")
        return known_development_receipt(original, self.replay_directory, self.collections.directory)

    def _development(self, collection_id, manifest, *, lineage_receipt=None):
        rows = {row["id"]: row for row in manifest["images"]}
        lineage_receipt = lineage_receipt or self._known_development_receipt(collection_id)
        original = checked(self.collections.path(collection_id) / "predictions.json")
        known_hashes = verify_development_receipt(lineage_receipt, original["sha256"])
        reserved_ids, reserved_hashes = self.reservations()
        source_manifest = checked(self.replay_directory / "manifest.json")
        protected_hashes = set().union(*(identities(row) for row in source_manifest["images"]
                                        if row["role"] not in ("train", "validation")))
        all_reviews = CollectionReviews(self.collections.path(collection_id)).export()
        reviewed_ids = {row["image_id"] for row in all_reviews["history"]}
        development, excluded = [], 0
        for review in all_reviews["latest"]:
            row = rows.get(review["image_id"])
            if row is None or review.get("manifest_sha256") != manifest["sha256"] or review.get("image_sha256") != row["image_sha256"]:
                raise ValueError("An inspection review no longer matches its collection image")
            if review["disposition"] not in ("confirmed", "corrected") or review.get("label") not in CLASSES:
                continue
            if ((collection_id, row["id"]) in reserved_ids or
                    identities(row) & (reserved_hashes | protected_hashes)):
                excluded += 1
                continue
            development.append({"image_id": row["id"], "label": review["label"],
                "review_sha256": review["sha256"], "review_revision": review["revision"],
                "reviewer": review["reviewer"], "image_sha256": row["image_sha256"],
                "source_sha256": row["sha256"], "role": "development", "selection": "researcher_selected"})
        # New check frames protect all recorded release fitting/selection
        # identities, not just the original replay subset. Historical frames
        # remain in their frozen check artifacts and are never reconstructed.
        reviewed_hashes = set().union(*(identities(rows[i]) for i in reviewed_ids if i in rows))
        frame = [row for row in manifest["images"] if row["id"] not in reviewed_ids and
                 (collection_id, row["id"]) not in reserved_ids and
                 not identities(row) & (reserved_hashes | reviewed_hashes | known_hashes)]
        return development, frame, excluded

    def _replay_sources(self):
        return {name: checked(self.replay_directory / name)["sha256"] for name in
                ("manifest.json", "development-labels.json", "classifier.json", "experiment.json", "predictions.json")}

    @staticmethod
    def training_signature(plan):
        """Effective fitted inputs, excluding review notes, names and revisions."""
        return digest({"collection_id": plan["collection_id"], "manifest_sha256": plan["manifest_sha256"],
            "original_prediction_sha256": plan["original_prediction_sha256"],
            "replay_sources": plan["replay_sources"], "classifier": plan["classifier"],
            "development_labels": sorted((row["image_id"], row["label"]) for row in plan["development_labels"])})

    def duplicate_training(self, collection_id, effective_plan):
        signature = self.training_signature(effective_plan)
        for path in self._plans(collection_id):
            if (path.parent / "check.json").exists() and self.training_signature(checked(path)) == signature:
                return path.parent.name
        return None

    def status(self, collection_id):
        candidates = [self.candidate(collection_id, path.parent.name) for path in self._plans(collection_id)]
        result = {"available": False, "reason": None, "scope": SCOPE, "gate": GATE,
                  "repeated_candidate_scope": REPEATED_SCOPE,
                  "classes": [{"id": key, "name": CLASS_NAMES[key]} for key in CLASSES],
                  "minimum_development": 10, "minimum_classes": 2, "default_check_size": 50,
                  "development_count": 0, "development_classes": [], "check_frame_size": 0,
                  "candidates": list(reversed(candidates)), "active_version": self.active_version(collection_id),
                  "versions": [self.version_record(collection_id, "original")] +
                    [self.version_record(collection_id, row["id"]) for row in candidates
                     if row["status"] == "applied"], "gpu_required": False}
        try:
            manifest = checked(self.collections.path(collection_id) / "manifest.json")
            original = checked(self.collections.path(collection_id) / "predictions.json")
            if not original.get("model", {}).get("feature_chunks"):
                raise ValueError("Completed CONCH classifier feature caches are required")
            require_teaching_cpu()
            labels, frame, excluded = self._development(collection_id, manifest)
            result.update(available=True, development_count=len(labels),
                          development_classes=sorted({row["label"] for row in labels}),
                          check_frame_size=len(frame), reserved_reviews_excluded=excluded)
            pending = any(row["status"] in ("training", "check_ready") for row in candidates)
            duplicate = self.duplicate_training(collection_id, {
                "collection_id": collection_id, "manifest_sha256": manifest["sha256"],
                "original_prediction_sha256": original["sha256"], "replay_sources": self._replay_sources(),
                "classifier": RECIPE, "development_labels": labels}) if candidates and not pending else None
            result["can_create"] = len(labels) >= 10 and len(result["development_classes"]) >= 2 and bool(frame) and not pending and not duplicate
            result["reason"] = ("Finish or stop the current candidate check first" if pending else
                "Review another image or change an approved tissue label before teaching again; these same labels already have a fixed check" if duplicate else
                "Confirm or correct at least 10 images covering two tissue categories" if len(labels) < 10 or len(result["development_classes"]) < 2 else
                "No eligible images remain for a fresh check" if not frame else None)
        except (ValueError, FileNotFoundError) as error:
            result.update(reason=str(error), can_create=False)
        return result

    def create(self, collection_id, *, reviewer, consent, check_size=50):
        reviewer = reviewer_name(reviewer)
        if consent is not True:
            raise ValueError("Explicitly approve using confirmed and corrected inspection labels for development")
        if not isinstance(check_size, int) or isinstance(check_size, bool) or not 20 <= check_size <= 200:
            raise ValueError("Choose a fixed check size from 20 to 200 images")
        require_teaching_cpu()
        # A restarted HTTP process has no executor tasks from its predecessor.
        # Reconcile before taking the new worker lock; the locked check below
        # still prevents concurrent callers from creating two pending versions.
        for path in self._plans(collection_id):
            self._reconcile_interrupted(collection_id, path.parent.name)
        if not self._process_lock.acquire(blocking=False):
            raise ValueError("A CPU teaching operation is already running; try again when it finishes")
        lock_handle = (self.collections.directory / ".teaching-worker.lock").open("a")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.collections.locked(collection_id):
                for path in self._plans(collection_id):
                    if self.candidate(collection_id, path.parent.name)["status"] in ("training", "check_ready"):
                        raise ValueError("Finish the current candidate check or stop it before creating another version")
                manifest = checked(self.collections.path(collection_id) / "manifest.json")
                original = checked(self.collections.path(collection_id) / "predictions.json")
                lineage_receipt = self._known_development_receipt(collection_id)
                labels, frame, excluded = self._development(collection_id, manifest, lineage_receipt=lineage_receipt)
                if len(labels) < 10 or len({row["label"] for row in labels}) < 2:
                    raise ValueError("Confirm or correct at least 10 images covering two tissue categories")
                if not frame:
                    raise ValueError("No eligible images remain for a fresh check")
                identifier = uuid.uuid4().hex
                baseline = self.active_version(collection_id)
                replay_sources = self._replay_sources()
                effective = {"collection_id": collection_id, "manifest_sha256": manifest["sha256"],
                    "original_prediction_sha256": original["sha256"], "replay_sources": replay_sources,
                    "classifier": RECIPE, "development_labels": labels}
                if self.duplicate_training(collection_id, effective):
                    raise ValueError("Review another image or change an approved tissue label before teaching again; these same labels already have a fixed check")
                plan = freeze(self.folder(collection_id, identifier) / "plan.json", {
                    "schema": "collection-teaching-plan-v1", "id": identifier,
                    "name": f"Taught version {len(self._plans(collection_id)) + 1}",
                    "created_at": now(), "collection_id": collection_id, "reviewer": reviewer,
                    "reviewer_identity": "self_declared", "consent": True, "role": "development",
                    "manifest_sha256": manifest["sha256"], "original_prediction_sha256": original["sha256"],
                    "baseline_version": baseline["id"], "baseline_prediction_sha256": baseline["prediction_sha256"],
                    "development_labels": labels, "reserved_reviews_excluded": excluded,
                    "initial_eligible_ids": [row["id"] for row in frame], "requested_check_size": check_size,
                    "known_development_receipt": lineage_receipt,
                    "source_files": source_files(), "replay_directory": str(self.replay_directory),
                    "replay_sources": replay_sources, "classifier": RECIPE,
                    "training_signature": self.training_signature(effective),
                    "scope": SCOPE, "gate": GATE, "existing_predictions_modified": False})
            self.executor.submit(self._worker, collection_id, identifier, lock_handle)
            lock_handle = None
            return self.candidate(collection_id, identifier)
        except Exception:
            self._process_lock.release()
            raise
        finally:
            if lock_handle is not None:
                lock_handle.close()

    def _load_replay(self):
        from .fusion_development import load_encoder
        blocks, binding = load_encoder(self.replay_directory, MODEL)
        classifier = checked(self.replay_directory / "classifier.json")
        if classifier["sha256"] != CLASSIFIER_SHA256 or classifier.get("refit_labels") != 5400:
            raise ValueError("Replay must match the deployed classifier's 5,400 development images")
        features = np.concatenate([blocks[role]["features"] for role in ("train", "validation")])
        labels = np.concatenate([blocks[role]["labels"] for role in ("train", "validation")])
        ids = blocks["train"]["ids"] + blocks["validation"]["ids"]
        if len(ids) != 5400 or len(set(ids)) != 5400 or set(labels) != set(CLASSES):
            raise ValueError("Original replay identities or class coverage changed")
        return features, labels, ids, binding

    def _worker(self, collection_id, candidate_id, lock_handle):
        folder = self.folder(collection_id, candidate_id)
        try:
            self.train(collection_id, candidate_id)
        except Exception as error:
            if not (folder / "failure.json").exists():
                freeze(folder / "failure.json", {"created_at": now(), "reason": str(error)[:1000],
                    **execution_provenance()})
        finally:
            lock_handle.close()
            self._process_lock.release()

    def train(self, collection_id, candidate_id):
        threads = require_teaching_cpu()
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        plan = self._plan(collection_id, candidate_id)
        folder = self.folder(collection_id, candidate_id)
        if plan["source_files"] != source_files():
            raise ValueError("Teaching implementation changed after this candidate was queued")
        if "known_development_receipt" in plan:
            verify_development_receipt(plan["known_development_receipt"],
                                       plan["original_prediction_sha256"], verify_sources=True)
        freeze(folder / "started.json", {"created_at": now(), "plan_sha256": plan["sha256"],
            **execution_provenance(), "hostname": socket.gethostname(), "gpus": 0})
        cached = CollectionFeatures(self.collections, collection_id)
        if cached.manifest["sha256"] != plan["manifest_sha256"]:
            raise ValueError("Collection changed after teaching approval")
        replay_x, replay_y, replay_ids, replay_binding = self._load_replay()
        if plan["replay_sources"] != {name: checked(self.replay_directory / name)["sha256"] for name in plan["replay_sources"]}:
            raise ValueError("Replay development sources changed after approval")
        positions = {identifier: i for i, identifier in enumerate(cached.ids)}
        selected = [positions[row["image_id"]] for row in plan["development_labels"]]
        reserved_ids, reserved_hashes = self.reservations()
        if any((collection_id, cached.ids[i]) in reserved_ids or identities(cached.rows[i]) & reserved_hashes for i in selected):
            raise ValueError("A selected development image is permanently reserved for checking")
        x = np.concatenate([replay_x, cached.features[selected]])
        y = np.concatenate([replay_y, np.array([row["label"] for row in plan["development_labels"]])])
        weights = np.concatenate([np.ones(len(replay_y)), np.full(len(selected), 5400. / len(selected))])
        with threadpool_limits(limits=min(2, threads)), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(C=10., solver="lbfgs", max_iter=2000, random_state=20261018)
            model.fit(x, y, sample_weight=weights)
            if model.classes_.tolist() != CLASSES or int(model.n_iter_.max()) >= 2000:
                raise ValueError("The bounded classifier fit did not converge with all nine classes")
            scores = score_features(cached.features.astype(np.float64), model.coef_, model.intercept_)
        with (folder / "coefficients.npz").open("xb") as handle:
            np.savez_compressed(handle, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)
        model_record = freeze(folder / "model.json", {"created_at": now(), "plan_sha256": plan["sha256"],
            "coefficients_sha256": file_hash(folder / "coefficients.npz"), "classes": CLASSES,
            "replay_binding": replay_binding, "replay_ids": replay_ids, "replay_count": len(replay_ids),
            "feature_chunks": cached.chunks, "development_count": len(selected),
            "feature_key": "classifier_embeddings", "iterations": model.n_iter_.tolist(),
            "source_files": plan["source_files"], "packages": {name: importlib.metadata.version(name)
                for name in ("numpy", "scikit-learn")}, **execution_provenance(), "gpus": 0})
        predictions = prediction_records(cached.rows, scores)
        counts = {label: sum(row["predicted_label"] == label for row in predictions) for label in CLASSES}
        inherited = cached.prediction["model"]
        candidate_model = {key: inherited[key] for key in (
            "configuration_sha256", "scope", "review_rule", "score_type", "manifest_sha256",
            "encoder_identity", "checkpoint_sha256", "code_revision", "worker_source_sha256",
            "feature_chunks", "classifier_representation", "retrieval_representation") if key in inherited}
        candidate_model.update(name=plan["name"], classifier_sha256=model_record["sha256"],
            coefficients_sha256=model_record["coefficients_sha256"], teaching_candidate_id=candidate_id,
            configuration_sha256=plan["sha256"],
            source_scan_configuration_sha256=inherited.get("configuration_sha256"),
            encoder_worker_source_sha256=inherited.get("worker_source_sha256"),
            worker_source_sha256=plan["source_files"]["collection_teaching.py"],
            model_training=True, reference_labels_used=True, reference_label_role="development",
            development_labels=len(selected), replay_development_labels=len(replay_ids),
            check_labels_used_for_training=False, **execution_provenance(),
            host=socket.gethostname(), device="CPU; cached CONCH pooled features", gpus=0,
            cached_images_reused=len(cached.ids), newly_encoded_images=0,
            source_scan_prediction_sha256=cached.prediction["sha256"],
            source_scan_model_sha256=digest(inherited), teaching_source_files=plan["source_files"])
        artifact = freeze(folder / "predictions.json", {"schema": "preliminary-collection-predictions-v1",
            "collection_id": collection_id, "created_at": now(), "manifest_sha256": cached.manifest["sha256"],
            "model_sha256": model_record["sha256"], "prediction_version": candidate_id,
            "model": candidate_model,
            "scope": SCAN_SCOPE, "score_type": SCORE_TYPE, "predictions": predictions,
            "summary": {"counts": counts, "predicted_count": len(predictions),
                        "uncertain_count": sum(row["uncertain"] for row in predictions), "failed_count": 0}})
        # Only after all candidate scores are frozen do we select a check sample.
        with self.collections.locked(collection_id):
            lineage_receipt = plan.get("known_development_receipt")
            if lineage_receipt is not None:
                verify_development_receipt(lineage_receipt, plan["original_prediction_sha256"], verify_sources=True)
            _, frame, _ = self._development(collection_id, cached.manifest, lineage_receipt=lineage_receipt)
            eligible = set(plan["initial_eligible_ids"])
            frame = [row for row in frame if row["id"] in eligible]
            if not frame:
                raise ValueError("No eligible images remain for this candidate's check")
            n = min(plan["requested_check_size"], len(frame))
            seed = secrets.randbits(63)
            indexes = np.random.default_rng(seed).choice(len(frame), size=n, replace=False)
            freeze(folder / "check.json", {"schema": "collection-teaching-check-v1", "created_at": now(),
                "collection_id": collection_id, "candidate_id": candidate_id, "plan_sha256": plan["sha256"],
                "prediction_sha256": artifact["sha256"], "baseline_prediction_sha256": plan["baseline_prediction_sha256"],
                "frame": [{"id": row["id"], "image_sha256": row["image_sha256"], "source_sha256": row["sha256"]} for row in frame],
                "frame_size": len(frame), "sample": [{"id": frame[int(i)]["id"],
                    "image_sha256": frame[int(i)]["image_sha256"], "source_sha256": frame[int(i)]["sha256"]} for i in indexes],
                "sample_size": n, "seed": seed, "inclusion_probability": n / len(frame),
                **({"known_development_receipt_sha256": lineage_receipt["sha256"],
                    "identity_scope": IDENTITY_SCOPE} if lineage_receipt is not None else {}),
                "sampling": "Uniform simple random sample without replacement; census if frame is smaller than requested size",
                "permanently_excluded_from_training": True, "labels_read_before_prediction_freeze": False,
                "scope": "Previously unreviewed images after excluding matches to recorded fitting/selection and reserved-check identities; finite-frame agreement with reviewer labels"})

    def _check(self, collection_id, candidate_id):
        plan = self._plan(collection_id, candidate_id)
        folder = self.folder(collection_id, candidate_id)
        check = checked(folder / "check.json")
        predictions = checked(folder / "predictions.json")
        if (check.get("collection_id") != collection_id or check.get("candidate_id") != candidate_id or
                check.get("plan_sha256") != plan["sha256"] or check.get("prediction_sha256") != predictions["sha256"]):
            raise ValueError("The check sample no longer matches its frozen candidate")
        if "known_development_receipt" in plan and check.get("known_development_receipt_sha256") != plan["known_development_receipt"]["sha256"]:
            raise ValueError("The check differs from its recorded development exclusions")
        frame, sample = check.get("frame", []), check.get("sample", [])
        frame_ids, sample_ids = [row["id"] for row in frame], [row["id"] for row in sample]
        if (not frame or not sample or len(set(frame_ids)) != len(frame) or len(set(sample_ids)) != len(sample) or
                not set(sample_ids).issubset(frame_ids) or not set(frame_ids).issubset(plan["initial_eligible_ids"]) or
                check.get("frame_size") != len(frame) or check.get("sample_size") != len(sample) or
                len(sample) != min(plan["requested_check_size"], len(frame)) or
                check.get("inclusion_probability") != len(sample) / len(frame)):
            raise ValueError("Frozen check frame or inclusion probabilities changed")
        expected = np.random.default_rng(check["seed"]).choice(len(frame), size=len(sample), replace=False)
        if sample != [frame[int(i)] for i in expected]:
            raise ValueError("Frozen check membership differs from its recorded uniform draw")
        return check

    def _labels(self, collection_id, candidate_id, check):
        folder = self.folder(collection_id, candidate_id)
        sample = {row["id"]: row for row in check["sample"]}
        labels = {}
        for path in (folder / "check-labels").glob("*.json"):
            row = checked(path)
            if (row.get("image_id") not in sample or row.get("check_sha256") != check["sha256"] or
                    row.get("role") != "check" or row.get("label") not in CLASSES or
                    path.stem != row.get("image_id") or
                    row.get("image_sha256") != sample[row["image_id"]]["image_sha256"]):
                raise ValueError("A check label no longer matches its reserved image")
            labels[row["image_id"]] = row
        return labels

    def _validate_result(self, collection_id, candidate_id, result):
        check = self._check(collection_id, candidate_id)
        labels = self._labels(collection_id, candidate_id, check)
        if (result.get("collection_id") != collection_id or result.get("candidate_id") != candidate_id or
                result.get("check_sha256") != check["sha256"] or
                result.get("prediction_sha256") != check["prediction_sha256"] or
                len(labels) != check["sample_size"] or result.get("check_label_sha256") !=
                [labels[row["id"]]["sha256"] for row in check["sample"]]):
            raise ValueError("The evaluated result differs from its frozen check evidence")
        return check

    def _close_evidence(self, collection_id, candidate_id):
        """Bind a stopped check to every saved response, including Unsure."""
        plan = self._plan(collection_id, candidate_id)
        folder = self.folder(collection_id, candidate_id)
        check = self._check(collection_id, candidate_id)
        labels = self._labels(collection_id, candidate_id, check)
        sample = {row["id"]: row for row in check["sample"]}
        unsure = []
        for path in sorted((folder / "unsure").glob("*.json")):
            row = checked(path)
            if (row.get("collection_id") != collection_id or row.get("candidate_id") != candidate_id or
                    row.get("check_sha256") != check["sha256"] or row.get("image_id") not in sample or
                    row.get("image_sha256") != sample[row["image_id"]]["image_sha256"] or
                    row.get("role") != "check" or row.get("label") != "unsure"):
                raise ValueError("An Unsure response no longer matches its reserved check image")
            unsure.append(row)
        return {"plan_sha256": plan["sha256"], "check_sha256": check["sha256"],
                "prediction_sha256": check["prediction_sha256"],
                "check_label_sha256": [labels[row["id"]]["sha256"] for row in check["sample"]
                                       if row["id"] in labels],
                "unsure_record_sha256": [row["sha256"] for row in unsure],
                "planned": check["sample_size"], "answered": len(labels),
                "remaining": check["sample_size"] - len(labels)}

    def _closure(self, collection_id, candidate_id):
        folder = self.folder(collection_id, candidate_id)
        if not (folder / "closed.json").exists():
            return None
        closure = checked(folder / "closed.json")
        evidence = self._close_evidence(collection_id, candidate_id)
        if (closure.get("schema") != "collection-teaching-closure-v1" or
                closure.get("collection_id") != collection_id or closure.get("candidate_id") != candidate_id or
                closure.get("previous_status") != "check_ready" or evidence["remaining"] <= 0 or
                any(closure.get(key) != value for key, value in evidence.items()) or
                (folder / "result.json").exists() or (folder / "failure.json").exists() or
                any(row["candidate_id"] == candidate_id for row in self._events(collection_id))):
            raise ValueError("The stopped candidate differs from its frozen closure evidence")
        return closure

    def _require_open(self, collection_id, candidate_id):
        if self._closure(collection_id, candidate_id):
            raise ValueError("This candidate was stopped; its check and saved responses are preserved")

    def close(self, collection_id, candidate_id, *, reviewer, note=""):
        """Stop an incomplete check without releasing or redrawing its sample."""
        reviewer = reviewer_name(reviewer)
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Use a note up to 2,000 characters")
        # The worker holds these locks until fitting and check creation both
        # finish. A status read alone has a small race after check.json appears.
        if not self._process_lock.acquire(blocking=False):
            raise ValueError("A CPU teaching operation is running; wait for it to finish before stopping a check")
        lock_handle = None
        try:
            lock_handle = (self.collections.directory / ".teaching-worker.lock").open("a")
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("A CPU teaching operation is running; wait for it to finish before stopping a check") from error
            with self.collections.locked(collection_id):
                if self._closure(collection_id, candidate_id):
                    return self.candidate(collection_id, candidate_id)
                candidate = self.candidate(collection_id, candidate_id)
                if not candidate["can_close"]:
                    raise ValueError("Only an incomplete candidate check can be stopped; completed or applied versions stay in history")
                evidence = self._close_evidence(collection_id, candidate_id)
                freeze(self.folder(collection_id, candidate_id) / "closed.json", {
                    "schema": "collection-teaching-closure-v1", "created_at": now(),
                    "collection_id": collection_id, "candidate_id": candidate_id,
                    "reviewer": reviewer, "reviewer_identity": "self_declared", "note": note.strip(),
                    "previous_status": candidate["status"], **evidence,
                    "explicit_user_action": True, "sample_reservations_preserved": True,
                    "existing_predictions_modified": False, "check_replaced": False})
                return self.candidate(collection_id, candidate_id)
        finally:
            if lock_handle is not None:
                lock_handle.close()
            self._process_lock.release()

    def _reconcile_interrupted(self, collection_id, candidate_id):
        """Record an abandoned fit only after proving no teaching worker owns it.

        Every submitted worker holds the global file lock from plan publication
        through check publication, including time waiting in the executor. A
        free lock therefore proves an unfinished plan has no surviving worker;
        elapsed time, PID reuse and scheduler availability are not used as guesses.
        Completion/check records are never changed or regenerated here.
        """
        folder = self.folder(collection_id, candidate_id)
        finished = ("check.json", "result.json", "failure.json", "closed.json")
        if any((folder / name).exists() for name in finished):
            return
        if not self._process_lock.acquire(blocking=False):
            return
        try:
            with (self.collections.directory / ".teaching-worker.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                with self.collections.locked(collection_id):
                    if any((folder / name).exists() for name in finished):
                        return
                    plan = self._plan(collection_id, candidate_id)
                    if any(row["candidate_id"] == candidate_id for row in self._events(collection_id)):
                        raise ValueError("Applied teaching candidate is missing its completion artifacts")
                    preserved = []
                    for name in ("plan.json", "started.json", "coefficients.npz", "model.json", "predictions.json"):
                        path = folder / name
                        if path.exists():
                            if path.is_symlink() or not path.is_file():
                                raise ValueError("Interrupted teaching artifact storage changed")
                            preserved.append({"file": name, "sha256": file_hash(path), "bytes": path.stat().st_size})
                    freeze(folder / "failure.json", {
                        "schema": "collection-teaching-failure-v1", "created_at": now(),
                        "kind": "interrupted", "collection_id": collection_id,
                        "candidate_id": candidate_id, "plan_sha256": plan["sha256"],
                        "reason": "Teaching was interrupted before its check was ready. The saved attempt is preserved; explicitly start a new candidate to try again.",
                        "recovery_basis": "No process holds the shared teaching-worker lock; no check or completion record exists",
                        "preserved_artifacts": preserved,
                        "reconciled_on_job_id": os.environ.get("SLURM_JOB_ID"),
                        "reconciled_on_hostname": socket.gethostname(),
                        "automatic_retry": False, "existing_predictions_modified": False,
                    })
        finally:
            self._process_lock.release()

    def candidate(self, collection_id, candidate_id):
        plan = self._plan(collection_id, candidate_id)
        self._reconcile_interrupted(collection_id, candidate_id)
        folder = self.folder(collection_id, candidate_id)
        state, summary, result, failure, failure_kind = "training", None, None, None, None
        if (folder / "failure.json").exists():
            failure_record = checked(folder / "failure.json")
            if failure_record.get("kind") == "interrupted" and (
                    failure_record.get("collection_id") != collection_id or
                    failure_record.get("candidate_id") != candidate_id or
                    failure_record.get("plan_sha256") != plan["sha256"]):
                raise ValueError("Interrupted teaching failure belongs to a different saved attempt")
            state, failure = "failed", failure_record["reason"]
            failure_kind = failure_record.get("kind", "worker_error")
        elif (folder / "check.json").exists():
            check = self._check(collection_id, candidate_id)
            labels = self._labels(collection_id, candidate_id, check)
            state = "check_ready"
            summary = {"planned": check["sample_size"], "answered": len(labels),
                       "remaining": check["sample_size"] - len(labels), "frame_size": check["frame_size"]}
            if (folder / "result.json").exists():
                state, result = "evaluated", checked(folder / "result.json")
                self._validate_result(collection_id, candidate_id, result)
            if any(row["candidate_id"] == candidate_id for row in self._events(collection_id)):
                state = "applied"
        closure = self._closure(collection_id, candidate_id)
        if closure:
            state = "closed"
        return {"id": candidate_id, "name": plan["name"], "created_at": plan["created_at"],
                "status": state, "development_count": len(plan["development_labels"]),
                "baseline_version": plan["baseline_version"], "check": summary, "result": result,
                "failure": failure, "failure_kind": failure_kind, "closure": closure,
                "can_close": bool(state == "check_ready" and summary and summary["remaining"] > 0),
                "can_apply": bool(result and result["can_apply"] and state == "evaluated"),
                "scope": SCOPE, "gpu_required": False}

    def check_unit(self, collection_id, candidate_id):
        self._require_open(collection_id, candidate_id)
        check = self._check(collection_id, candidate_id)
        labels = self._labels(collection_id, candidate_id, check)
        remaining = [row for row in check["sample"] if row["id"] not in labels]
        result = {"complete": not remaining, "image": None, "planned": check["sample_size"],
                  "answered": len(labels), "remaining": len(remaining), "frame_size": check["frame_size"],
                  "classes": [{"id": key, "name": CLASS_NAMES[key]} for key in CLASSES],
                  "scope": "Assign a tissue label without seeing either classifier's prediction"}
        if remaining:
            row = self.collections.image(collection_id, remaining[0]["id"])
            result.update(position=len(labels) + 1, image={"id": row["id"],
                "image_url": f"api/collections/{collection_id}/images/{row['id']}/image"})
        return result

    def save_check_label(self, collection_id, candidate_id, *, image_id, label, reviewer, note=""):
        reviewer = reviewer_name(reviewer)
        if label not in CLASSES + ["unsure"] or not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Choose a tissue category or Unsure, with a note up to 2,000 characters")
        with self.collections.locked(collection_id):
            self._require_open(collection_id, candidate_id)
            check = self._check(collection_id, candidate_id)
            folder = self.folder(collection_id, candidate_id)
            if (folder / "result.json").exists():
                raise ValueError("This check has already been frozen and evaluated")
            sample = {row["id"]: row for row in check["sample"]}
            if image_id not in sample:
                raise ValueError("This image is not a planned check unit")
            labels = self._labels(collection_id, candidate_id, check)
            if image_id in labels:
                if labels[image_id]["label"] == label and labels[image_id]["reviewer"] == reviewer:
                    return self.candidate(collection_id, candidate_id)
                raise ValueError("A submitted check label is immutable")
            record = {"created_at": now(), "collection_id": collection_id, "candidate_id": candidate_id,
                "check_sha256": check["sha256"], "image_id": image_id,
                "image_sha256": sample[image_id]["image_sha256"], "reviewer": reviewer,
                "reviewer_identity": "self_declared", "label": label, "note": note.strip(), "role": "check"}
            if label == "unsure":
                freeze(folder / "unsure" / (uuid.uuid4().hex + ".json"), record)
            else:
                freeze(folder / "check-labels" / (image_id + ".json"), record)
        return self.candidate(collection_id, candidate_id)

    def evaluate(self, collection_id, candidate_id):
        from .collection_teaching_stats import assess_check, assess_class_proportions
        with self.collections.locked(collection_id):
            self._require_open(collection_id, candidate_id)
            folder = self.folder(collection_id, candidate_id)
            if (folder / "result.json").exists():
                return self.candidate(collection_id, candidate_id)
            plan = self._plan(collection_id, candidate_id)
            check = self._check(collection_id, candidate_id)
            labels = self._labels(collection_id, candidate_id, check)
            if len(labels) != check["sample_size"]:
                raise ValueError("Label every planned check image before evaluating; unsure images cannot be replaced")
            base = self.predictions(collection_id, plan["baseline_version"])
            new = checked(folder / "predictions.json")
            if base["sha256"] != plan["baseline_prediction_sha256"]:
                raise ValueError("The frozen comparison version changed")
            base_by_id = {row["id"]: row["predicted_label"] for row in base["predictions"]}
            new_by_id = {row["id"]: row["predicted_label"] for row in new["predictions"]}
            ids = [row["id"] for row in check["sample"]]
            assessment = assess_check([labels[i]["label"] for i in ids], [base_by_id[i] for i in ids],
                                      [new_by_id[i] for i in ids], check["frame_size"])
            frame_ids = [row["id"] for row in check["frame"]]
            frame_positions = {identifier: i for i, identifier in enumerate(frame_ids)}
            measurements = assess_class_proportions([labels[i]["label"] for i in ids],
                [base_by_id[i] for i in frame_ids], [new_by_id[i] for i in frame_ids],
                [frame_positions[i] for i in ids], CLASSES)
            can_apply = assessment["observed_nonworse"]
            freeze(folder / "result.json", {"created_at": now(), "collection_id": collection_id,
                "candidate_id": candidate_id, "check_sha256": check["sha256"],
                "prediction_sha256": new["sha256"], "baseline_prediction_sha256": base["sha256"],
                "check_label_sha256": [labels[i]["sha256"] for i in ids], **assessment,
                "can_apply": can_apply, "gate": GATE, "gate_reason":
                    "Observed check accuracy is no worse; you may explicitly use this version" if can_apply else
                    "The candidate made more errors on this check; this version cannot be applied",
                "count_changes": {key: new["summary"]["counts"].get(key, 0) - base["summary"]["counts"].get(key, 0)
                                  for key in CLASSES}, "counts_scope": "Predicted image counts, not tissue area",
                "measurements": measurements,
                "repeated_candidate_scope": REPEATED_SCOPE,
                "labels_used_for_training": False, "automatic_promotion": False})
        return self.candidate(collection_id, candidate_id)

    def apply(self, collection_id, candidate_id, *, reviewer):
        reviewer = reviewer_name(reviewer)
        with self.collections.locked(collection_id):
            self._require_open(collection_id, candidate_id)
            folder = self.folder(collection_id, candidate_id)
            result = checked(folder / "result.json")
            check = self._validate_result(collection_id, candidate_id, result)
            if (result.get("check_sha256") != check["sha256"] or not result.get("can_apply") or
                    len(self._labels(collection_id, candidate_id, check)) != check["sample_size"]):
                raise ValueError("This candidate has not passed its complete fresh check")
            plan = self._plan(collection_id, candidate_id)
            events = self._events(collection_id)
            if events and events[-1]["candidate_id"] == candidate_id:
                return self.candidate(collection_id, candidate_id)
            active = events[-1]["candidate_id"] if events else "original"
            if active != plan["baseline_version"]:
                raise ValueError("The active model changed; this check does not compare with the current version")
            freeze(self.directory(collection_id) / "applications" / (f"{len(events)+1:06d}-" + uuid.uuid4().hex + ".json"), {
                "created_at": now(), "collection_id": collection_id, "candidate_id": candidate_id,
                "ordinal": len(events) + 1, "previous_application_sha256": events[-1]["sha256"] if events else None,
                "reviewer": reviewer, "reviewer_identity": "self_declared", "result_sha256": result["sha256"],
                "prediction_sha256": result["prediction_sha256"], "explicit_user_action": True,
                "original_predictions_modified": False})
        return self.candidate(collection_id, candidate_id)

    def export(self, collection_id, version_id):
        """Self-contained numerical evidence for an explicitly applied version."""
        version = self.version_record(collection_id, version_id)
        if version["id"] == "original" or not version.get("application"):
            raise ValueError("Choose an applied teaching version to export its check evidence")
        folder = self.folder(collection_id, version_id)
        plan = self._plan(collection_id, version_id)
        check = self._check(collection_id, version_id)
        labels = self._labels(collection_id, version_id, check)
        baseline = self.predictions(collection_id, plan["baseline_version"])
        candidate = checked(folder / "predictions.json")
        baseline_rows = {row["id"]: row for row in baseline["predictions"]}
        candidate_rows = {row["id"]: row for row in candidate["predictions"]}
        record = {"schema": "collection-teaching-evidence-export-v1", "exported_at": now(),
            "collection_id": collection_id, "version_id": version_id, "name": version["name"],
            "plan": plan, "model": checked(folder / "model.json"), "check": check,
            "check_labels": [labels[row["id"]] for row in check["sample"]],
            "result": version["result"], "application": version["application"],
            "frame_predictions": [{"id": row["id"],
                "baseline_label": baseline_rows[row["id"]]["predicted_label"],
                "candidate_label": candidate_rows[row["id"]]["predicted_label"]} for row in check["frame"]],
            "scope": SCOPE, "reviewer_identity": "self_declared",
            "reproduction": "Recompute finite-frame accuracy and class proportions using the frozen check sample, check labels and full-frame predicted labels. Model fitting additionally requires the referenced immutable image feature caches.",
            "original_predictions_modified": False}
        return {**record, "sha256": digest(record)}
