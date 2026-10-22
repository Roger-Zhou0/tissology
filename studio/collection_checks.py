"""Standalone probability checks of a frozen collection prediction version.

No fitting, feature loading or inference. Saved samples and labels are permanent
check evidence, never development examples. The estimand is accuracy relative to
the reviewer's labels in the recorded eligible finite image frame.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import secrets
import uuid

import numpy as np

from .collection_reviews import CollectionReviews
from .collection_storage import CLASS_NAMES, CLASSES
from .collection_teaching import checked, identities, reviewer_name, _ID
from .collection_lineage import known_development_receipt
from .feedback_storage import freeze
from .storage import digest, now, read

SCOPE = ("Accuracy relative to the submitted reviewer labels in the frozen eligible image frame. "
         "Matches to recorded previously reviewed, development/training and reserved check identities, "
         "and unsuccessfully classified images, are excluded before sampling. Each image counts once; "
         "this is not tissue area, diagnostic accuracy, or patient-level generalization.")
IDENTITY_SCOPE = ("Exclusions match the recorded original-byte, canonical-image and available pixel identities. "
                  "Legacy records can lack pixel fingerprints, so re-encoded or transformed copies may evade "
                  "matching. Unknown foundation-model pretraining overlap is not assessed. This check does "
                  "not establish an entirely training-independent collection or patient-independent validation.")
INTERVAL_METHOD = "Exact equal-tailed hypergeometric inversion for simple random sampling without replacement"
CLASSES_PUBLIC = [{"id": code, "name": CLASS_NAMES[code]} for code in CLASSES]


def image_identity(row):
    return {"id": row["id"], "image_sha256": row["image_sha256"],
            "source_sha256": row["sha256"],
            **{key: row[key] for key in ("pixel_sha256", "pixel_digest") if key in row}}


class CollectionChecksManager:
    def __init__(self, collections, teaching):
        self.collections = collections
        self.teaching = teaching

    def directory(self, collection_id):
        path = self.collections.path(collection_id) / "checks"
        if path.is_symlink():
            raise ValueError("Scan check storage changed")
        return path

    def folder(self, collection_id, check_id):
        if not isinstance(check_id, str) or not _ID.fullmatch(check_id):
            raise ValueError("Unknown scan check")
        path = self.directory(collection_id) / check_id
        if path.is_symlink():
            raise ValueError("Scan check storage changed")
        return path

    def _plans(self, collection_id):
        return sorted(self.directory(collection_id).glob("*/plan.json"),
                      key=lambda path: (checked(path)["created_at"], str(path)))

    @contextmanager
    def _sampling_lock(self):
        # Share both locks with teaching: creation cannot race candidate fitting
        # or the final sample reservation written by its worker.
        if not self.teaching._process_lock.acquire(blocking=False):
            raise ValueError("A teaching operation is running; try again when it finishes")
        handle = None
        try:
            handle = (self.collections.directory / ".teaching-worker.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("A teaching operation is running; try again when it finishes") from error
            yield
        finally:
            if handle is not None:
                handle.close()
            self.teaching._process_lock.release()

    def _snapshot(self, collection_id, prediction_version=None):
        manifest = checked(self.collections.path(collection_id) / "manifest.json")
        version = (self.teaching.active_version(collection_id) if prediction_version in (None, "")
                   else self.teaching.version_record(collection_id, prediction_version))
        prediction = self.teaching.predictions(collection_id, version["id"])
        scan = read(self.collections.path(collection_id) / "scan.json")
        rows = manifest.get("images", [])
        if (scan.get("status") != "complete" or manifest.get("collection_id") != collection_id or
                manifest.get("role") != "unlabeled_inspection" or not rows or
                len({row["id"] for row in rows}) != len(rows) or
                prediction.get("collection_id") != collection_id or
                prediction.get("manifest_sha256") != manifest["sha256"] or
                prediction.get("sha256") != version.get("prediction_sha256")):
            raise ValueError("Finish a verified collection scan before checking its accuracy")
        predictions = prediction.get("predictions", [])
        if (len({row["id"] for row in predictions}) != len(predictions) or
                not {row["id"] for row in predictions}.issubset({row["id"] for row in rows})):
            raise ValueError("Prediction identities differ from their collection")
        return manifest, prediction, version

    def _known_development(self, original):
        """Resolve known fitting/selection identities without opening image features."""
        receipt = known_development_receipt(original, self.teaching.replay_directory,
                                            self.collections.directory)
        return set(receipt["identity_hashes"]), receipt["sources"]

    def _frame(self, collection_id, manifest, prediction):
        original = checked(self.collections.path(collection_id) / "predictions.json")
        development, sources = self._known_development(original)
        reserved_ids, reserved_hashes = self.teaching.reservations()
        reviewed_ids, reviewed_hashes = set(), set()
        for path in self.collections.directory.glob("*/reviews.sqlite3"):
            for review in CollectionReviews(path.parent).export()["history"]:
                reviewed_ids.add((review["collection_id"], review["image_id"]))
                reviewed_hashes.update(identities(review))
        predictions = {row["id"]: row for row in prediction["predictions"]}
        frame, exclusions = [], []
        counts = {key: 0 for key in ("previously_reviewed", "reserved_check", "known_development", "no_successful_prediction")}
        for row in manifest["images"]:
            identity = identities(row)
            pair = (collection_id, row["id"])
            predicted = predictions.get(row["id"], {})
            reasons = []
            if pair in reviewed_ids or identity & reviewed_hashes:
                reasons.append("previously_reviewed")
            if pair in reserved_ids or identity & reserved_hashes:
                reasons.append("reserved_check")
            if identity & development:
                reasons.append("known_development")
            if predicted.get("predicted_label") not in CLASSES or predicted.get("error"):
                reasons.append("no_successful_prediction")
            if reasons:
                exclusions.append({**image_identity(row), "reasons": reasons})
                for reason in reasons:
                    counts[reason] += 1
            else:
                frame.append(image_identity(row))
        return frame, exclusions, counts, sources

    def _plan(self, collection_id, check_id):
        plan = checked(self.folder(collection_id, check_id) / "plan.json")
        manifest, prediction, _ = self._snapshot(collection_id, plan.get("prediction_version"))
        rows = {row["id"]: image_identity(row) for row in manifest["images"]}
        frame, sample = plan.get("frame", []), plan.get("sample", [])
        frame_ids = [row["id"] for row in frame]
        excluded_ids = [row["id"] for row in plan.get("exclusions", [])]
        if (plan.get("id") != check_id or plan.get("collection_id") != collection_id or
                plan.get("manifest_sha256") != manifest["sha256"] or
                plan.get("prediction_sha256") != prediction["sha256"] or
                not frame or not sample or len(set(frame_ids)) != len(frame) or
                len(set(excluded_ids)) != len(excluded_ids) or set(frame_ids) & set(excluded_ids) or
                set(frame_ids) | set(excluded_ids) != set(rows) or
                any(rows.get(row["id"]) != row for row in frame) or
                plan.get("frame_size") != len(frame) or plan.get("sample_size") != len(sample) or
                len(sample) != min(plan.get("requested_check_size", 0), len(frame)) or
                plan.get("inclusion_probability") != len(sample) / len(frame)):
            raise ValueError("The scan check differs from its frozen frame or prediction version")
        indexes = np.random.default_rng(plan["seed"]).choice(len(frame), size=len(sample), replace=False)
        if sample != [frame[int(i)] for i in indexes]:
            raise ValueError("The scan check differs from its recorded random draw")
        return plan, prediction

    def _responses(self, collection_id, check_id, plan):
        folder = self.folder(collection_id, check_id)
        sample = {row["id"]: row for row in plan["sample"]}
        labels, unsure = {}, []
        for kind in ("labels", "unsure"):
            for path in sorted((folder / kind).glob("*.json")):
                row = checked(path)
                if (row.get("collection_id") != collection_id or row.get("check_id") != check_id or
                        row.get("plan_sha256") != plan["sha256"] or row.get("image_id") not in sample or
                        row.get("image_sha256") != sample[row["image_id"]]["image_sha256"] or
                        row.get("role") != "check" or
                        (kind == "labels" and (row.get("label") not in CLASSES or path.stem != row["image_id"])) or
                        (kind == "unsure" and row.get("label") != "unsure")):
                    raise ValueError("A saved check response differs from its reserved image")
                if kind == "labels":
                    labels[row["image_id"]] = row
                else:
                    unsure.append(row)
        return labels, unsure

    @staticmethod
    def _evidence(plan, labels, unsure):
        return {"plan_sha256": plan["sha256"], "prediction_sha256": plan["prediction_sha256"],
                "label_sha256": [labels[row["id"]]["sha256"] for row in plan["sample"] if row["id"] in labels],
                "unsure_sha256": [row["sha256"] for row in unsure]}

    @staticmethod
    def _assessment(plan, prediction, labels):
        from .collection_teaching_stats import _finite_population_interval
        by_id = {row["id"]: row["predicted_label"] for row in prediction["predictions"]}
        correct = sum(labels[row["id"]]["label"] == by_id[row["id"]] for row in plan["sample"])
        return {"correct": correct, "sample_size": plan["sample_size"], "frame_size": plan["frame_size"],
                "accuracy": correct / plan["sample_size"],
                "interval": _finite_population_interval(correct, plan["sample_size"], plan["frame_size"]),
                "confidence_level": .95, "interval_method": INTERVAL_METHOD,
                "scope": SCOPE, "identity_scope": IDENTITY_SCOPE, "labels_used_for_training": False,
                "measurement": {"numerator": "Images whose frozen predicted class agrees with the submitted reviewer class",
                    "denominator": "All images in the frozen eligible check frame",
                    "unit": "image", "physical_scale": "None; image accuracy is not tissue area",
                    "reference": "Self-declared researcher labels; reference-label errors are not covered by this interval"}}

    def check(self, collection_id, check_id):
        plan, prediction = self._plan(collection_id, check_id)
        labels, unsure = self._responses(collection_id, check_id, plan)
        folder = self.folder(collection_id, check_id)
        result, closure, status = None, None, "ready"
        evidence = self._evidence(plan, labels, unsure)
        if (folder / "result.json").exists():
            result = checked(folder / "result.json")
            if (len(labels) != plan["sample_size"] or result.get("collection_id") != collection_id or
                    result.get("check_id") != check_id or
                    any(result.get(key) != value for key, value in evidence.items()) or
                    any(result.get(key) != value for key, value in self._assessment(plan, prediction, labels).items())):
                raise ValueError("The accuracy result differs from its frozen evidence")
            status = "evaluated"
        if (folder / "closed.json").exists():
            closure = checked(folder / "closed.json")
            if (result or len(labels) == plan["sample_size"] or closure.get("collection_id") != collection_id or
                    closure.get("check_id") != check_id or
                    any(closure.get(key) != value for key, value in evidence.items())):
                raise ValueError("The stopped check differs from its saved evidence")
            status = "closed"
        return {"id": check_id, "status": status, "created_at": plan["created_at"],
                "prediction_version": plan["prediction_version"], "version_name": plan["version_name"],
                "prediction_sha256": plan["prediction_sha256"], "scope": SCOPE, "identity_scope": IDENTITY_SCOPE,
                "progress": {"planned": plan["sample_size"], "answered": len(labels),
                    "remaining": plan["sample_size"] - len(labels), "frame_size": plan["frame_size"],
                    "unsure_responses": len(unsure)},
                "result": result, "closure": closure,
                "exclusion_counts": plan["exclusion_counts"], "excluded_images": len(plan["exclusions"]),
                "can_close": status == "ready" and len(labels) < plan["sample_size"]}

    def status(self, collection_id, prediction_version=None):
        response = {"available": False, "can_create": False, "reason": None,
                    "frame_size": 0, "default_check_size": 50, "classes": CLASSES_PUBLIC,
                    "selected_version": None, "checks": [], "scope": SCOPE,
                    "identity_scope": IDENTITY_SCOPE, "gpu_required": False}
        try:
            manifest, prediction, version = self._snapshot(collection_id, prediction_version)
            response["selected_version"] = {key: version.get(key) for key in ("id", "name", "prediction_sha256")}
            checks = [self.check(collection_id, path.parent.name) for path in self._plans(collection_id)]
            response["checks"] = list(reversed([check for check in checks
                                                if check["prediction_version"] == version["id"]]))
            existing = next((check for check in checks if check["prediction_version"] == version["id"]), None)
            if existing:
                response.update(available=True, frame_size=existing["progress"]["frame_size"],
                                reason="This prediction version already has a fixed check; open its saved check")
            else:
                frame, excluded, counts, _ = self._frame(collection_id, manifest, prediction)
                response.update(available=True, can_create=bool(frame), frame_size=len(frame),
                    excluded_images=len(excluded), exclusion_counts=counts,
                    reason=None if frame else "No eligible images remain for a fresh check")
        except (ValueError, OSError) as error:
            response.update(available=False, can_create=False, reason=str(error))
        return response

    def create(self, collection_id, *, reviewer, check_size=50, prediction_version=None):
        reviewer = reviewer_name(reviewer)
        if not isinstance(check_size, int) or isinstance(check_size, bool) or not 20 <= check_size <= 200:
            raise ValueError("Choose a fixed check size from 20 to 200 images")
        with self._sampling_lock(), self.collections.locked(collection_id):
            manifest, prediction, version = self._snapshot(collection_id, prediction_version)
            for path in self._plans(collection_id):
                plan = checked(path)
                if plan["prediction_version"] == version["id"]:
                    return self.check(collection_id, path.parent.name)
            frame, excluded, counts, sources = self._frame(collection_id, manifest, prediction)
            if not frame:
                raise ValueError("No eligible images remain for a fresh check")
            check_id, seed = uuid.uuid4().hex, secrets.randbits(63)
            size = min(check_size, len(frame))
            indexes = np.random.default_rng(seed).choice(len(frame), size=size, replace=False)
            freeze(self.folder(collection_id, check_id) / "plan.json", {
                "schema": "collection-scan-check-plan-v1", "id": check_id, "collection_id": collection_id,
                "created_at": now(), "reviewer": reviewer, "reviewer_identity": "self_declared",
                "manifest_sha256": manifest["sha256"], "prediction_sha256": prediction["sha256"],
                "prediction_version": version["id"], "version_name": version["name"],
                "frame": frame, "frame_size": len(frame), "sample": [frame[int(i)] for i in indexes],
                "sample_size": size, "requested_check_size": check_size, "seed": seed,
                "inclusion_probability": size / len(frame),
                "sampling": "Uniform simple random sample without replacement; census if frame is smaller than requested size",
                "exclusions": excluded, "exclusion_counts": counts,
                "exclusion_counts_overlap": True, "known_development_sources": sources,
                "permanently_excluded_from_training": True, "scope": SCOPE, "identity_scope": IDENTITY_SCOPE,
                "model_training": False, "predictions_modified": False})
            return self.check(collection_id, check_id)

    def unit(self, collection_id, check_id):
        current = self.check(collection_id, check_id)
        if current["status"] == "closed":
            raise ValueError("This check was stopped; its sample and saved responses are preserved")
        plan, _ = self._plan(collection_id, check_id)
        labels, _ = self._responses(collection_id, check_id, plan)
        remaining = [row for row in plan["sample"] if row["id"] not in labels]
        response = {"complete": not remaining, "image": None, **current["progress"],
                    "classes": CLASSES_PUBLIC,
                    "scope": "Assign one tissue category without seeing the classifier prediction. Unsure does not replace this image."}
        if remaining:
            row = remaining[0]
            response.update(position=len(labels) + 1, image={"id": row["id"],
                "image_url": f"api/collections/{collection_id}/images/{row['id']}/image"})
        return response

    def save_label(self, collection_id, check_id, *, image_id, label, reviewer, note=""):
        reviewer = reviewer_name(reviewer)
        if label not in CLASSES + ["unsure"] or not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Choose a tissue category or Unsure, with a note up to 2,000 characters")
        with self.collections.locked(collection_id):
            current = self.check(collection_id, check_id)
            if current["status"] != "ready":
                raise ValueError("This check is already evaluated or stopped; saved responses are immutable")
            plan, _ = self._plan(collection_id, check_id)
            sample = {row["id"]: row for row in plan["sample"]}
            if image_id not in sample:
                raise ValueError("This image is not a planned check unit")
            labels, unsure = self._responses(collection_id, check_id, plan)
            if image_id in labels:
                old = labels[image_id]
                if old["label"] == label and old["reviewer"] == reviewer and old["note"] == note.strip():
                    return current
                raise ValueError("A submitted check label is immutable")
            # Enforce frozen review order so outcomes cannot choose which unit
            # receives a reference. A deferred Unsure remains the current unit.
            if image_id != next(row["id"] for row in plan["sample"] if row["id"] not in labels):
                raise ValueError("Label the current planned check image first")
            record = {"created_at": now(), "collection_id": collection_id, "check_id": check_id,
                "plan_sha256": plan["sha256"], "image_id": image_id,
                "image_sha256": sample[image_id]["image_sha256"], "reviewer": reviewer,
                "reviewer_identity": "self_declared", "label": label, "note": note.strip(), "role": "check"}
            if label == "unsure":
                if not any(all(row[key] == record[key] for key in ("image_id", "reviewer", "label", "note")) for row in unsure):
                    freeze(self.folder(collection_id, check_id) / "unsure" / (uuid.uuid4().hex + ".json"), record)
            else:
                freeze(self.folder(collection_id, check_id) / "labels" / (image_id + ".json"), record)
            return self.check(collection_id, check_id)

    def evaluate(self, collection_id, check_id):
        with self.collections.locked(collection_id):
            current = self.check(collection_id, check_id)
            if current["status"] == "evaluated":
                return current
            if current["status"] == "closed":
                raise ValueError("This check was stopped; it cannot be evaluated")
            plan, prediction = self._plan(collection_id, check_id)
            labels, unsure = self._responses(collection_id, check_id, plan)
            if len(labels) != plan["sample_size"]:
                raise ValueError("Label every planned check image before evaluating; unsure images cannot be dropped or replaced")
            freeze(self.folder(collection_id, check_id) / "result.json", {
                "schema": "collection-scan-check-result-v1", "created_at": now(),
                "collection_id": collection_id, "check_id": check_id,
                **self._evidence(plan, labels, unsure), **self._assessment(plan, prediction, labels)})
            return self.check(collection_id, check_id)

    def close(self, collection_id, check_id, *, reviewer, note=""):
        reviewer = reviewer_name(reviewer)
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Use a note up to 2,000 characters")
        with self.collections.locked(collection_id):
            current = self.check(collection_id, check_id)
            if current["status"] == "closed":
                return current
            if not current["can_close"]:
                raise ValueError("Only an incomplete check can be stopped")
            plan, _ = self._plan(collection_id, check_id)
            labels, unsure = self._responses(collection_id, check_id, plan)
            freeze(self.folder(collection_id, check_id) / "closed.json", {
                "schema": "collection-scan-check-closure-v1", "created_at": now(),
                "collection_id": collection_id, "check_id": check_id, "reviewer": reviewer,
                "reviewer_identity": "self_declared", "note": note.strip(),
                **self._evidence(plan, labels, unsure), "sample_reservations_preserved": True,
                "check_replaced": False, "predictions_modified": False})
            return self.check(collection_id, check_id)

    def export(self, collection_id, check_id):
        current = self.check(collection_id, check_id)
        plan, prediction = self._plan(collection_id, check_id)
        labels, unsure = self._responses(collection_id, check_id, plan)
        value = {"schema": "collection-scan-check-export-v1", "exported_at": now(),
                 "collection_id": collection_id, "check_id": check_id, "plan": plan,
                 "labels": [labels[row["id"]] for row in plan["sample"] if row["id"] in labels],
                 "unsure_responses": unsure, "check": current, "scope": SCOPE, "identity_scope": IDENTITY_SCOPE,
                 "predictions_disclosed": current["status"] == "evaluated",
                 "reproduction": "Use the recorded seed and ordered frame to reproduce the sample. After complete labeling, compare frozen predictions with reviewer labels and invert the hypergeometric distribution for frame accuracy."}
        if current["status"] == "evaluated":
            by_id = {row["id"]: row for row in prediction["predictions"]}
            value["sample_evidence"] = [{**row, "reference_label": labels[row["id"]]["label"],
                "predicted_label": by_id[row["id"]]["predicted_label"],
                "correct": labels[row["id"]]["label"] == by_id[row["id"]]["predicted_label"],
                "image_url": f"api/collections/{collection_id}/images/{row['id']}/image"} for row in plan["sample"]]
        return {**value, "sha256": digest(value)}

    def exports(self, collection_id, prediction_version):
        return [self.export(collection_id, path.parent.name) for path in self._plans(collection_id)
                if checked(path).get("prediction_version") == prediction_version]
