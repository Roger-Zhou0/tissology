"""Visual collection summaries and frozen, targeted inspection batches.

These helpers consult predictions, cached image features and researcher reviews.
They never read reference labels or construct an accuracy/audit sample. A batch
is an immutable selection plan; ordinary review history supplies its progress.
"""
from pathlib import Path
import re

import numpy as np

from .collection_index import checked
from .collection_reviews import CollectionReviews, ReviewConflict
from .collection_storage import CLASSES, CLASS_NAMES
from .feedback_storage import freeze
from .storage import digest, now

BATCH_SCOPE = ("Targeted inspection across predicted classes and appearances. This is not a "
               "probability sample and does not estimate collection accuracy. Saved assessments "
               "remain ordinary researcher reviews; no model is trained by starting or finishing a batch.")
_ID = re.compile(r"^[a-f0-9]{32}$")


def quantiles(rows, count):
    """Deterministic evenly spaced ranks; never duplicate a short class."""
    if len(rows) <= count:
        return list(rows)
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def priority(row):
    return (float(row.get("margin", row.get("confidence", 1))), row["id"])


class CollectionWorkflows:
    def __init__(self, collections):
        self.collections = collections

    def population(self, identifier, version):
        summary = self.collections.summary(identifier, version)
        page = self.collections.images(identifier, 0, self.collections.limits["max_images"],
                                       prediction_version=summary["selected_prediction_version"]["id"])
        rows = [row for row in page["images"] if row.get("predicted_label") in CLASSES and not row.get("error")]
        return summary, rows

    def overview(self, identifier, version=None):
        summary, rows = self.population(identifier, version)
        selected = summary["selected_prediction_version"]
        flag_reasons = {"input_format": 0, "low_score": 0, "close_alternatives": 0, "other": 0}
        reason_keys = {"native format outside the validated tile task": "input_format",
                       "top score below 0.8": "low_score", "top-two score margin below 0.2": "close_alternatives"}
        flagged = [row for row in rows if row.get("uncertain")]
        for row in flagged:
            categories = {reason_keys.get(reason, "other") for reason in row.get("review_reasons", [])} or {"other"}
            for key in categories:
                flag_reasons[key] += 1
        groups = []
        for label in CLASSES:
            members = [row for row in rows if row["predicted_label"] == label]
            examples = quantiles(sorted(members, key=lambda row: (float(row.get("confidence", 0)), row["id"])), 3)
            groups.append({"code": label, "name": CLASS_NAMES[label], "count": len(members),
                           "reviewed": sum(bool(row.get("review")) for row in members),
                           "reviewed_current_version": sum(bool(row.get("review")) and
                               row["review"].get("prediction_version", "original") == selected["id"] for row in members),
                           "examples": examples})
        return {"collection_id": identifier, "selected_prediction_version": selected,
                "manifest_sha256": summary["manifest_sha256"], "classes": groups,
                "flag_reasons": {"flagged_count": len(flagged), "counts": flag_reasons,
                                 "scope": "Recorded reasons among flagged images on this version; categories overlap. Flags are not confirmed errors."},
                "selection": "Up to three examples per predicted class, spanning its model-score range. "
                             "Examples are not expert-verified or a representative sample.",
                "scope": "Counts are predicted images, not measured tissue areas or cell counts."}

    def _root(self, identifier):
        path = self.collections.path(identifier) / "review-batches"
        if path.is_symlink():
            raise ValueError("Review batches must stay inside their collection")
        return path

    def _path(self, identifier, batch_id):
        if not isinstance(batch_id, str) or not _ID.fullmatch(batch_id):
            raise ValueError("Invalid review batch identity")
        path = self._root(identifier) / (batch_id + ".json")
        if path.is_symlink():
            raise ValueError("Review batches must stay inside their collection")
        return path

    def _reserved(self, identifier, rows):
        manager = getattr(self.collections, "teaching", None)
        if manager is None:
            return set()
        if hasattr(manager, "reservations"):
            from .collection_teaching import identities
            ids, hashes = manager.reservations()
            return {row["id"] for row in rows if (identifier, row["id"]) in ids or identities(row) & hashes}
        return {row["id"] for row in rows if manager.is_reserved(identifier, row["id"])}

    def _features(self, identifier, rows):
        manager = getattr(self.collections, "semantic", None)
        if manager is None:
            return None, {"method": "class_score_spread", "reason": "Verified image features are not configured."}
        from .collection_semantic import SearchUnavailable
        try:
            index = manager.get(identifier)
        except SearchUnavailable:
            return None, {"method": "class_score_spread", "reason": "Verified image features are unavailable."}
        index.assert_current()
        positions = {image_id: position for position, image_id in enumerate(index.ids)}
        if any(row["id"] not in positions for row in rows):
            raise ValueError("Review candidates do not match the cached image features")
        return {row["id"]: index.features[positions[row["id"]]] for row in rows}, {
            "method": "class_cached_appearance", "catalog_sha256": index.catalog["sha256"],
            "representation": index.catalog.get("representation"),
            "reason": "Cached image appearance is used within each predicted class; no inference is run."}

    def create(self, identifier, *, size=20, prediction_version="original", client_request_id):
        if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 50:
            raise ValueError("Choose between 1 and 50 images")
        path = self._path(identifier, client_request_id)
        request = {"size": size, "prediction_version": prediction_version, "client_request_id": client_request_id}
        with self.collections.locked(identifier):
            if path.exists():
                saved = checked(path)
                if saved.get("request_sha256") != digest(request):
                    raise ReviewConflict("This batch identity belongs to a different request")
                return self.detail(identifier, client_request_id)
            summary, rows = self.population(identifier, prediction_version)
            if summary["status"] != "complete" or not summary["prediction_sha256"]:
                raise ValueError("Finish a scan before starting a review batch")
            reserved = self._reserved(identifier, rows)
            candidates = [row for row in rows if not row.get("review") and row["id"] not in reserved]
            if not candidates:
                raise ValueError("No unreviewed images are eligible for a new batch. Separate-check images are excluded.")
            vectors, feature_record = self._features(identifier, candidates)
            groups = {label: sorted([row for row in candidates if row["predicted_label"] == label], key=priority)
                      for label in CLASSES}
            order = sorted([label for label in CLASSES if groups[label]], key=lambda label: (-len(groups[label]), label))
            pools = {label: quantiles(groups[label], 256) for label in order}
            chosen = {label: [] for label in order}
            sample = []
            while len(sample) < min(size, len(candidates)):
                for label in order:
                    remaining = [row for row in pools[label] if row["id"] not in chosen[label]]
                    if not remaining:
                        continue
                    if not chosen[label]:
                        selected = remaining[0]
                        reason = "Lowest class-score margin in this predicted category."
                    elif vectors is not None:
                        prior = np.stack([vectors[image_id] for image_id in chosen[label]])
                        selected = min(remaining, key=lambda row: (
                            float(np.max(prior @ vectors[row["id"]])), *priority(row)))
                        reason = "Different cached image appearance within this predicted category."
                    else:
                        ranks = {row["id"]: i for i, row in enumerate(pools[label])}
                        selected = min(remaining, key=lambda row: (
                            -min(abs(ranks[row["id"]] - ranks[image_id]) for image_id in chosen[label]), *priority(row)))
                        reason = "Different model-score rank within this predicted category; appearance features unavailable."
                    chosen[label].append(selected["id"])
                    sample.append({"image_id": selected["id"], "image_sha256": selected["image_sha256"],
                                   "predicted_label": label, "confidence": selected.get("confidence"),
                                   "margin": selected.get("margin"), "flagged": bool(selected.get("uncertain")),
                                   "reason": reason, "review_revision_at_start": 0})
                    if len(sample) >= min(size, len(candidates)):
                        break
            plan = {"schema": "collection-review-batch-v1", "id": client_request_id, "collection_id": identifier,
                    "created_at": now(), "manifest_sha256": summary["manifest_sha256"],
                    "prediction_version": summary["selected_prediction_version"], "requested_size": size,
                    "request_sha256": digest(request), "eligible_count": len(candidates),
                    "reserved_excluded": len(reserved), "sample": sample, "role": "targeted_inspection",
                    "features": feature_record,
                    "selection": {"algorithm": "class-interleaved-diverse-inspection-v1",
                                  "class_order": order, "max_candidates_per_class": 256,
                                  "candidate_rule": "Unreviewed successfully predicted images, excluding permanent teaching-check reservations.",
                                  "pool_rule": "Evenly spaced class-margin ranks, capped at 256 per class.",
                                  "within_class": "Lowest-margin seed, then farthest cosine appearance from selected images; "
                                                  "farthest score-rank fallback when verified features are unavailable.",
                                  "tie_break": "Ascending class margin, then image ID.",
                                  "probability_sample": False, "inclusion_probabilities": None},
                    "scope": BATCH_SCOPE}
            freeze(path, plan)
        return self.detail(identifier, client_request_id)

    def _plan(self, identifier, batch_id):
        path = self._path(identifier, batch_id)
        if not path.is_file():
            raise FileNotFoundError("Unknown review batch")
        plan = checked(path)
        if plan.get("collection_id") != identifier or plan.get("id") != batch_id or plan.get("schema") != "collection-review-batch-v1":
            raise ValueError("Review batch belongs to another collection")
        summary = self.collections.summary(identifier, plan["prediction_version"]["id"])
        if (summary["manifest_sha256"] != plan["manifest_sha256"] or
                summary["prediction_sha256"] != plan["prediction_version"]["prediction_sha256"]):
            raise ValueError("Review batch no longer matches its frozen images and predictions")
        return plan

    def detail(self, identifier, batch_id):
        plan = self._plan(identifier, batch_id)
        observations = CollectionReviews(self.collections.path(identifier))
        reserved = self._reserved(identifier, [self.collections.image(identifier, row["image_id"]) for row in plan["sample"]])
        progress = []
        for row in plan["sample"]:
            _, history = observations.image(row["image_id"])
            matches = [event for event in history if event["revision"] > row["review_revision_at_start"] and
                       event.get("prediction_version", "original") == plan["prediction_version"]["id"] and
                       event["image_sha256"] == row["image_sha256"] and
                       event["prediction_sha256"] == plan["prediction_version"]["prediction_sha256"]]
            review = matches[-1] if matches else None
            progress.append({"image_id": row["image_id"], "review_sha256": review["sha256"] if review else None,
                             "disposition": review["disposition"] if review else None,
                             "status": "reviewed" if review else "reserved" if row["image_id"] in reserved else "pending"})
        remaining = [row["image_id"] for row in progress if row["status"] == "pending"]
        return {"plan": plan, "progress": progress, "reviewed": sum(row["status"] == "reviewed" for row in progress),
                "reserved": sum(row["status"] == "reserved" for row in progress), "remaining": len(remaining),
                "next_image_id": remaining[0] if remaining else None, "complete": not remaining,
                "progress_source": "Saved ordinary reviews for this exact prediction version; Unsure is a completed assessment."}

    def listing(self, identifier, version=None):
        selected = self.collections.summary(identifier, version)["selected_prediction_version"]
        paths = sorted(self._root(identifier).glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        result = []
        for path in paths:
            plan = checked(path)
            if plan.get("prediction_version", {}).get("id") != selected["id"]:
                continue
            result.append(self.detail(identifier, path.stem))
            if len(result) >= 20:
                break
        return {"collection_id": identifier, "prediction_version": selected, "batches": result, "scope": BATCH_SCOPE}
