"""Collection-local appearance search and immutable image-backed findings.

Only completed, hash-bound CONCH scan caches are used. Imported images retain
their unlabeled inspection role; relevance feedback never changes predictions.
The text encoder is shared with the prepared library, but its images, labels,
ranker, and saved records are never consulted.
"""
from collections import OrderedDict
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from .collection_scan import image_binding
from .collection_regions import CollectionRegions, MIN_SIDE, MAX_SIDE, MAX_REGIONS
from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, MODEL, REPRESENTATION, file_hash
from .feedback_storage import freeze
from .semantic import ASSESSMENTS, EXAMPLES, bounded_text, refine_vector
from .semantic_examples import validate_examples
from .storage import digest, now, read
from .study import verify

SCOPE = ("Appearance search within this uploaded collection. Scores rank visual similarity; "
         "they are not accuracy, a diagnosis, or a measured tissue proportion. "
         "Saved findings contain researcher-selected images and reviewer judgments.")
VERSION = "collection-personal-relevance-rbf-v1"
RBF_ALGORITHM = {"kernel": "rbf", "C": 10.0, "gamma": 5.0, "class_weight": "balanced"}
SOURCE_SHA256 = file_hash(__file__)


class SearchUnavailable(ValueError):
    """The collection cannot safely supply a search index yet."""


def signature(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SearchUnavailable("A required collection search artifact is missing or is a symbolic link")
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class CollectionSemanticStore:
    def __init__(self, collections, identifier, shared_encoder, region_encoder=None):
        self.collections, self.collection_id = collections, identifier
        self.root = collections.path(identifier)
        self.directory = self.root / "semantic"
        if self.directory.is_symlink():
            raise SearchUnavailable("Collection search storage must remain inside its collection")
        self._dependencies = {}
        self._shared_encoder = shared_encoder
        metadata = self._read_bound(self.root / "collection.json")
        self.collection = {"id": identifier, "name": metadata["name"]}
        scan_path = self.root / "scan.json"
        scan = read(scan_path)
        self._dependencies[scan_path] = signature(scan_path)
        if scan.get("status") != "complete":
            raise SearchUnavailable("Finish scanning this collection before searching its images")
        if shared_encoder is None:
            raise SearchUnavailable("The CONCH text encoder is not configured for this server")
        self.encoder_config = dict(shared_encoder.encoder_config)
        verify(self.encoder_config)
        if (self.encoder_config.get("source_checkpoint_sha256") != CHECKPOINT_SHA256 or
                self.encoder_config.get("code_revision") != CODE_REVISION):
            raise SearchUnavailable("Collection images and text encoder must use the same pinned CONCH model")
        encoder_path = Path(shared_encoder.directory) / "encoder.json"
        if self._read_bound(encoder_path) != self.encoder_config:
            raise SearchUnavailable("The configured text encoder record has changed")
        manifest = self._read_bound(self.root / "manifest.json")
        predictions = self._read_bound(self.root / "predictions.json")
        rows = manifest.get("images", [])
        if (metadata.get("id") != identifier or manifest.get("collection_id") != identifier or
                manifest.get("role") != "unlabeled_inspection" or not rows or len(rows) > 10000 or
                predictions.get("collection_id") != identifier or
                predictions.get("manifest_sha256") != manifest["sha256"] or
                scan.get("prediction_sha256") != predictions["sha256"]):
            raise SearchUnavailable("Collection identities or frozen prediction bindings differ")
        self.ids = [row["id"] for row in rows]
        if (len(set(self.ids)) != len(rows) or
                any(not re.fullmatch(r"[a-f0-9]{32}", i) for i in self.ids) or
                len({row["image_sha256"] for row in rows}) != len(rows)):
            raise SearchUnavailable("Collection search needs unique frozen images")
        self.rows = {row["id"]: row for row in rows}
        self.positions = {identifier: i for i, identifier in enumerate(self.ids)}
        model = predictions.get("model", {})
        identity = model.get("encoder_identity", {})
        if (model.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                model.get("code_revision") != CODE_REVISION or
                model.get("retrieval_representation") != REPRESENTATION or
                identity.get("model") != MODEL or identity.get("representation") != REPRESENTATION or
                identity.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                model.get("manifest_sha256") != manifest["sha256"]):
            raise SearchUnavailable("Search requires verified projected CONCH image features from this collection")
        binding = {"manifest_sha256": manifest["sha256"],
                   "configuration_sha256": model.get("configuration_sha256"),
                   "encoder_identity_sha256": digest(identity),
                   "worker_source_sha256": model.get("worker_source_sha256")}
        chunks = model.get("feature_chunks", [])
        if not chunks or len(chunks) > len(rows):
            raise SearchUnavailable("Completed image feature caches are unavailable for this collection")
        arrays, offset = [], 0
        for index, receipt in enumerate(chunks):
            expected = f"batch-{index:06d}.npz"
            if receipt.get("file") != expected:
                raise SearchUnavailable("Collection feature chunks are missing or out of order")
            path = self.root / "features" / expected
            meta = self._read_bound(path.with_suffix(".json"))
            self._dependencies[path] = signature(path)
            count = len(meta.get("images", []))
            selected = rows[offset:offset + count]
            if (not count or len(selected) != count or meta.get("binding") != binding or
                    meta.get("images") != image_binding(selected) or meta.get("file") != expected or
                    receipt.get("metadata_sha256") != meta["sha256"] or
                    receipt.get("sha256") != meta.get("file_sha256") or
                    file_hash(path) != meta.get("file_sha256")):
                raise SearchUnavailable("Collection feature caches differ from their frozen scan receipts")
            with np.load(path, allow_pickle=False) as cache:
                if set(cache.files) != {"ids", "classifier_embeddings", "embeddings"}:
                    raise SearchUnavailable("Unexpected collection feature arrays")
                values = cache["embeddings"].copy()
                if (cache["ids"].tolist() != [row["id"] for row in selected] or
                        values.shape != (count, 512) or values.dtype != np.float32 or
                        not np.isfinite(values).all() or
                        not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)):
                    raise SearchUnavailable("Invalid or misaligned projected CONCH collection features")
            arrays.append(values)
            offset += count
        if offset != len(rows):
            raise SearchUnavailable("Search features do not cover every collection image")
        self.features = np.concatenate(arrays)
        self.features.setflags(write=False)
        provenance = {"schema": "collection-semantic-catalog-v1", "collection": self.collection,
                      "collection_sha256": metadata["sha256"], "manifest_sha256": manifest["sha256"],
                      "prediction_sha256": predictions["sha256"], "encoder_sha256": self.encoder_config["sha256"],
                      "feature_chunks": chunks, "model": MODEL, "source_checkpoint_sha256": CHECKPOINT_SHA256,
                      "representation": REPRESENTATION, "scope": SCOPE, "role": "unlabeled_inspection"}
        self.catalog = {**provenance, "sha256": digest(provenance)}
        self.regions = CollectionRegions(self, region_encoder)
        self.assert_current()

    def _read_bound(self, path):
        initial = signature(path)
        result = read(path)
        verify(result)
        if signature(path) != initial:
            raise SearchUnavailable("A collection search artifact changed while it was read")
        self._dependencies[path] = initial
        return result

    def assert_current(self):
        for path, expected in self._dependencies.items():
            if signature(path) != expected:
                raise SearchUnavailable("Collection search artifacts changed; restore the frozen scan before continuing")
        if self.directory.is_symlink():
            raise SearchUnavailable("Collection search storage has changed")

    def row(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.rows:
            raise ValueError("Image does not belong to this collection")
        return self.rows[identifier]

    def image_url(self, identifier):
        self.row(identifier)
        return f"api/collections/{self.collection_id}/semantic/images/{identifier}"

    def image_path(self, identifier):
        self.assert_current()
        row = self.row(identifier)
        path = self.root / "images" / (identifier + ".png")
        signature(path)
        if path.resolve().parent != self.root / "images" or file_hash(path) != row["image_sha256"]:
            raise ValueError("Collection image bytes differ from the frozen source")
        return path

    def _path(self, kind, identifier):
        if kind not in ("searches", "concepts", "descriptions"):
            raise ValueError("Unknown collection search record type")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid saved-record ID")
        path = self.directory / kind / (identifier + ".json")
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("Collection search records must remain inside their collection")
        return path

    def _save(self, kind, record):
        self.assert_current()
        value = {"schema": "collection-" + kind + "-v1", "id": uuid.uuid4().hex,
                 "created_at": now(), "collection": self.collection, "collection_id": self.collection_id,
                 "catalog_sha256": self.catalog["sha256"], "provenance": self.catalog,
                 "scope": SCOPE, "model": MODEL, **record}
        return freeze(self._path(kind, value["id"]), value)

    def load(self, kind, identifier):
        self.assert_current()
        result = read(self._path(kind, identifier))
        verify(result)
        if (result.get("id") != identifier or result.get("collection_id") != self.collection_id or
                result.get("catalog_sha256") != self.catalog["sha256"] or
                result.get("provenance") != self.catalog):
            raise ValueError("Saved record belongs to a different collection or frozen feature index")
        if kind == "searches":
            validate_examples(self, result["positive_ids"], result["negative_ids"])
            reference = result.get("image_query")
            region = result.get("region_query")
            if reference and region:
                raise ValueError("Saved search cannot combine an image and a region query")
            if region:
                _, expected = self.regions.query(region["id"], encode=False)
                if (region != expected or region["image_id"] in result["positive_ids"] + result["negative_ids"] or
                        any(item["id"] == region["image_id"] for item in result["results"]) or
                        result.get("prediction_version") != region["prediction_version"]):
                    raise ValueError("Saved search region reference changed")
            if reference:
                row = self.row(reference["id"])
                if (reference != self.image_query(row["id"]) or
                        row["id"] in result["positive_ids"] + result["negative_ids"] or
                        any(item["id"] == row["id"] for item in result["results"])):
                    raise ValueError("Saved search image reference changed")
                self.image_path(row["id"])
            if result.get("prediction_version"):
                version = result["prediction_version"]
                if self.collections.prediction_view(self.collection_id, version["id"])[1] != version:
                    raise ValueError("Saved search prediction version has changed")
            for row in result["results"]:
                if (self.row(row["id"])["image_sha256"] != row["image_sha256"] or
                        row["image_url"] != self.image_url(row["id"])):
                    raise ValueError("Saved search image evidence changed")
        if kind == "concepts":
            query = self.load("searches", result["search_id"])
            if query["sha256"] != result["search_sha256"]:
                raise ValueError("Saved search definition changed")
        if kind == "descriptions":
            if result.get("prediction_version"):
                version = result["prediction_version"]
                _, current = self.collections.prediction_view(self.collection_id, version["id"])
                if current != version:
                    raise ValueError("Saved finding prediction version has changed")
            for claim in result["claims"]:
                if (claim["text"] not in result["text"] or claim["assessment"] not in ASSESSMENTS or
                        [e["image_id"] for e in claim["evidence"] if not e.get("region_id")] != claim["image_ids"] or
                        [e["region_id"] for e in claim["evidence"] if e.get("region_id")] != claim.get("region_ids", [])):
                    raise ValueError("Saved finding claims or evidence IDs changed")
                for evidence in claim["evidence"]:
                    if evidence.get("region_id"):
                        if (evidence != self.regions.evidence(evidence["region_id"]) or
                                evidence["prediction_version"] != result.get("prediction_version")):
                            raise ValueError("Saved finding region evidence changed")
                        continue
                    if (self.row(evidence["image_id"])["image_sha256"] != evidence["image_sha256"] or
                            evidence["image_url"] != self.image_url(evidence["image_id"])):
                        raise ValueError("Saved finding evidence differs from this collection")
            if result.get("parent_id"):
                parent = read(self._path("descriptions", result["parent_id"]))
                verify(parent)
                if (parent["sha256"] != result["parent_sha256"] or
                        parent["catalog_sha256"] != self.catalog["sha256"] or
                        parent["collection_id"] != self.collection_id):
                    raise ValueError("Saved finding parent belongs to another collection or has changed")
        return result

    def public(self):
        self.assert_current()
        paths = self._record_paths("concepts")[:30]
        concepts = [{key: row[key] for key in ("id", "name", "search_id")} for row in
                    (self.load("concepts", path.stem) for path in paths)]
        descriptions = self.list_descriptions()
        return {"available": True, "population": len(self.ids), "collection": self.collection,
                "selected_prediction_version": self.collections.prediction_view(self.collection_id)[1],
                "scope": SCOPE, "catalog_sha256": self.catalog["sha256"], "model": MODEL,
                "query_limit": 600, "examples": EXAMPLES, "gallery_available": True,
                "image_search_available": True,
                "region_evidence_available": True,
                "region_search_available": self.regions.encoder is not None,
                "region_min_size_px": MIN_SIDE, "region_limit": MAX_REGIONS,
                "region_max_size_px": MAX_SIDE,
                "guidance_available": False, "language_interpretation_available": False,
                "retrieval_version": VERSION, "evaluation_asset": None, "concepts": concepts,
                "descriptions": descriptions["descriptions"], "descriptions_total": descriptions["total"]}

    def _record_paths(self, kind):
        directory = self.directory / kind
        if directory.is_symlink():
            raise ValueError("Collection search records must remain inside their collection")
        return sorted(directory.glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)

    def list_descriptions(self, offset=0, limit=30):
        self.assert_current()
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Use a nonnegative finding offset and a page size between 1 and 100")
        paths = self._record_paths("descriptions")
        descriptions = [{key: row.get(key) for key in ("id", "text", "reviewer", "created_at", "prediction_version")} for row in
                        (self.load("descriptions", path.stem) for path in paths[offset:offset + limit])]
        return {"descriptions": descriptions, "total": len(paths), "offset": offset, "limit": limit,
                "next_offset": offset + limit if offset + limit < len(paths) else None,
                "collection": self.collection}

    def gallery(self):
        self.assert_current()
        # Evenly spaced frozen manifest positions provide a bounded overview.
        # No labels, classifier scores, or prepared-library examples enter it.
        # The initial screen must span the entire population even when import
        # order groups sources or tissue types. Additional examples fill the
        # largest remaining gaps, without labels or embedding-dependent choices.
        positions = np.linspace(0, len(self.ids) - 1, min(20, len(self.ids)), dtype=int).tolist()
        while len(positions) < min(40, len(self.ids)):
            ordered = sorted(positions)
            left, right = max(zip(ordered, ordered[1:]), key=lambda pair: (pair[1] - pair[0], -pair[0]))
            positions.append((left + right) // 2)
        return {"schema": "collection-semantic-gallery-v1", "catalog_sha256": self.catalog["sha256"],
                "collection": self.collection, "primary_count": min(20, len(positions)),
                "selection": "Initial 20 evenly span the full frozen import order; additional images fill remaining gaps. Not a probability sample or an error audit",
                "scope": SCOPE, "results": [{"id": self.ids[i], "filename": self.rows[self.ids[i]]["filename"],
                    "image_url": self.image_url(self.ids[i])} for i in positions]}

    def image_query(self, identifier):
        row = self.row(identifier)
        return {"id": identifier, "filename": row["filename"],
                "image_sha256": row["image_sha256"], "image_url": self.image_url(identifier),
                "role": "query reference only; not a relevance label or selected evidence"}

    def search(self, text="", positive_ids=None, negative_ids=None, limit=24, guidance_id=None,
               image_id=None, prediction_version=None, region_id=None):
        self.assert_current()
        if guidance_id is not None:
            raise ValueError("Prepared-library guidance does not apply to uploaded collections")
        if image_id is not None and region_id is not None:
            raise ValueError("Choose one image or one selected region as the query")
        text = bounded_text(text, 600, "Search text", allow_empty=image_id is not None or region_id is not None)
        positive, negative = validate_examples(self, positive_ids or [], negative_ids or [])
        if type(limit) is not int or not 1 <= limit <= 60:
            raise ValueError("Choose between 1 and 60 search results")
        region = self.regions.load(region_id) if region_id is not None else None
        if region and prediction_version is None:
            prediction_version = region["prediction_version"]["id"]
        _, version = self.collections.prediction_view(self.collection_id, prediction_version)
        if region and region["prediction_version"] != version:
            raise ValueError("Search the selected region with its recorded prediction version")
        reference = None
        if image_id is not None:
            reference = self.image_query(image_id)
            self.image_path(image_id)
            if image_id in positive + negative:
                raise ValueError("The source image is a query reference; choose other images for relevance feedback")
        region_reference, region_vector = None, None
        if region:
            if region["image_id"] in positive + negative:
                raise ValueError("The region source image is a query reference; choose other images for relevance feedback")
            region_vector, region_reference = self.regions.query(region_id)
        text_vector = None
        if text:
            text_vector = np.asarray(self._shared_encoder.encode([text])[0], dtype=np.float32)
            if text_vector.shape != (512,) or not np.isfinite(text_vector).all() or not np.isclose(np.linalg.norm(text_vector), 1, atol=1e-4):
                raise ValueError("Invalid projected CONCH text features")
        original = region_vector.copy() if region else self.features[self.positions[image_id]].copy() if reference else text_vector
        visual_query = reference is not None or region is not None
        if visual_query and text_vector is not None:
            original = original + 0.5 * text_vector
            original /= np.linalg.norm(original)
        vector = refine_vector(original, self.features[[self.positions[i] for i in positive]],
                               self.features[[self.positions[i] for i in negative]])
        scores = self.features @ vector
        query_scores = self.features @ original
        text_scores = self.features @ text_vector if text_vector is not None else None
        base = ("region" if region else "image") + (" + 0.5 text" if text else "") if visual_query else "text"
        method = "normalized " + base + " + 0.6 mean relevant images - 0.3 mean irrelevant images"
        score_interpretation = "Cosine ranking score, not a probability or verification of a claim"
        notices = ["These matches have not been validated for your collection.",
                   "The nearest images may not contain the pattern you intend.",
                   "Negation, counts and spatial relationships are not enforced by appearance search.",
                   "Reviewed examples are omitted from new suggestions."]
        ranking_provenance = {"version": VERSION, "source_sha256": SOURCE_SHA256,
            "catalog_sha256": self.catalog["sha256"], "encoder_sha256": self.encoder_config["sha256"],
            "representation": REPRESENTATION, "review_count": len(positive) + len(negative),
            "query_mode": ("region_text" if text else "region") if region else "image_text" if reference and text else "image" if reference else "text",
            "image_query": reference, "image_text_weight": 0.5 if visual_query and text else None,
            "reviewed_image_sha256": {i: self.rows[i]["image_sha256"] for i in positive + negative},
            "feedback_overrides_text_ranking": bool(positive and negative),
            "fallback": "collection-conch-rocchio-v1", "binary_model": None}
        if region_reference:
            ranking_provenance["region_query"] = region_reference
            notices.append("The selected rectangle is the query. Results are whole images; matching regions are not located within them.")
        if positive and negative:
            # The same fixed generic algorithm as the prepared-library personal
            # ranker. No prepared taxonomy, labels, calibration or configuration
            # enters this collection's temporary task model.
            import sklearn
            from sklearn.svm import SVC
            features = np.asarray(self.features, dtype=np.float64)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            training_ids = positive + negative
            positions = [self.positions[i] for i in training_ids]
            model = SVC(**RBF_ALGORITHM)
            model.fit(features[positions], np.r_[np.ones(len(positive)), np.zeros(len(negative))])
            scores = model.decision_function(features)
            if scores.shape != (len(self.ids),) or not np.isfinite(scores).all():
                raise ValueError("The collection relevance model returned invalid scores")
            method = "personal_binary_rbf"
            score_interpretation = "Binary relevance decision value, not a probability or verification of a claim"
            binary_model = {"algorithm": RBF_ALGORITHM, "sklearn_version": sklearn.__version__,
                "support_ids": [training_ids[i] for i in model.support_],
                "dual_coefficients": model.dual_coef_.tolist(), "intercept": model.intercept_.tolist(),
                "classes": model.classes_.tolist(), "n_support": model.n_support_.tolist()}
            ranking_provenance["binary_model"] = {**binary_model, "sha256": digest(binary_model)}
            notices.append("Matching and nonmatching examples now determine this search ranking. The query remains recorded as context; tissue predictions and the shared model are unchanged.")
        else:
            notices.append("Mark both matching and nonmatching examples to learn a ranking for this search. Until then, results use exploratory text and example similarity.")
        excluded = set(positive + negative)
        if reference:
            excluded.add(image_id)
            notices.append("The source image is omitted from results. It is a query reference, not a relevance judgment or finding evidence.")
        if region:
            excluded.add(region["image_id"])
            notices.append("The source image is omitted. The selected query region is not automatically added as finding evidence.")
        order = np.lexsort((np.asarray(self.ids), -scores))
        selected = [int(i) for i in order if self.ids[i] not in excluded][:limit]
        results = [{"id": self.ids[i], "filename": self.rows[self.ids[i]]["filename"],
                    "similarity": float(scores[i]), "query_similarity": float(query_scores[i]),
                    "text_similarity": float(text_scores[i]) if text_scores is not None else None,
                    "image_sha256": self.rows[self.ids[i]]["image_sha256"],
                    "image_url": self.image_url(self.ids[i])} for i in selected]
        return self._save("searches", {**({"region_query": region_reference} if region_reference else {}),
            "text": text, "encoded_text": text, "positive_ids": positive,
            "image_query": reference, "prediction_version": version,
            "negative_ids": negative, "guidance_id": None, "guidance": None, "exclusions": "",
            "feedback_purpose": "this search's relevance only; not tissue labels, a shared model update, or a statistical audit",
            "method": method, "ranking_provenance": ranking_provenance,
            "retrieval_version": VERSION, "query_embedding": vector.tolist(),
            "text_embedding": text_vector.tolist() if text_vector is not None else None,
            "results": results, "score_interpretation": score_interpretation, "notices": notices})

    def concept(self, name, search_id, reviewer, note=""):
        query = self.load("searches", search_id)
        return self._save("concepts", {"name": bounded_text(name, 100, "Name"),
            "reviewer": bounded_text(reviewer, 100, "Reviewer"), "note": bounded_text(note, 2000, "Note", True),
            "search_id": search_id, "search_sha256": query["sha256"], "text": query["text"],
            "positive_ids": query["positive_ids"], "negative_ids": query["negative_ids"],
            "image_query": query.get("image_query"), **({"region_query": query["region_query"]} if query.get("region_query") else {}),
            "prediction_version": query.get("prediction_version"),
            "definition_type": "versioned appearance search", "validation_status": "exploratory; not a validated phenotype measurement"})

    def describe(self, text, reviewer, claims, parent_id=None, prediction_version=None):
        self.assert_current()
        text, reviewer = bounded_text(text, 6000, "Finding"), bounded_text(reviewer, 100, "Reviewer")
        if not isinstance(claims, list) or not 1 <= len(claims) <= 20:
            raise ValueError("Link 1–20 explicit claims in the finding")
        parent = self.load("descriptions", parent_id) if parent_id else None
        if prediction_version is None and parent:
            prediction_version = parent.get("prediction_version", {}).get("id", "original")
        if prediction_version is None:
            first_region = next((claim.get("region_ids", [])[0] for claim in claims if claim.get("region_ids")), None)
            if first_region:
                prediction_version = self.regions.load(first_region)["prediction_version"]["id"]
        _, version = self.collections.prediction_view(self.collection_id, prediction_version)
        records = []
        for claim in claims:
            claim_text = bounded_text(claim["text"], 1000, "Claim")
            assessment = claim["assessment"]
            if claim_text not in text or assessment not in ASSESSMENTS:
                raise ValueError("Each claim must occur in the finding and use an explicit reviewer assessment")
            ids, _ = validate_examples(self, claim.get("image_ids", []), [])
            region_ids = claim.get("region_ids", [])
            if (not isinstance(region_ids, list) or len(region_ids) > 8 or
                    any(not isinstance(i, str) or not re.fullmatch(r"[a-f0-9]{64}", i) for i in region_ids) or
                    len(set(region_ids)) != len(region_ids)):
                raise ValueError("Choose up to eight distinct saved region identifiers")
            if len(ids) + len(region_ids) > 8 or (assessment != "unresolved" and not (ids or region_ids)):
                raise ValueError("Supported or contradicted claims need 1–8 evidence images or regions")
            for identifier in ids:
                self.image_path(identifier)
            evidence = [{"image_id": i, "image_sha256": self.row(i)["image_sha256"],
                         "annotation_extent": "whole uploaded image", "image_url": self.image_url(i)} for i in ids]
            for region_id in region_ids:
                region_evidence = self.regions.evidence(region_id)
                if region_evidence["prediction_version"] != version:
                    raise ValueError("Finding regions must use the finding's recorded prediction version")
                evidence.append(region_evidence)
            records.append({"text": claim_text, "assessment": assessment, "image_ids": ids,
                            **({"region_ids": region_ids} if region_ids else {}), "evidence": evidence,
                            "note": bounded_text(claim.get("note", ""), 2000, "Claim note", True),
                            "assessment_source": "reviewer judgment; not automatically verified by the model"})
        return self._save("descriptions", {"text": text, "reviewer": reviewer, "claims": records,
            "prediction_version": version,
            "parent_id": parent_id, "parent_sha256": parent["sha256"] if parent else None,
            "reviewer_expertise_verified": False, "automatic_claim_verification": False,
            "source_image_ids": sorted({e["image_id"] for c in records for e in c["evidence"]})})


class CollectionSemanticManager:
    """At most two collection feature matrices; text weights shared once."""
    def __init__(self, collections, shared_encoder, capacity=2, region_encoder=None):
        self.collections, self.shared_encoder = collections, shared_encoder
        self.region_encoder = region_encoder
        self.capacity, self.cache, self.lock = capacity, OrderedDict(), threading.RLock()

    def get(self, identifier):
        if self.collections is None:
            raise SearchUnavailable("Image collections are not configured for this server")
        self.collections.path(identifier)
        with self.lock:
            if identifier not in self.cache:
                self.cache[identifier] = CollectionSemanticStore(self.collections, identifier, self.shared_encoder, self.region_encoder)
                while len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            result = self.cache[identifier]
            result.assert_current()
            self.cache.move_to_end(identifier)
            return result

    def status(self, identifier):
        if self.collections is None:
            return {"available": False, "reason": "Image collections are not configured", "population": 0,
                    "concepts": [], "descriptions": [], "collection": {"id": identifier, "name": "Collection"}}
        root = self.collections.path(identifier)
        metadata = read(root / "collection.json")
        try:
            return self.get(identifier).public()
        except (ValueError, OSError, KeyError) as error:
            return {"available": False, "reason": str(error), "population": 0,
                    "collection": {"id": identifier, "name": metadata["name"]}, "scope": SCOPE,
                    "concepts": [], "descriptions": [], "gallery_available": False,
                    "guidance_available": False, "evaluation_asset": None}
