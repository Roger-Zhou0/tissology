"""Appearance search, explicit relevance feedback, and reviewer-linked descriptions."""
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from .conch_infer import CHECKPOINT_SHA256, MODEL, REPRESENTATION, file_hash
from .feedback_storage import freeze
from .storage import now, read
from .study import verify

POSITIVE_WEIGHT = 0.6
NEGATIVE_WEIGHT = 0.3
EXAMPLES = ["many small dark round dots packed closely together",
            "large empty round spaces with thin pink borders",
            "wavy pink strands with scattered thin dark marks"]
ASSESSMENTS = {"supported_by_reviewer", "contradicted_by_reviewer", "unresolved"}


def bounded_text(value, maximum, name, allow_empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not value.strip() and not allow_empty):
        raise ValueError(f"{name} must contain {'0' if allow_empty else '1'}–{maximum} characters")
    return value.strip()


def refine_vector(text_vector, positives=None, negatives=None):
    """Fixed Rocchio relevance feedback in one projected CONCH space."""
    vector = np.asarray(text_vector, dtype=np.float64).copy()
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Invalid query embedding")
    for values, weight in ((positives, POSITIVE_WEIGHT), (negatives, -NEGATIVE_WEIGHT)):
        if values is not None and len(values):
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[1] != 512 or not np.isfinite(values).all():
                raise ValueError("Invalid feedback embeddings")
            vector += weight * values.mean(axis=0)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Feedback cancels the query; change examples")
    return (vector / norm).astype(np.float32)


class SemanticStore:
    def __init__(self, directory, encoder=None):
        self.directory = Path(directory).resolve()
        self.catalog = read(self.directory / "catalog.json")
        verify(self.catalog)
        if (self.catalog["model"] != MODEL or self.catalog["representation"] != REPRESENTATION or
                self.catalog["source_checkpoint_sha256"] != CHECKPOINT_SHA256):
            raise ValueError("Semantic library needs pinned projected CONCH features")
        if file_hash(self.directory / "index.npz") != self.catalog["index_sha256"]:
            raise ValueError("Semantic index changed")
        with np.load(self.directory / "index.npz", allow_pickle=False) as cache:
            self.ids = cache["ids"].tolist()
            self.features = cache["features"].copy()
        rows = self.catalog["examples"]
        if (not rows or any(r["role"] != "train" for r in rows) or
                self.ids != [r["id"] for r in rows] or len(set(self.ids)) != len(rows) or
                len({r["image_sha256"] for r in rows}) != len(rows) or
                self.features.shape != (len(rows), 512) or not np.isfinite(self.features).all() or
                not np.allclose(np.linalg.norm(self.features, axis=1), 1, atol=1e-5)):
            raise ValueError("Only unique, bound training-role images may enter semantic search")
        self.rows = {r["id"]: r for r in rows}
        self.positions = {identifier: i for i, identifier in enumerate(self.ids)}
        self.encoder_config = read(self.directory / "encoder.json")
        verify(self.encoder_config)
        if self.encoder_config["sha256"] != self.catalog["encoder_sha256"]:
            raise ValueError("Semantic text encoder binding changed")
        self._encoder = encoder
        self._encoder_lock = threading.Lock()

    def encode(self, texts):
        with self._encoder_lock:
            if self._encoder is None:
                from .semantic_encoder import TextEncoder
                self._encoder = TextEncoder(self.encoder_config)
        return self._encoder.encode(texts)

    def _id(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Invalid saved-record ID")
        return identifier

    def row(self, identifier):
        if identifier not in self.rows:
            raise ValueError("Image is not in the development search library")
        return self.rows[identifier]

    def image_path(self, identifier):
        row = self.row(identifier)
        if file_hash(row["path"]) != row["image_sha256"]:
            raise ValueError("Source image bytes changed")
        return row["path"]

    def _examples(self, identifiers):
        if not isinstance(identifiers, list) or len(identifiers) > 20 or len(set(identifiers)) != len(identifiers):
            raise ValueError("Choose up to 20 distinct library examples per feedback type")
        for identifier in identifiers:
            self.row(identifier)
        return identifiers

    def load(self, kind, identifier):
        if kind not in ("searches", "concepts", "descriptions"):
            raise ValueError("Unknown record type")
        result = read(self.directory / kind / (self._id(identifier) + ".json"))
        verify(result)
        if result["id"] != identifier or result["catalog_sha256"] != self.catalog["sha256"]:
            raise ValueError("Saved record differs from this library")
        if kind == "concepts":
            query = self.load("searches", result["search_id"])
            if query["sha256"] != result["search_sha256"]:
                raise ValueError("Saved search definition changed")
        if kind == "descriptions":
            for claim in result["claims"]:
                if claim["text"] not in result["text"] or claim["assessment"] not in ASSESSMENTS:
                    raise ValueError("Saved claim differs from its description")
                if [e["image_id"] for e in claim["evidence"]] != claim["image_ids"]:
                    raise ValueError("Saved claim evidence IDs differ")
                for evidence in claim["evidence"]:
                    if self.row(evidence["image_id"])["image_sha256"] != evidence["image_sha256"]:
                        raise ValueError("Saved evidence differs from its source image")
            if result["parent_id"]:
                parent = read(self.directory / "descriptions" / (self._id(result["parent_id"]) + ".json"))
                verify(parent)
                if parent["sha256"] != result["parent_sha256"] or parent["catalog_sha256"] != self.catalog["sha256"]:
                    raise ValueError("Saved description parent changed")
        return result

    def public(self):
        summaries = {}
        for kind, keys in (("concepts", ("id", "name", "search_id")),
                           ("descriptions", ("id", "text", "reviewer"))):
            paths = sorted((self.directory / kind).glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)[:30]
            summaries[kind] = [{key: row[key] for key in keys} for row in
                               (self.load(kind, path.stem) for path in paths)]
        return {"available": True, "population": len(self.ids), "scope": self.catalog["scope"],
                "model": MODEL, "query_limit": 600, "examples": EXAMPLES, **summaries}

    def search(self, text, positive_ids=None, negative_ids=None, limit=24):
        text = bounded_text(text, 600, "Search text")
        positive = self._examples(positive_ids or [])
        negative = self._examples(negative_ids or [])
        if set(positive) & set(negative):
            raise ValueError("An example cannot both match and not match")
        if not isinstance(limit, int) or not 1 <= limit <= 60:
            raise ValueError("Choose between 1 and 60 search results")
        original = self.encode([text])[0]
        vector = refine_vector(original, self.features[[self.positions[i] for i in positive]],
                               self.features[[self.positions[i] for i in negative]])
        scores, text_scores = self.features @ vector, self.features @ original
        order = np.lexsort((np.array(self.ids), -scores))
        # Reviewed examples are excluded from the new suggestions, so improvement
        # cannot consist merely of resurfacing the examples the user supplied.
        excluded = set(positive + negative)
        selected = [int(i) for i in order if self.ids[i] not in excluded][:limit]
        results = [{"id": self.ids[i], "similarity": float(scores[i]), "text_similarity": float(text_scores[i]),
                    "development_label": self.rows[self.ids[i]]["development_label"],
                    "image_sha256": self.rows[self.ids[i]]["image_sha256"],
                    "image_url": "api/semantic/images/" + self.ids[i]} for i in selected]
        identifier = uuid.uuid4().hex
        return freeze(self.directory / "searches" / (identifier + ".json"), {
            "id": identifier, "created_at": now(), "catalog_sha256": self.catalog["sha256"],
            "text": text, "encoded_text": text, "positive_ids": positive, "negative_ids": negative,
            "feedback_purpose": "query relevance only; not tissue labels or model training",
            "method": "normalized text + 0.6 mean relevant images - 0.3 mean irrelevant images",
            "query_embedding": vector.tolist(), "text_embedding": np.asarray(original).tolist(), "results": results,
            "scope": self.catalog["scope"], "score_interpretation": "Cosine ranking score, not probability or evidence that a claim is true",
            "notices": ["The closest available tiles may not contain the pattern you intend.",
                        "Negation, quantities and spatial relationships are not enforced by this appearance search.",
                        "Reviewed examples are omitted from the new suggestions."]})

    def concept(self, name, search_id, reviewer, note=""):
        query = self.load("searches", search_id)
        identifier = uuid.uuid4().hex
        return freeze(self.directory / "concepts" / (identifier + ".json"), {
            "id": identifier, "created_at": now(), "catalog_sha256": self.catalog["sha256"],
            "name": bounded_text(name, 100, "Name"), "reviewer": bounded_text(reviewer, 100, "Reviewer"),
            "note": bounded_text(note, 2000, "Note", True), "search_id": search_id, "search_sha256": query["sha256"],
            "definition_type": "versioned appearance search", "validation_status": "exploratory; not a validated phenotype measurement",
            "model": MODEL, "positive_ids": query["positive_ids"], "negative_ids": query["negative_ids"], "text": query["text"]})

    def describe(self, text, reviewer, claims, parent_id=None):
        text, reviewer = bounded_text(text, 6000, "Description"), bounded_text(reviewer, 100, "Reviewer")
        if not isinstance(claims, list) or not 1 <= len(claims) <= 20:
            raise ValueError("Link 1–20 explicit claims in the description")
        parent = self.load("descriptions", parent_id) if parent_id else None
        records = []
        for claim in claims:
            claim_text = bounded_text(claim["text"], 1000, "Claim")
            if claim_text not in text:
                raise ValueError("Each claim must occur in the current description text")
            assessment = claim["assessment"]
            if assessment not in ASSESSMENTS:
                raise ValueError("Unknown reviewer assessment")
            ids = self._examples(claim.get("image_ids", []))
            if len(ids) > 8 or (assessment != "unresolved" and not ids):
                raise ValueError("Supported or contradicted claims need 1–8 evidence tiles")
            evidence = [{"image_id": i, "image_sha256": self.row(i)["image_sha256"],
                         "annotation_extent": "whole development tile", "image_url": "api/semantic/images/" + i} for i in ids]
            records.append({"text": claim_text, "assessment": assessment, "image_ids": ids, "evidence": evidence,
                            "note": bounded_text(claim.get("note", ""), 2000, "Claim note", True),
                            "assessment_source": "reviewer judgment; not automatically verified by the model"})
        identifier = uuid.uuid4().hex
        return freeze(self.directory / "descriptions" / (identifier + ".json"), {
            "id": identifier, "created_at": now(), "catalog_sha256": self.catalog["sha256"],
            "text": text, "reviewer": reviewer, "claims": records, "parent_id": parent_id,
            "parent_sha256": parent["sha256"] if parent else None, "scope": self.catalog["scope"],
            "reviewer_expertise_verified": False, "automatic_claim_verification": False,
            "model": MODEL, "source_image_ids": sorted({i for c in records for i in c["image_ids"]})})
