"""Persist task-scoped searches while preserving the original retrieval protocol."""
from pathlib import Path
import json
import uuid

import numpy as np

from .feedback_storage import freeze
from .semantic import bounded_text, refine_vector
from .semantic_examples import validate_examples
from .storage import now


class OriginalRanker:
    """Compatibility scorer, explicitly identified; not a precision upgrade."""
    version = "raw-conch-v1"
    evaluation_asset = "static/semantic-evaluation.json"

    def __init__(self, store):
        self.store = store

    def rank(self, text, positive_ids, negative_ids, exclusions=""):
        store = self.store
        original = store.encode([text])[0]
        vector = refine_vector(original, store.features[[store.positions[i] for i in positive_ids]],
                               store.features[[store.positions[i] for i in negative_ids]])
        scores = store.features @ vector
        provenance = {"version": self.version, "text_embedding": original.tolist(),
                      "query_embedding": vector.tolist(), "catalog_sha256": store.catalog["sha256"]}
        notices = []
        if exclusions:
            unwanted = store.encode([exclusions])[0]
            scores = scores - .3 * (store.features @ unwanted)
            provenance.update(exclusion_embedding=unwanted.tolist(), exclusion_weight=.3)
            notices.append("Unwanted appearance influences ranking; this is not a verified tissue or spatial exclusion.")
        return {"scores": scores, "text_scores": store.features @ original,
                "method": "Original CONCH text/example similarity with separately encoded unwanted appearance",
                "provenance": provenance, "notices": notices,
                "score_interpretation": "Ranking scores, not calibrated probabilities or evidence that a claim is true"}


def load_selected_ranker(store):
    path = store.directory / "ranking-config.json"
    if not path.exists():
        return OriginalRanker(store)
    # A malformed configured version fails closed, without substituting another model.
    from .semantic_v2_ranker import load_ranker
    ranker = load_ranker(store, path)
    language_path = store.directory / "language-config.json"
    if language_path.exists():
        schema = json.loads(language_path.read_text()).get("schema")
        if schema == "crc-language-runtime-v2":
            from .semantic_language_ranker_v2 import LanguageGroundedRankerV2
            ranker = LanguageGroundedRankerV2(store, ranker, language_path)
        elif schema == "crc-language-runtime-v1":
            from .semantic_language_ranker import LanguageGroundedRanker
            ranker = LanguageGroundedRanker(store, ranker, language_path)
        else:
            raise ValueError("Unknown language interpreter version")
    return ranker


class SearchRuntime:
    def __init__(self, store, guidance, ranker=None):
        self.store, self.guidance = store, guidance
        self.ranker = ranker if ranker is not None else load_selected_ranker(store)

    def public(self):
        evaluation_asset = self.ranker.evaluation_asset
        if evaluation_asset is None and self.ranker.version == "personal-relevance-rbf-v2-002":
            evaluation_asset = "static/semantic-evaluation-v2.json"
        return {"guidance_available": True,
                "exclusion_text_supported": isinstance(self.ranker, OriginalRanker),
                "language_interpretation_available": getattr(self.ranker, "language_interpretation_available", False),
                "retrieval_version": self.ranker.version,
                "evaluation_asset": evaluation_asset}

    def search(self, text, positive_ids=None, negative_ids=None, limit=24, guidance_id=None):
        store = self.store
        text = bounded_text(text, 600, "Search text")
        positive, negative = validate_examples(store, positive_ids or [], negative_ids or [])
        if type(limit) is not int or not 1 <= limit <= 60:
            raise ValueError("Choose between 1 and 60 search results")
        applied = self.guidance.application(guidance_id) if guidance_id else None
        if applied and (text != applied["effective_query"] or
                        set(positive) != set(applied["positive_ids"]) or
                        set(negative) != set(applied["negative_ids"])):
            raise ValueError("Search differs from applied guidance; revise and explicitly apply it again")
        exclusions = applied["exclusions"] if applied else ""
        ranking = self.ranker.rank(text, positive, negative, exclusions=exclusions)
        scores, text_scores = np.asarray(ranking["scores"]), np.asarray(ranking["text_scores"])
        if (scores.shape != (len(store.ids),) or text_scores.shape != scores.shape or
                not np.isfinite(scores).all() or not np.isfinite(text_scores).all()):
            raise ValueError("Retrieval returned invalid or misaligned scores")
        excluded = set(positive + negative)
        order = np.lexsort((np.asarray(store.ids), -scores))
        selected = [int(i) for i in order if store.ids[i] not in excluded][:limit]
        results = [{"id": store.ids[i], "similarity": float(scores[i]), "text_similarity": float(text_scores[i]),
                    "development_label": store.rows[store.ids[i]]["development_label"],
                    "image_sha256": store.rows[store.ids[i]]["image_sha256"],
                    "image_url": "api/semantic/images/" + store.ids[i]} for i in selected]
        notices = ["The closest available tiles may not contain the pattern you intend.",
                   "Quantities and spatial relationships require explicit measurements; appearance ranking does not establish them.",
                   "Reviewed examples are omitted from the new suggestions.", *ranking.get("notices", [])]
        if applied:
            # Catch a revision/retraction during ranking before publishing its result.
            if self.guidance.application(guidance_id)["sha256"] != applied["sha256"]:
                raise ValueError("Guidance changed during search")
        identifier = uuid.uuid4().hex
        return freeze(store.directory / "searches" / (identifier + ".json"), {
            "schema": "task-scoped-search-v2", "id": identifier, "created_at": now(),
            "catalog_sha256": store.catalog["sha256"], "text": text, "encoded_text": text,
            "positive_ids": positive, "negative_ids": negative, "guidance_id": guidance_id,
            "guidance": applied, "exclusions": exclusions,
            "retrieval_version": self.ranker.version, "method": ranking["method"],
            "ranking_provenance": ranking["provenance"], "results": results,
            "feedback_purpose": "this task's relevance only; not tissue labels or a global model update",
            "scope": store.catalog["scope"], "score_interpretation": ranking["score_interpretation"],
            "notices": notices})
