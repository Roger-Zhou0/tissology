"""Bounded feedback API; paths and scheduler commands are server configuration."""
import io
import threading
from pathlib import Path

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .conch_infer import file_hash
from .feedback import BLOCKED, FeedbackStore


class FlagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    image_id: str = Field(min_length=1, max_length=100)
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(min_length=1, max_length=2000)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reviewer: str = Field(min_length=1, max_length=100)
    disposition: str
    source: str = "manual_review"
    label: str | None = None
    note: str = Field(min_length=1, max_length=2000)


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def register_feedback(app, directory=None):
    store = FeedbackStore(directory) if directory else None
    mounted = set()
    mount_lock = threading.RLock()

    def evaluation_backend():
        from . import feedback_evaluation
        return feedback_evaluation

    def with_evaluation(candidate):
        try:
            assessment = evaluation_backend().eligibility(store.directory, candidate["id"])
        except ImportError:
            assessment = {"available": False, "eligible_for_new_study": False, "reason": BLOCKED, "status": "not_configured"}
        return {**candidate, "fresh": assessment,
                "eligible_for_new_study": assessment["eligible_for_new_study"],
                "promotion_block": assessment["reason"]}

    def mount_study(study_id):
        with mount_lock:
            if study_id not in mounted:
                path = evaluation_backend().resolve_study(store.directory, study_id)
                from .app import create_app
                app.mount("/feedback-studies/" + study_id, create_app(path))
                mounted.add(study_id)
        return "feedback-studies/" + study_id + "/"


    def configured():
        if store is None:
            raise HTTPException(409, "Feedback development workspace is not configured")
        return store

    @app.get("/feedback")
    def page():
        return FileResponse(Path(__file__).parent / "static" / "feedback.html")

    @app.get("/api/feedback")
    def status():
        return {"available": store is not None, "task": "CRC whole-tile classification",
                "spatial_adaptation_available": False,
                "spatial_block": "Breast MuTILs area adaptation requires approved region masks or complete area-component labels and separate validation. CRC tile labels cannot train it.",
                "promotion_block": "Each frozen candidate must pass its development and fresh-test checks before it can start a separate study.",
                "classes": store.catalog["classes"] if store else [],
                "base_classifier_sha256": store.catalog["base_classifier_sha256"] if store else None,
                "development_examples": len(store.rows) if store else 0,
                "approved_corrections": len(store.corrections()) if store else 0,
                "events": store.events() if store else [], "candidates": [with_evaluation(c) for c in store.candidates()] if store else []}

    @app.get("/api/feedback/examples")
    def examples(offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=48), disagreements: bool = False):
        return configured().examples(offset, limit, disagreements)

    @app.get("/api/feedback/examples/{image_id}")
    def example(image_id: str):
        current = configured()
        return current.public(current.row(image_id))

    @app.get("/api/feedback/examples/{image_id}/neighbors")
    def neighbors(image_id: str):
        return configured().neighbors(image_id)

    @app.get("/api/feedback/images/{image_id}")
    def image(image_id: str):
        row = configured().row(image_id)
        if file_hash(row["path"]) != row["image_sha256"]:
            raise ValueError("Development image bytes changed")
        with Image.open(row["path"]) as source:
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    @app.post("/api/feedback/flags")
    def flag(request: FlagRequest):
        return configured().flag(request.image_id, request.reviewer, request.note)

    @app.post("/api/feedback/flags/{flag_id}/decisions")
    def decision(flag_id: str, request: DecisionRequest):
        return configured().decide(flag_id, request.reviewer, request.disposition, request.label, request.note, request.source)

    @app.post("/api/feedback/candidates")
    def train(request: EmptyRequest):
        return with_evaluation(configured().new_candidate())

    @app.get("/api/feedback/candidates/{candidate_id}")
    def candidate(candidate_id: str):
        return with_evaluation(configured().candidate(candidate_id))

    @app.post("/api/feedback/candidates/{candidate_id}/studies")
    def new_study(candidate_id: str, request: EmptyRequest):
        current = with_evaluation(configured().candidate(candidate_id))
        if not current["eligible_for_new_study"]:
            raise HTTPException(409, current["promotion_block"])
        created = evaluation_backend().create_study(store.directory, candidate_id)
        return {**created, "url": mount_study(created["id"])}

    @app.post("/api/feedback/candidates/{candidate_id}/evaluation")
    def evaluate(candidate_id: str, request: EmptyRequest):
        configured().candidate(candidate_id)
        try:
            return evaluation_backend().request_evaluation(store.directory, candidate_id)
        except ImportError:
            raise HTTPException(409, "Fresh evaluation backend is not configured")

    @app.get("/api/feedback/studies/{study_id}")
    def published_study(study_id: str):
        configured()
        return {"id": study_id, "url": mount_study(study_id)}

    if store:
        try:
            backend = evaluation_backend()
        except ImportError:
            backend = None
        if backend:
            for study_id in backend.list_studies(store.directory):
                mount_study(study_id)
    return store
