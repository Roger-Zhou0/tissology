"""Small CPU-only collection teaching API; no model downloads or GPU scheduling."""
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .collection_teaching import CollectionTeachingManager


class CreateCandidate(BaseModel):
    reviewer: str = Field(min_length=1, max_length=120)
    consent: bool
    check_size: int = Field(default=50, ge=20, le=200)


class CheckLabel(BaseModel):
    image_id: str
    label: str
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


class ApplyCandidate(BaseModel):
    reviewer: str = Field(min_length=1, max_length=120)


class CloseCandidate(BaseModel):
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


def register_collection_teaching(app, collections, manager=None):
    manager = manager or CollectionTeachingManager(collections)
    prefix = "/api/collections/{collection_id}/teaching"

    def call(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (ValueError, FileNotFoundError, BlockingIOError) as error:
            raise HTTPException(409, str(error)) from error

    @app.get(prefix)
    def status(collection_id: str):
        return call(manager.status, collection_id)

    @app.post(prefix + "/candidates")
    def create(collection_id: str, request: CreateCandidate):
        return call(manager.create, collection_id, **request.model_dump())

    @app.get(prefix + "/candidates/{candidate_id}")
    def candidate(collection_id: str, candidate_id: str):
        return call(manager.candidate, collection_id, candidate_id)

    @app.get(prefix + "/candidates/{candidate_id}/check")
    def check(collection_id: str, candidate_id: str):
        return call(manager.check_unit, collection_id, candidate_id)

    @app.post(prefix + "/candidates/{candidate_id}/check")
    def label(collection_id: str, candidate_id: str, request: CheckLabel):
        return call(manager.save_check_label, collection_id, candidate_id, **request.model_dump())

    @app.post(prefix + "/candidates/{candidate_id}/evaluate")
    def evaluate(collection_id: str, candidate_id: str):
        return call(manager.evaluate, collection_id, candidate_id)

    @app.post(prefix + "/candidates/{candidate_id}/apply")
    def apply(collection_id: str, candidate_id: str, request: ApplyCandidate):
        return call(manager.apply, collection_id, candidate_id, **request.model_dump())

    @app.post(prefix + "/candidates/{candidate_id}/close")
    def close(collection_id: str, candidate_id: str, request: CloseCandidate):
        return call(manager.close, collection_id, candidate_id, **request.model_dump())

    @app.get(prefix + "/versions/{version_id}")
    def version(collection_id: str, version_id: str):
        return call(manager.version_record, collection_id, version_id)

    @app.get(prefix + "/versions/{version_id}/export")
    def export(collection_id: str, version_id: str):
        return call(manager.export, collection_id, version_id)

    return manager
