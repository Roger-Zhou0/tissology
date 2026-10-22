"""Standalone scan-check HTTP routes; no training or inference."""
from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .collection_checks import CollectionChecksManager


class CreateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=120)
    check_size: int = Field(default=50, ge=20, le=200, strict=True)
    prediction_version: str = Field(default="original", pattern="^(original|[a-f0-9]{32})$")


class CheckLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    label: str
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


class CloseCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


def register_collection_checks(app, collections, teaching, manager=None):
    manager = manager or CollectionChecksManager(collections, teaching)
    prefix = "/api/collections/{collection_id}/checks"

    def call(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (ValueError, FileNotFoundError, BlockingIOError) as error:
            raise HTTPException(409, str(error)) from error

    @app.get(prefix)
    def status(collection_id: str, prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        return call(manager.status, collection_id, prediction_version)

    @app.post(prefix)
    def create(collection_id: str, request: CreateCheck):
        return call(manager.create, collection_id, **request.model_dump())

    @app.get(prefix + "/{check_id}")
    def detail(collection_id: str, check_id: str):
        return call(manager.check, collection_id, check_id)

    @app.get(prefix + "/{check_id}/unit")
    def unit(collection_id: str, check_id: str):
        return call(manager.unit, collection_id, check_id)

    @app.post(prefix + "/{check_id}/labels")
    def label(collection_id: str, check_id: str, request: CheckLabel):
        return call(manager.save_label, collection_id, check_id, **request.model_dump())

    @app.post(prefix + "/{check_id}/evaluate")
    def evaluate(collection_id: str, check_id: str):
        return call(manager.evaluate, collection_id, check_id)

    @app.post(prefix + "/{check_id}/close")
    def close(collection_id: str, check_id: str, request: CloseCheck):
        return call(manager.close, collection_id, check_id, **request.model_dump())

    @app.get(prefix + "/{check_id}/export")
    def export(collection_id: str, check_id: str):
        return JSONResponse(call(manager.export, collection_id, check_id), headers={
            "Content-Disposition": f'attachment; filename="scan-check-{check_id}.json"'})

    return manager
