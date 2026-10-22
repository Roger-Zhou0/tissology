"""HTTP interface to a configured development image library."""
import io
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .semantic import SemanticStore
from .semantic_gallery import gallery
from .guidance import register_guidance
from .semantic_runtime import SearchRuntime


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequest(StrictRequest):
    text: str = Field(min_length=1, max_length=600)
    positive_ids: list[str] = Field(default_factory=list, max_length=40)
    negative_ids: list[str] = Field(default_factory=list, max_length=40)
    limit: int = Field(default=24, ge=1, le=60)
    guidance_id: str | None = None


class ConceptRequest(StrictRequest):
    name: str = Field(min_length=1, max_length=100)
    search_id: str
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=2000)


class ClaimRequest(StrictRequest):
    text: str = Field(min_length=1, max_length=1000)
    image_ids: list[str] = Field(default_factory=list, max_length=8)
    assessment: str = "unresolved"
    note: str = Field(default="", max_length=2000)


class DescriptionRequest(StrictRequest):
    text: str = Field(min_length=1, max_length=6000)
    reviewer: str = Field(min_length=1, max_length=100)
    parent_id: str | None = None
    claims: list[ClaimRequest] = Field(min_length=1, max_length=20)


def register_semantic(app, directory=None, encoder=None):
    store = SemanticStore(directory, encoder) if directory else None
    guidance = register_guidance(app, store)
    runtime = SearchRuntime(store, guidance) if store else None
    if store and os.environ.get("TISSUE_SEMANTIC_WARMUP") == "1":
        store.encode(["a histopathology image"])

    gallery_cache = None

    def configured():
        if store is None:
            raise HTTPException(409, "Description search library is not configured")
        return store

    @app.get("/semantic")
    def page():
        return FileResponse(Path(__file__).parent / "static/semantic.html")

    @app.get("/api/semantic")
    def status():
        return {**store.public(), "gallery_available": True, **runtime.public()} if store else {"available": False, "population": 0, "concepts": [], "descriptions": []}

    @app.get("/api/semantic/gallery")
    def browse_gallery():
        nonlocal gallery_cache
        if gallery_cache is None:
            configured()
            identifiers = getattr(runtime.ranker, "gallery_ids", None)
            if identifiers:
                gallery_cache = {
                    "schema": "semantic-gallery-v2", "catalog_sha256": store.catalog["sha256"],
                    "retrieval_version": runtime.ranker.version, "primary_count": 20,
                    "selection": "Frozen nested gallery of development-image cluster representatives; no reference labels or query targets used.",
                    "scope": "Twenty initial examples, with twenty additional examples available. Every matching or nonmatching answer counts as one review.",
                    "results": [{"id": identifier, "image_url": "api/semantic/images/" + identifier}
                                for identifier in identifiers]}
            else:
                gallery_cache = gallery(store)
        return gallery_cache

    @app.post("/api/semantic/search")
    def search(request: SearchRequest):
        configured()
        return runtime.search(**request.model_dump())

    @app.get("/api/semantic/searches/{identifier}")
    def query(identifier: str):
        return configured().load("searches", identifier)

    @app.get("/api/semantic/images/{identifier}")
    def image(identifier: str):
        with Image.open(configured().image_path(identifier)) as source:
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    @app.post("/api/semantic/concepts")
    def concept(request: ConceptRequest):
        return configured().concept(**request.model_dump())

    @app.get("/api/semantic/concepts/{identifier}")
    def saved_concept(identifier: str):
        return configured().load("concepts", identifier)

    @app.post("/api/semantic/descriptions")
    def description(request: DescriptionRequest):
        return configured().describe(**request.model_dump())

    @app.get("/api/semantic/descriptions/{identifier}")
    def saved_description(identifier: str):
        return configured().load("descriptions", identifier)

    @app.get("/api/semantic/descriptions/{identifier}/export")
    def export(identifier: str):
        return JSONResponse(configured().load("descriptions", identifier), headers={
            "Content-Disposition": f'attachment; filename="description-{identifier}.json"'})

    return store
