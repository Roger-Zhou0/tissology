"""Collection-scoped routes sharing only the pinned text encoder with the library."""
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import Field, StrictInt

from .collection_semantic import CollectionSemanticManager, SearchUnavailable
from .collection_regions import RegionBusy
from .semantic_api import ClaimRequest, ConceptRequest, DescriptionRequest, SearchRequest, StrictRequest


class CollectionSearchRequest(SearchRequest):
    text: str = Field(default="", max_length=600)
    image_id: str | None = Field(default=None, pattern="^[a-f0-9]{32}$")
    region_id: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")
    prediction_version: str | None = Field(default=None, pattern="^(original|[a-f0-9]{32})$")


class CollectionClaimRequest(ClaimRequest):
    region_ids: list[str] = Field(default_factory=list, max_length=8)


class CollectionDescriptionRequest(DescriptionRequest):
    claims: list[CollectionClaimRequest] = Field(min_length=1, max_length=20)
    prediction_version: str | None = Field(default=None, pattern="^(original|[a-f0-9]{32})$")


class CollectionRegionRequest(StrictRequest):
    image_id: str = Field(pattern="^[a-f0-9]{32}$")
    bounds_px: list[StrictInt] = Field(min_length=4, max_length=4)
    prediction_version: str | None = Field(default=None, pattern="^(original|[a-f0-9]{32})$")


def register_collection_semantic(app, collections, shared_encoder, region_encoder=None):
    manager = CollectionSemanticManager(collections, shared_encoder, region_encoder=region_encoder)
    if collections is not None:
        collections.semantic = manager
    prefix = "/api/collections/{collection_id}/semantic"

    def configured(collection_id):
        try:
            return manager.get(collection_id)
        except SearchUnavailable as error:
            raise HTTPException(409, str(error))

    @app.get(prefix)
    def status(collection_id: str):
        return manager.status(collection_id)

    @app.get(prefix + "/gallery")
    def gallery(collection_id: str):
        return configured(collection_id).gallery()

    @app.post(prefix + "/search")
    def search(collection_id: str, request: CollectionSearchRequest):
        try:
            return configured(collection_id).search(**request.model_dump())
        except RegionBusy as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"})
        except RuntimeError as error:
            raise HTTPException(503, "Region search is temporarily unavailable; the selected region remains saved.") from error

    @app.post(prefix + "/regions", status_code=201)
    def region(collection_id: str, request: CollectionRegionRequest):
        try:
            return configured(collection_id).regions.create(**request.model_dump())
        except RegionBusy as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"})

    @app.get(prefix + "/regions/{identifier}")
    def saved_region(collection_id: str, identifier: str):
        return configured(collection_id).regions.load(identifier)

    @app.get(prefix + "/regions/{identifier}/image")
    def region_image(collection_id: str, identifier: str):
        return Response(configured(collection_id).regions.image_bytes(identifier), media_type="image/png")

    @app.get(prefix + "/searches/{identifier}")
    def saved_search(collection_id: str, identifier: str):
        return configured(collection_id).load("searches", identifier)

    @app.get(prefix + "/images/{identifier}")
    def image(collection_id: str, identifier: str):
        return FileResponse(configured(collection_id).image_path(identifier), media_type="image/png")

    @app.post(prefix + "/concepts")
    def concept(collection_id: str, request: ConceptRequest):
        return configured(collection_id).concept(**request.model_dump())

    @app.get(prefix + "/concepts/{identifier}")
    def saved_concept(collection_id: str, identifier: str):
        return configured(collection_id).load("concepts", identifier)

    @app.post(prefix + "/descriptions")
    def description(collection_id: str, request: CollectionDescriptionRequest):
        return configured(collection_id).describe(**request.model_dump())

    @app.get(prefix + "/descriptions")
    def descriptions(collection_id: str, offset: int = Query(default=0, ge=0),
                     limit: int = Query(default=30, ge=1, le=100)):
        return configured(collection_id).list_descriptions(offset, limit)

    @app.get(prefix + "/descriptions/{identifier}")
    def saved_description(collection_id: str, identifier: str):
        return configured(collection_id).load("descriptions", identifier)

    @app.get(prefix + "/descriptions/{identifier}/export")
    def export(collection_id: str, identifier: str):
        return JSONResponse(configured(collection_id).load("descriptions", identifier), headers={
            "Content-Disposition": f'attachment; filename="collection-{collection_id}-finding-{identifier}.json"'})

    return manager
