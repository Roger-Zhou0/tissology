"""HTTP collection intake, frozen predictions, and separate researcher reviews."""
import asyncio
import csv
import io
import json
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .collection_storage import CLASS_NAMES, CLASSES, LIMITS, SCOPE, CollectionStore
from .collection_reviews import ReviewConflict
from .collection_workflows import CollectionWorkflows


class CollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    source_kind: str = "upload"
    folder_path: str | None = Field(default=None, max_length=2000)
    declared_domain: str = "unknown"
    auto_scan: bool = False
    client_request_id: str | None = Field(default=None, min_length=32, max_length=64)


class ImageReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=120)
    disposition: str
    label: str | None = None
    note: str = Field(default="", max_length=2000)
    expected_revision: int = Field(ge=0, strict=True)
    client_request_id: str = Field(pattern="^[a-f0-9-]{32,64}$")
    prediction_version: str = Field(default="original", pattern="^(original|[a-f0-9]{32})$")


class ReviewBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = Field(default=20, ge=1, le=50, strict=True)
    prediction_version: str = Field(default="original", pattern="^(original|[a-f0-9]{32})$")
    client_request_id: str = Field(pattern="^[a-f0-9]{32}$")


def register_collections(app, directory=None, import_roots=None, submit_scan=None, limits=None):
    store = CollectionStore(directory, import_roots=import_roots, limits=limits) if directory else None
    workflows = CollectionWorkflows(store) if store else None

    def scan_configured():
        if store is None or submit_scan is None:
            return False
        if callable(getattr(submit_scan, 'available', None)):
            return submit_scan.available()
        return True

    @app.get("/api/collections/validation")
    def active_validation():
        """Show evidence for this workspace's configured model, including rollback."""
        from .storage import digest
        records = {
            'c55b4f70800235d2a7882099c877de8a0c76e61bceb523f2bb426196e73b7dce':
                'collection-validation.json',
            '1593966cc99c8b99317ba1492c92116694cc876fcaa01cb9c1eb3da263a5b8ea':
                'collection-validation-conch-retention-001.json',
        }
        try:
            if store is None:
                raise ValueError('Collections are not configured')
            config = json.loads((store.directory / 'scan-model.json').read_text())
            if config.get('sha256') != digest({k: v for k, v in config.items() if k != 'sha256'}):
                raise ValueError('Model configuration is not intact')
            filename = records.get(config.get('classifier_sha256'))
            if filename is None:
                raise ValueError('No matching model evidence')
            record = json.loads((Path(__file__).parent / 'static' / filename).read_text())
            if record.get('classifier_sha256') != config.get('classifier_sha256'):
                raise ValueError('Evidence identifies a different model')
            if 'sha256' in record and record['sha256'] != digest({k: v for k, v in record.items() if k != 'sha256'}):
                raise ValueError('Model evidence is not intact')
            return JSONResponse(record, headers={'Cache-Control': 'no-store'})
        except (OSError, ValueError, TypeError):
            return JSONResponse({'available': False, 'reason': 'Model-matched benchmark evidence is unavailable'},
                                headers={'Cache-Control': 'no-store'})

    def configured():
        if store is None:
            raise HTTPException(409, "User collections are not configured for this server")
        return store

    def finish_and_maybe_scan(identifier):
        result = configured().finalize(identifier)
        if result.get("auto_scan") and result["status"] == "ready" and scan_configured():
            result = store.request_scan(identifier, submit_scan)
        return result

    def folder_import(identifier, admission=None):
        configured().import_folder(identifier, admission=admission)
        result = store.summary(identifier)
        if result.get("auto_scan") and result["status"] == "ready" and scan_configured():
            store.request_scan(identifier, submit_scan)

    def admission(kind, count):
        try:
            return configured().acquire_admission(kind, count)
        except ValueError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"})

    @app.get("/collections")
    def page():
        return FileResponse(Path(__file__).parent / "static/collections.html")

    @app.get("/api/collections")
    def listing():
        if store:
            for collection in store.list():
                store.reconcile_publications(collection["id"])
                store.reconcile_import(collection["id"])
                store.reconcile_scan(collection["id"], submit_scan)
        return {"available": store is not None, "collections": store.list() if store else [],
                "limits": store.limits if store else LIMITS,
                "classes": [{"code": code, "name": CLASS_NAMES[code]} for code in CLASSES],
                "import_roots": [str(path) for path in store.import_roots] if store else [],
                "scan_available": scan_configured(), "scope": SCOPE,
                "capabilities": {"resumable_uploads": True, "filename_search": True, "folder_resume": True, "idempotent_create": True, "image_reviews": True, "visual_overview": True, "review_batches": True},
                "concurrent_uploads": 4, "concurrent_folder_imports": 1}

    @app.post("/api/collections", status_code=201)
    def create(payload: CollectionRequest, tasks: BackgroundTasks):
        if payload.auto_scan and not scan_configured():
            raise HTTPException(409, "Scheduled scanning is unavailable; import for browsing instead")
        slot = admission("folder", 1) if payload.source_kind == "folder" else None
        try:
            result = configured().create(**payload.model_dump())
            if payload.source_kind == "folder" and result["status"] == "importing":
                tasks.add_task(folder_import, result["id"], slot)
                slot = None
            return result
        finally:
            if slot:
                slot.close()

    @app.get("/api/collections/{identifier}")
    def detail(identifier: str, prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        configured().reconcile_publications(identifier)
        store.reconcile_import(identifier)
        store.reconcile_scan(identifier, submit_scan)
        return store.summary(identifier, prediction_version)

    @app.post("/api/collections/{identifier}/files")
    async def upload(identifier: str, request: Request, filename: str = Query(min_length=1, max_length=500),
                     upload_key: str | None = Query(default=None, pattern="^[a-f0-9]{64}$"),
                     source_sha256: str | None = Query(default=None, pattern="^[a-f0-9]{64}$")):
        current = configured()
        # Admission happens before buffering, across processes. No hidden queue of
        # unbounded request bodies; clients receive a retryable, explicit response.
        slot = admission("upload", 4)
        try:
            data = bytearray()
            from starlette.concurrency import run_in_threadpool
            try:
                async with asyncio.timeout(120):
                    async for chunk in request.stream():
                        if len(data) + len(chunk) > current.limits["max_file_bytes"]:
                            result = await run_in_threadpool(current.record_rejection, identifier, filename, "File exceeds the per-image byte limit")
                            return {**result, "collection": current.summary(identifier)}
                        data.extend(chunk)
            except TimeoutError:
                raise HTTPException(408, "Image upload timed out; reselect the same files to resume")
            result = await run_in_threadpool(current.add_file, identifier, filename, bytes(data),
                                             upload_key=upload_key, source_sha256=source_sha256)
            return {**result, "collection": current.summary(identifier)}
        finally:
            slot.close()

    @app.get("/api/collections/{identifier}/uploads")
    def uploads(identifier: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=500, ge=1, le=500)):
        return configured().uploads(identifier, offset, limit)

    @app.post("/api/collections/{identifier}/resume-import")
    def resume_import(identifier: str, tasks: BackgroundTasks):
        current = configured()
        previous = current.summary(identifier)
        slot = admission("folder", 1) if previous["source_kind"] == "folder" else None
        try:
            result = current.resume_import(identifier)
            if result["source_kind"] == "folder":
                tasks.add_task(folder_import, identifier, slot)
                slot = None
            return result
        finally:
            if slot:
                slot.close()

    @app.post("/api/collections/{identifier}/finalize")
    def finalize(identifier: str):
        return finish_and_maybe_scan(identifier)

    @app.post("/api/collections/{identifier}/scan")
    def scan(identifier: str):
        if not scan_configured():
            raise HTTPException(409, "New image scanning is unavailable. Existing images, cached search, teaching and reports remain available.")
        return configured().request_scan(identifier, submit_scan)

    @app.get("/api/collections/{identifier}/images")
    def images(identifier: str, offset: int = Query(default=0, ge=0),
               limit: int = Query(default=40, ge=1, le=100),
               label: str | None = None, uncertain: bool | None = None, q: str | None = Query(default=None, max_length=500),
               review: str = "all", prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        return configured().images(identifier, offset, limit, label, uncertain, q, review, prediction_version)

    @app.get("/api/collections/{identifier}/overview")
    def overview(identifier: str, prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        configured()
        return workflows.overview(identifier, prediction_version)

    @app.post("/api/collections/{identifier}/review-batches", status_code=201)
    def create_review_batch(identifier: str, payload: ReviewBatchRequest):
        configured()
        try:
            return workflows.create(identifier, **payload.model_dump())
        except ReviewConflict as error:
            raise HTTPException(409, str(error))

    @app.get("/api/collections/{identifier}/review-batches")
    def review_batches(identifier: str, prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        configured()
        return workflows.listing(identifier, prediction_version)

    @app.get("/api/collections/{identifier}/review-batches/{batch_id}")
    def review_batch(identifier: str, batch_id: str):
        configured()
        return workflows.detail(identifier, batch_id)

    @app.get("/api/collections/{identifier}/review-batches/{batch_id}/export")
    def export_review_batch(identifier: str, batch_id: str):
        configured()
        return JSONResponse(workflows.detail(identifier, batch_id), headers={
            "Content-Disposition": f'attachment; filename="inspection-batch-{batch_id}.json"'})

    @app.get("/api/collections/{identifier}/images/{image_id}")
    def image_detail(identifier: str, image_id: str, prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        return configured().image_detail(identifier, image_id, prediction_version)

    @app.post("/api/collections/{identifier}/images/{image_id}/review")
    def review_image(identifier: str, image_id: str, payload: ImageReviewRequest):
        try:
            return configured().save_review(identifier, image_id, **payload.model_dump())
        except ReviewConflict as error:
            raise HTTPException(409, str(error))

    @app.get("/api/collections/{identifier}/reviews/export")
    def export_reviews(identifier: str, format: str = Query(default="json", pattern="^(json|csv)$")):
        result = configured().export_reviews(identifier)
        headers = {"Content-Disposition": f'attachment; filename="collection-{identifier[:8]}-reviews.{format}"'}
        if format == "json":
            return JSONResponse(result, headers=headers)
        stream = io.StringIO(newline="")
        columns = ["collection_id", "image_id", "filename", "original_prediction", "reviewer_label", "disposition",
                   "reviewer", "note", "revision", "created_at", "image_sha256", "prediction_sha256", "review_sha256",
                   "model_provenance_sha256", "selection", "role", "reviewer_identity", "prediction_version", "reviewed_prediction"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["latest"]:
            values = {key: row.get(key, "") for key in columns}
            values.update(original_prediction=row["original_prediction"].get("predicted_label", ""),
                          reviewed_prediction=row.get("reviewed_prediction", row["original_prediction"]).get("predicted_label", ""),
                          prediction_version=row.get("prediction_version", "original"),
                          reviewer_label=row["label"] or "", review_sha256=row["sha256"])
            # Prevent filenames, notes, and reviewer names becoming spreadsheet formulas.
            values = {key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value
                      for key, value in values.items()}
            writer.writerow(values)
        return Response(stream.getvalue(), media_type="text/csv; charset=utf-8", headers=headers)

    @app.get("/api/collections/{identifier}/images/{image_id}/thumbnail")
    def thumbnail(identifier: str, image_id: str):
        return FileResponse(configured().image_path(identifier, image_id, thumbnail=True), media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/collections/{identifier}/images/{image_id}/image")
    def image(identifier: str, image_id: str):
        return FileResponse(configured().image_path(identifier, image_id), media_type="image/png",
                            headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/collections/{identifier}/export")
    def export(identifier: str, prediction_version: str | None = Query(default=None, pattern="^(original|[a-f0-9]{32})$")):
        result = configured().export(identifier, prediction_version)
        checks = getattr(store, "checks", None)
        if checks is not None:
            version = result["collection"]["selected_prediction_version"]["id"]
            result["scan_checks"] = checks.exports(identifier, version)
        return JSONResponse(result, headers={
            "Content-Disposition": f'attachment; filename="collection-{identifier[:8]}-preliminary.json"'})

    return store
