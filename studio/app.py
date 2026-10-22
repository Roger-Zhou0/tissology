"""Research interface served on a CPU compute node through Open OnDemand or SSH."""
import io
import os
from pathlib import Path
import threading

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .audit import indicator
from .examples import ExampleStore
from .prepare import ENDPOINTS
from .quality import load_quality, load_test_evaluations
from .storage import read
from .study import Study


class PlanRequest(BaseModel):
    budget: int = Field(default=120, ge=2)
    method: str = "uniform"
    seed: int = Field(default=20261019, ge=0, le=2**32 - 1)


class ReviewRequest(BaseModel):
    image_id: str | None = None
    label: str | None = None
    simulate: bool = False
    simulate_all: bool = False


class SpatialPlanRequest(BaseModel):
    budget: int = Field(default=24, ge=1)
    method: str = "uniform"
    seed: int = Field(default=1729, ge=0)


class SpatialReviewRequest(BaseModel):
    unit_id: str | None = None
    simulate: bool = False
    simulate_all: bool = False


def create_app(directory=None, spatial_directory=None, research_directory=None, feedback_directory=None,
               semantic_directory=None, collections_directory=None, collection_import_roots=None,
               collection_submit_scan=None, collection_scans_enabled=True, lab_directory=None,
               lab_public_origin=None, lab_public_path="/", lab_secure_cookie=True,
               collection_region_encoder=None, collection_text_encoder=None, collection_limits=None):
    if lab_directory:
        credential_root = Path(lab_directory).resolve()
        exposed_roots = [directory, spatial_directory, research_directory, feedback_directory,
                         semantic_directory, collections_directory, Path(__file__).parent / "static",
                         *(collection_import_roots or [])]
        if any(credential_root.is_relative_to(Path(root).resolve()) for root in exposed_roots if root):
            raise ValueError("Lab credentials must be outside scientific, static and image-import directories")
    study = Study(directory) if directory else None
    app = FastAPI(title="Tissology", docs_url=None, redoc_url=None)
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")
    atlas = read(study.directory / "atlas.json") if study and (study.directory / "atlas.json").exists() else None
    embeddings, embedding_ids = None, []
    if study:
        with np.load(study.directory / "predictions-embeddings.npz", allow_pickle=False) as cache:
            embeddings = cache["embeddings"]
            embedding_ids = cache["ids"].tolist()
    embedding_index = {identifier: i for i, identifier in enumerate(embedding_ids)}
    example_store = None
    example_store_lock = threading.Lock()

    def require_study():
        if study is None:
            raise HTTPException(404, "The prepared tile study is not installed on this server")

    def get_example_store():
        # Optional development examples verify a large external provenance
        # graph. Defer those reads until requested, without bypassing any of
        # ExampleStore's validation or publishing a partially initialized store.
        nonlocal example_store
        with example_store_lock:
            if example_store is None:
                example_store = ExampleStore(study.directory, study.manifest, study.predictions)
            return example_store
    spatial = None
    if spatial_directory:
        from .spatial import SpatialStudy
        spatial = SpatialStudy(spatial_directory)
    research = None
    if research_directory:
        from .research_api import register_research
        research = register_research(app, research_directory)

    from .feedback_api import register_feedback
    register_feedback(app, feedback_directory)
    from .semantic_api import register_semantic
    semantic = register_semantic(app, semantic_directory)
    from .collection_api import register_collections
    if not collection_scans_enabled:
        collection_submit_scan = None
    elif collections_directory and collection_submit_scan is None:
        from .collection_scheduler import SlurmScanLauncher
        collection_submit_scan = SlurmScanLauncher(collections_directory)
    collections = register_collections(app, collections_directory,
                                      import_roots=collection_import_roots,
                                      submit_scan=collection_submit_scan, limits=collection_limits)
    if collections is not None:
        from .collection_teaching_api import register_collection_teaching
        collections.teaching = register_collection_teaching(app, collections)
        from .collection_checks_api import register_collection_checks
        collections.checks = register_collection_checks(app, collections, collections.teaching)
    from .collection_semantic_api import register_collection_semantic
    register_collection_semantic(app, collections, collection_text_encoder or semantic,
                                 region_encoder=collection_region_encoder)
    app.state.collections = collections
    from .workspace_catalog import register_workspace
    register_workspace(app, collections=collections_directory, semantic=semantic_directory,
                       research=research_directory,
                       collection_encoder=getattr(collection_text_encoder, "directory", None))

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={"detail": "Requested study artifact was not found"})

    @app.get("/")
    def index(request: Request):
        # Preserve older study links while giving a first visit one shared entry.
        if research is not None and any(key in request.query_params for key in ("study", "audit", "step")):
            return FileResponse(static / "research.html")
        return FileResponse(static / "workspace.html")

    @app.get("/workspace")
    def workspace_page():
        return FileResponse(static / "workspace.html")

    @app.get("/report")
    def report_page():
        return FileResponse(static / "report.html")

    @app.get("/classification", dependencies=[Depends(require_study)])
    def classification_page():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "study_tiles": len(study.rows) if study else 0,
                "classification_available": study is not None,
                "spatial_available": spatial is not None,
                "prediction_sha256": study.predictions["sha256"] if study else None,
                "feedback_available": bool(feedback_directory), "semantic_available": bool(semantic_directory),
                "collections_available": collections is not None,
                "research_available": research is not None,
                "research_windows": len(research.source.rows) if research is not None else 0}

    @app.get("/api/study", dependencies=[Depends(require_study)])
    def summary():
        initial = indicator([r["predicted_label"] for r in study.rows]).mean(axis=0)
        return {"study": study.config, "provenance": study.predictions["provenance"],
                "population": len(study.rows), "pilot": len(study.pilot),
                "spatial_available": spatial is not None,
                "manifest_sha256": study.manifest["sha256"],
                "initial": dict(zip(ENDPOINTS, initial.tolist())),
                "audits": study.audit_summaries(),
                "atlas_description": atlas["method"] if atlas else None,
                "atlas_explained_variance": atlas["explained_variance_fraction"] if atlas else None}

    @app.get("/api/tiles", dependencies=[Depends(require_study)])
    def tiles():
        return [{**row, "coordinates": atlas["points"][row["id"]] if atlas else None}
                for row in study.rows]

    @app.get("/api/model-quality", dependencies=[Depends(require_study)])
    def model_quality():
        # Aggregate evaluation only: per-image references stay behind review.
        return load_quality(study)

    @app.get("/api/test-evaluations", dependencies=[Depends(require_study)])
    def test_evaluations():
        # Aggregate held-out benchmarks; test labels never enter the study audit.
        return load_test_evaluations(study)

    @app.get("/api/images/{image_id}", dependencies=[Depends(require_study)])
    def image(image_id: str):
        if image_id not in study.images:
            raise HTTPException(404, "Unknown image")
        with Image.open(study.images[image_id]["path"]) as source:
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/neighbors/{image_id}", dependencies=[Depends(require_study)])
    def neighbors(image_id: str):
        if image_id not in embedding_index:
            raise HTTPException(404, "Unknown image")
        similarities = embeddings @ embeddings[embedding_index[image_id]]
        order = np.argsort(similarities)[::-1]
        valid = [int(i) for i in order if embedding_ids[i] != image_id and
                 study.images[embedding_ids[i]]["role"] == "study"][:12]
        return [{**study.by_id[embedding_ids[i]], "similarity": float(similarities[i])} for i in valid]

    @app.get("/api/examples/{label}", dependencies=[Depends(require_study)])
    def examples(label: str):
        # Reference labels may be displayed only for this disjoint pilot collection.
        return [{"id": r["id"], "reference_label": study.pilot_labels[r["id"]],
                 "role": "pilot"} for r in study.pilot if study.pilot_labels[r["id"]] == label][:4]

    @app.get("/api/reference-examples/{label}", dependencies=[Depends(require_study)])
    def reference_examples(label: str):
        try:
            return get_example_store().public(label)
        except KeyError:
            raise HTTPException(404, "Unknown tissue class")

    @app.get("/api/reference-example-images/{image_id}", dependencies=[Depends(require_study)])
    def reference_example_image(image_id: str):
        try:
            path = get_example_store().image_path(image_id)
        except KeyError:
            raise HTTPException(404, "Unknown development example")
        with Image.open(path) as source:
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    @app.post("/api/audits", dependencies=[Depends(require_study)])
    def new_audit(request: PlanRequest):
        return study.new_plan(request.budget, request.method, request.seed)

    @app.get("/api/audits/{audit_id}", dependencies=[Depends(require_study)])
    def audit(audit_id: str):
        plan = study.get_plan(audit_id)
        annotations = study.annotations(audit_id)
        return {"plan": plan, "annotations": annotations, "result": study.result(audit_id),
                "predictions": {u["id"]: study.by_id[u["id"]] for u in plan["units"]}}

    @app.post("/api/audits/{audit_id}/review", dependencies=[Depends(require_study)])
    def review(audit_id: str, request: ReviewRequest):
        study.annotate(audit_id, request.image_id, request.label,
                       simulate=request.simulate, simulate_all=request.simulate_all)
        return audit(audit_id)

    @app.get("/api/audits/{audit_id}/export", dependencies=[Depends(require_study)])
    def export(audit_id: str, request: Request):
        bundle = study.export(audit_id)
        mount_prefix = request.scope.get("root_path", "").rstrip("/")
        if mount_prefix:
            for evidence in bundle["evidence"]:
                evidence["relative_image_url"] = mount_prefix + evidence["relative_image_url"]
        return JSONResponse(bundle, headers={
            "Content-Disposition": f'attachment; filename="tissue-study-{audit_id[:8]}.json"'})

    @app.get("/api/evaluation", dependencies=[Depends(require_study)])
    def evaluation():
        # Separate evaluation workspace intentionally reveals population truth.
        # Allocation never consumes this artifact.
        path = study.directory / "evaluation.json"
        if not path.exists():
            return {"available": False, "reason": "No repeated audit-method comparison has been run for this frozen study."}
        return read(path)

    def spatial_study():
        if spatial is None:
            raise HTTPException(404, "Spatial pilot is not configured")
        return spatial

    def spatial_window(identifier):
        current = spatial_study()
        if identifier not in current.by_id:
            raise HTTPException(404, "Unknown spatial window")
        return current.by_id[identifier]

    def png_response(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    def mask_image(mask):
        # Region classes collapse to the three explicitly defined area components.
        palette = np.array([[225, 229, 234], [190, 91, 134], [71, 139, 124], [114, 93, 172],
                            [190, 91, 134], [225, 229, 234], [225, 229, 234], [225, 229, 234],
                            [225, 229, 234]], dtype=np.uint8)
        labels = np.asarray(mask)
        if labels.ndim != 2 or np.any((labels < 0) | (labels > 8)):
            raise ValueError("Invalid spatial mask codes")
        return Image.fromarray(palette[labels.astype(int)])

    @app.get("/spatial")
    def spatial_page():
        spatial_study()
        return FileResponse(static / "spatial.html")

    @app.get("/api/spatial/study")
    def spatial_summary():
        if spatial is None:
            return {"available": False}
        totals = spatial.predicted.sum(axis=0)
        return {"available": True, "study": spatial.config, "population": len(spatial.rows),
                "regions": len(spatial.rois), "windows": spatial.rows,
                "initial": [float(value / totals[3]) if totals[3] else None for value in totals[:3]],
                "predicted_totals": totals.astype(int).tolist(), "audits": spatial.audit_summaries(),
                "target": spatial.frame["target"], "window_side_pixels": spatial.frame["window_side_pixels"],
                "context_pixels": spatial.frame["context_pixels"]}

    @app.get("/api/spatial/windows/{unit_id}/context")
    def spatial_context(unit_id: str):
        from .spatial import check_file
        unit = spatial_window(unit_id)
        roi = spatial.rois[unit["roi_id"]]
        check_file(roi["image_path"], roi["image_sha256"])
        box = unit["context_bounds"]
        with Image.open(roi["image_path"]) as source:
            image = source.convert("RGB").crop(tuple(box[key] for key in ("left", "top", "right", "bottom")))
        bounds = unit["bounds"]
        ImageDraw.Draw(image).rectangle((bounds["left"] - box["left"], bounds["top"] - box["top"],
                                        bounds["right"] - box["left"] - 1, bounds["bottom"] - box["top"] - 1),
                                       outline=(220, 35, 35), width=2)
        return png_response(image)

    @app.get("/api/spatial/windows/{unit_id}/prediction")
    def spatial_prediction(unit_id: str):
        from .spatial import check_file
        unit = spatial_window(unit_id)
        roi = spatial.rois[unit["roi_id"]]
        check_file(roi["prediction_mask_path"], roi["prediction_mask_sha256"])
        with Image.open(roi["prediction_mask_path"]) as source:
            mask = np.asarray(source.crop(tuple(unit["bounds"][key] for key in ("left", "top", "right", "bottom"))))
        return png_response(mask_image(mask))

    @app.post("/api/spatial/audits")
    def new_spatial_audit(request: SpatialPlanRequest):
        return spatial_study().new_plan(request.budget, request.method, request.seed)

    @app.get("/api/spatial/audits/{audit_id}")
    def spatial_audit(audit_id: str):
        current = spatial_study()
        plan = current.get_plan(audit_id)
        return {"plan": plan, "annotations": current.annotations(audit_id), "result": current.result(audit_id),
                "windows": [current.by_id[unit["id"]] for unit in plan["units"]],
                "predictions": {unit["id"]: current.predicted_by_id[unit["id"]] for unit in plan["units"]}}

    @app.post("/api/spatial/audits/{audit_id}/review")
    def review_spatial_audit(audit_id: str, request: SpatialReviewRequest):
        if not request.simulate and not request.simulate_all:
            raise HTTPException(422, "This pilot reveals existing annotations for a simulated review")
        spatial_study().annotate(audit_id, request.unit_id, simulate=request.simulate, simulate_all=request.simulate_all)
        return spatial_audit(audit_id)

    @app.get("/api/spatial/audits/{audit_id}/reference/{unit_id}")
    def spatial_reference(audit_id: str, unit_id: str):
        spatial_window(unit_id)
        return png_response(mask_image(spatial_study().reference_crop(audit_id, unit_id)))

    @app.get("/api/spatial/audits/{audit_id}/export")
    def spatial_export(audit_id: str):
        return JSONResponse(spatial_study().export(audit_id), headers={
            "Content-Disposition": f'attachment; filename="spatial-study-{audit_id[:8]}.json"'})

    if lab_directory:
        from .lab_api import register_lab
        register_lab(app, lab_directory, public_origin=lab_public_origin,
                     public_path=lab_public_path, secure_cookie=lab_secure_cookie)
    else:
        @app.get("/api/lab/session")
        def unconfigured_lab():
            return {"configured": False, "authenticated": False}

        @app.get("/lab")
        def hidden_lab():
            return RedirectResponse("workspace", status_code=303)
    return app


def from_environment():
    roots = [path for path in os.environ.get('TISSUE_COLLECTION_IMPORT_ROOTS', '').split(os.pathsep) if path]
    region_encoder = None
    if os.environ.get('TISSUE_REGION_SEARCH_ENABLED', '0') == '1':
        from .region_encoder import RegionEncoder
        collections = os.environ.get('TISSUE_COLLECTIONS_DIR')
        if not collections:
            raise ValueError('Region search requires an image collection store')
        region_encoder = RegionEncoder(Path(collections) / 'scan-model.json')
    return create_app(os.environ["TISSUE_STUDY_DIR"], os.environ.get("TISSUE_SPATIAL_DIR"),
                      os.environ.get("TISSUE_RESEARCH_DIR"), os.environ.get("TISSUE_FEEDBACK_DIR"),
                      os.environ.get("TISSUE_SEMANTIC_DIR"), os.environ.get('TISSUE_COLLECTIONS_DIR'),
                      collection_import_roots=roots,
                      collection_scans_enabled=os.environ.get('TISSUE_COLLECTION_SCANS_ENABLED', '1') == '1',
                      lab_directory=os.environ.get('TISSUE_LAB_DIR') if os.environ.get('TISSUE_LAB_ENABLED', '0') == '1' else None,
                      lab_public_origin=os.environ.get('TISSUE_LAB_PUBLIC_ORIGIN'),
                      lab_public_path=os.environ.get('TISSUE_LAB_PUBLIC_PATH', '/'),
                      lab_secure_cookie=os.environ.get('TISSUE_LAB_SECURE_COOKIE', '1') == '1',
                      collection_region_encoder=region_encoder)
