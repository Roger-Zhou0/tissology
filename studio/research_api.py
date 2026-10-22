"""HTTP adapter for the finite-field research workflow."""
import io
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .research import ResearchWorkspace
from .spatial import check_file, seal
from .research_provenance import collection_provenance
from .storage import read


class CreateStudy(BaseModel):
    collection_id: str
    endpoint_ids: list[str]
    name: str | None = Field(default=None, min_length=1, max_length=120)


class CreateAudit(BaseModel):
    budget: int = Field(default=96, ge=2)
    method: str = "uniform"
    seed: int = Field(default=20261030, ge=0, le=2**32 - 1)
    alpha: float = .05


class Review(BaseModel):
    unit_id: str | None = None
    simulate: bool = False
    simulate_all: bool = False


def color_mask(mask):
    palette = np.array([[225, 229, 234], [190, 91, 134], [71, 139, 124], [114, 93, 172],
                        [190, 91, 134], [225, 229, 234], [225, 229, 234], [225, 229, 234],
                        [225, 229, 234]], dtype=np.uint8)
    if mask.ndim != 2 or np.any(mask < 0) or np.any(mask > 8):
        raise ValueError("Invalid region codes")
    return Image.fromarray(palette[mask.astype(int)])


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


def register_research(app, directory):
    workspace = ResearchWorkspace(directory)
    provenance = collection_provenance(workspace)
    router = APIRouter()
    base = "api/research"
    collection_base = f"{base}/collections/{workspace.collection['id']}"

    def public_collection():
        collection = dict(workspace.collection)
        # The exact minimum is computed from the frozen prediction strata.
        methods = [dict(m) for m in collection["audit_methods"]]
        for method in methods:
            if method["id"] in ("shared", "stratified"):
                method["min_budget"] = workspace.design.get("minimum_budget", 20)
        collection["audit_methods"] = methods
        if provenance is not None:
            collection["patient_count"] = provenance["patient_count"]
            collection["provenance_sha256"] = provenance["sha256"]
        return collection

    def unit_urls(unit, study_id=None, audit_id=None):
        result = dict(unit)
        result["context_url"] = f"{collection_base}/windows/{unit['id']}/context"
        result["prediction_url"] = f"{collection_base}/windows/{unit['id']}/prediction"
        if study_id and audit_id:
            result["reference_url"] = f"{base}/studies/{study_id}/audits/{audit_id}/reference/{unit['id']}"
        return result

    def study_response(study_id):
        study = workspace.study(study_id)
        rois = [{"id": roi["id"], "slide_id": roi["slide_id"], "fov": roi["manual_fov"],
                 "image_url": f"{collection_base}/rois/{roi['id']}/image",
                 "prediction_url": f"{collection_base}/rois/{roi['id']}/prediction"}
                for roi in workspace.source.rois.values()]
        if provenance is not None:
            metadata = {row["roi_id"]: row for row in provenance["rows"]}
            rois = [{**roi, "patient_id": metadata[roi["id"]]["patient_id"], "hospital_code": metadata[roi["id"]]["hospital_code"]} for roi in rois]
        return {**study, "collection": public_collection(), "endpoint_definitions": study["measurement_definitions"],
                "initial": workspace.initial(study), "rois": rois, "audits": workspace.audits(study_id),
                "population_size": len(workspace.source.rows), "audit_methods": public_collection()["audit_methods"],
                "source_provenance": provenance}

    def audit_response(study_id, audit_id):
        plan = workspace.plan(study_id, audit_id)
        units = [unit_urls(unit, study_id, audit_id) for unit in plan["units"]]
        annotations = workspace.annotation_artifact(study_id, audit_id)
        result = workspace.result(study_id, audit_id)
        if result:
            study = workspace.study(study_id)
            result = {**result, **result["statistics"]}
            result["endpoints"] = [dict(row) for row in result["endpoints"] if row["id"] in study["endpoint_ids"]]
            for endpoint in result["endpoints"]:
                if endpoint.get("standard_error") == 0 and endpoint.get("status") != "exact_census":
                    endpoint["warnings"] = list(endpoint.get("warnings", [])) + ["No observed variation in this endpoint; the approximate interval may be unreliable."]
            result["warnings"] = list(result.get("warnings", []))
            comparison_path = workspace.directory / "comparison-public.json"
            if comparison_path.exists():
                comparison = read(comparison_path)
                condition = next((row for row in comparison["rows"] if row["method"] == plan["method"] and row["budget"] == plan["budget"]), None)
                if condition and condition["simultaneous_coverage"] < .95:
                    result["warnings"].append(f"In {comparison['repetitions']:,} simulated audits with this method and budget, the approximate ranges covered both reference fractions {100*condition['simultaneous_coverage']:.2f}% of the time, below the nominal 95%. Inspect the conservative bounds before making a conclusion.")
            if plan["method"] == "manual":
                result["evidence_title"] = "Reviewed contributions to the manual estimate"
                result["evidence_description"] = "Ranked by the largest inverse-probability-weighted reference component contribution. These are contributions to the manual estimate, not model-error corrections."
            result["evidence"] = [{**e, **unit_urls(workspace.source.by_id[e["unit_id"]], study_id, audit_id)} for e in result["evidence"]]
        return {"plan": {**plan, "units": units},
                "annotations": {row["id"]: row for row in annotations["rows"]},
                "result": result, "predictions": {u["id"]: workspace.source.predicted_by_id[u["id"]] for u in units}}

    @router.get("/research")
    def page():
        return FileResponse(Path(__file__).parent / "static" / "research.html")

    @router.get("/api/research/collections")
    def collections():
        return {"collections": [public_collection()], "studies": [
            {**study, "latest_audit": next(iter(workspace.audits(study["id"])), None)}
            for study in workspace.studies()]}

    @router.post("/api/research/studies")
    def create_study(request: CreateStudy):
        return study_response(workspace.create_study(request.collection_id, request.endpoint_ids, name=request.name)["id"])

    @router.get("/api/research/studies/{study_id}")
    def get_study(study_id: str):
        return study_response(study_id)

    @router.post("/api/research/studies/{study_id}/audits")
    def create_audit(study_id: str, request: CreateAudit):
        if request.alpha != .05:
            raise ValueError("This frozen demonstration uses a 95% simultaneous confidence family")
        plan = workspace.new_audit(study_id, request.budget, request.method, request.seed)
        return audit_response(study_id, plan["id"])

    @router.get("/api/research/studies/{study_id}/audits/{audit_id}")
    def get_audit(study_id: str, audit_id: str):
        return audit_response(study_id, audit_id)

    @router.post("/api/research/studies/{study_id}/audits/{audit_id}/review")
    def review(study_id: str, audit_id: str, request: Review):
        if not request.simulate and not request.simulate_all:
            raise ValueError("This demonstration reveals existing manual annotations as simulated review")
        workspace.review(study_id, audit_id, request.unit_id, simulate_all=request.simulate_all)
        return audit_response(study_id, audit_id)

    @router.get("/api/research/studies/{study_id}/audits/{audit_id}/reference/{unit_id}")
    def reference(study_id: str, audit_id: str, unit_id: str):
        return png(color_mask(workspace.reference_crop(study_id, audit_id, unit_id)))

    @router.get("/api/research/studies/{study_id}/audits/{audit_id}/export")
    def export(study_id: str, audit_id: str):
        artifact = workspace.export(study_id, audit_id)
        if provenance is not None:
            artifact = seal({**{k: v for k, v in artifact.items() if k != "sha256"}, "source_provenance": provenance})
        return JSONResponse(artifact, headers={
            "Content-Disposition": f'attachment; filename="tissue-research-{study_id[:8]}-{audit_id[:8]}.json"'})

    def check_collection(identifier):
        if identifier != workspace.collection["id"]:
            raise HTTPException(404, "Unknown collection")

    @router.get("/api/research/collections/{collection_id}/rois/{roi_id}/{kind}")
    def roi_image(collection_id: str, roi_id: str, kind: str):
        check_collection(collection_id)
        if roi_id not in workspace.source.rois or kind not in ("image", "prediction"):
            raise HTTPException(404, "Unknown image")
        roi = workspace.source.rois[roi_id]
        key = "image" if kind == "image" else "prediction_mask"
        check_file(roi[f"{key}_path"], roi[f"{key}_sha256"])
        with Image.open(roi[f"{key}_path"]) as source:
            image = source.convert("RGB") if kind == "image" else color_mask(np.array(source))
        b = roi["manual_fov"]
        ImageDraw.Draw(image).rectangle((b["left"], b["top"], b["right"] - 1, b["bottom"] - 1), outline=(225, 35, 35), width=3)
        return png(image)

    @router.get("/api/research/collections/{collection_id}/windows/{unit_id}/{kind}")
    def window_image(collection_id: str, unit_id: str, kind: str):
        check_collection(collection_id)
        if unit_id not in workspace.source.by_id or kind not in ("context", "prediction"):
            raise HTTPException(404, "Unknown image window")
        unit = workspace.source.by_id[unit_id]
        roi = workspace.source.rois[unit["roi_id"]]
        key = "image" if kind == "context" else "prediction_mask"
        check_file(roi[f"{key}_path"], roi[f"{key}_sha256"])
        b = unit["context_bounds"] if kind == "context" else unit["bounds"]
        with Image.open(roi[f"{key}_path"]) as source:
            image = source.crop(tuple(b[k] for k in ("left", "top", "right", "bottom")))
            image = image.convert("RGB") if kind == "context" else color_mask(np.array(image))
        if kind == "context":
            center = unit["bounds"]
            ImageDraw.Draw(image).rectangle((center["left"] - b["left"], center["top"] - b["top"],
                                            center["right"] - b["left"] - 1, center["bottom"] - b["top"] - 1), outline=(225, 35, 35), width=2)
        return png(image)

    @router.get("/api/research/comparison")
    def comparison():
        path = workspace.directory / "comparison-public.json"
        return read(path) if path.exists() else {"available": False, "status": "The frozen repeated-audit experiment is being computed."}

    @router.get("/api/research/studies/{study_id}/comparison")
    def study_comparison(study_id: str):
        workspace.study(study_id)
        return comparison()

    app.include_router(router)
    return workspace
