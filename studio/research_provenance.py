"""Expose recorded slide/patient/source metadata without changing frozen estimates."""
from pathlib import Path

from .spatial import check_file, seal, verify
from .storage import read


def collection_provenance(workspace):
    source_path = Path(workspace.directory) / "inference" / "final" / "frozen-plan.json"
    if not source_path.exists():
        return None
    source_hash = workspace.source.predictions["source_plan_sha256"]
    check_file(source_path, source_hash)
    source = read(source_path)
    rows = [{"roi_id": row["roi_id"], "slide_id": row["slide"], "patient_id": row.get("patient"),
             "hospital_code": row.get("hospital"), "role": row.get("role", "final")}
            for row in source["samples"]]
    if {row["roi_id"] for row in rows} != set(workspace.source.rois) or len(rows) != len(workspace.source.rois):
        raise ValueError("Source provenance and frame differ")
    if any(row["slide_id"] != workspace.source.rois[row["roi_id"]]["slide_id"] for row in rows):
        raise ValueError("Source slide identifiers differ")
    return seal({"schema": "research-provenance-v1", "frame_sha256": workspace.source.frame["sha256"],
                 "collection_sha256": workspace.collection["sha256"], "source_plan_sha256": source_hash,
                 "source": "Public manual PanopTILs ROI release; TCGA identifiers from released ROI names and fold metadata",
                 "reference_annotation": "Manual pathologist-approved region masks, channel 0; simulated review",
                 "clinical_comparison_groups": None, "physical_scale_verified": False,
                 "patient_count": len({r["patient_id"] for r in rows}) if all(r["patient_id"] for r in rows) else None,
                 "scope": "Identifiers support provenance and separation; they do not make selected fields representative of patients.",
                 "rows": rows})


def verify_provenance_export(artifact):
    metadata = artifact.get("source_provenance")
    if metadata is None:
        return
    verify(metadata)
    if (metadata["frame_sha256"] != artifact["frame"]["sha256"] or metadata["collection_sha256"] != artifact["collection"]["sha256"]
            or metadata["source_plan_sha256"] != artifact["predictions"]["source_plan_sha256"]):
        raise ValueError("Source metadata belongs to another collection")
    rois = {r["id"]: r for r in artifact["frame"]["rois"]}
    if len(metadata["rows"]) != len(rois) or {r["roi_id"] for r in metadata["rows"]} != set(rois):
        raise ValueError("Source metadata ROI identifiers differ")
    if any(r["slide_id"] != rois[r["roi_id"]]["slide_id"] for r in metadata["rows"]):
        raise ValueError("Source metadata slide identifiers differ")
