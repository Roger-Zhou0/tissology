"""Persistent studies, probability audits and gated image evidence."""
from __future__ import annotations

from pathlib import Path
import threading
import uuid

import numpy as np
from PIL import Image

from .research_data import ENDPOINT_DEFINITIONS
from .spatial import SpatialStudy, _validate_components, check_file, components_from_mask, seal, verify
from .storage import now, read, write

METHODS = [
    {"id": "uniform", "name": "Uniform model-assisted", "description": "Random windows; use the frozen predictions and correct their errors.", "min_budget": 2, "default": True},
    {"id": "manual", "name": "Manual random", "description": "Random windows; estimate areas from their references alone.", "min_budget": 2},
    {"id": "stratified", "name": "Stratified model-assisted", "description": "Pilot-informed allocation minimizes the sum of anticipated normalized variances.", "min_budget": 8},
    {"id": "shared", "name": "Shared allocation", "description": "Pilot-informed allocation targets the largest anticipated normalized variance across the two measurements.", "min_budget": 8},
]


def valid_id(identifier):
    if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
        raise ValueError("Invalid study or audit identifier")
    return identifier


def reference_components(source, units):
    """Decode only the requested source windows; never use this in selection."""
    masks, output = {}, []
    for unit in units:
        identifier = unit["roi_id"]
        if identifier not in masks:
            reference = source.reference_by_roi[identifier]
            check_file(reference["path"], reference["sha256"])
            with Image.open(reference["path"]) as image:
                array = np.array(image)
            if array.shape != (1024, 1024, 3):
                raise ValueError("Unexpected manual reference geometry")
            masks[identifier] = array[..., reference["channel"]]
        b = unit["bounds"]
        output.append(components_from_mask(masks[identifier][b["top"]:b["bottom"], b["left"]:b["right"]]))
    return np.asarray(output, dtype=int)


def build_collection(root):
    """Pilot references may inform allocation; final truth is never opened here."""
    from .spatial_audit import freeze_design
    root = Path(root)
    final, pilot = SpatialStudy(root / "collection" / "final"), SpatialStudy(root / "collection" / "pilot")
    final_slides = {r["slide_id"] for r in final.rois.values()}
    pilot_slides = {r["slide_id"] for r in pilot.rois.values()}
    if final_slides & pilot_slides:
        raise ValueError("Pilot and final slides overlap")
    refs = reference_components(pilot, pilot.rows)
    provenance = {"final_frame_sha256": final.frame["sha256"], "final_prediction_sha256": final.predictions["sha256"],
                  "pilot_frame_sha256": pilot.frame["sha256"], "pilot_prediction_sha256": pilot.predictions["sha256"],
                  "pilot_reference_sources_sha256": pilot.reference_sources["sha256"],
                  "inference_plan_sha256": read(root / "inference" / "inference-plan.json")["sha256"]}
    design = freeze_design(final.predicted[:, [0, 2, 3]], final.areas,
                           pilot.predicted[:, [0, 2, 3]], refs[:, [0, 2, 3]], pilot.areas,
                           budgets=(48, 96, 160), repetitions=5000, seed=20261029, provenance=provenance)
    write(root / "design.json", design)
    write(root / "pilot-reference.json", seal({"role": "allocation_pilot", "frame_sha256": pilot.frame["sha256"],
                                                "units": [{"id": unit["id"], "components": value.tolist()}
                                                          for unit, value in zip(pilot.rows, refs, strict=True)]}))
    collection = seal({"schema": "research-collection-v1", "id": "panoptils-heldout-001", "created_at": now(),
                       "name": "Breast tissue: 32 held-out fields", "status": "ready",
                       "scope": "Finite collection of selected manual breast-tissue fields; no patient-level inference.",
                       "roi_count": len(final.rois), "slide_count": len(final_slides), "patient_count": None,
                       "pilot_roi_count": len(pilot.rois), "pilot_slide_count": len(pilot_slides), "pilot_window_count": len(pilot.rows),
                       "population_size": len(final.rows), "final_pixels": int(final.areas.sum()),
                       "model": final.config["model"], "heldout_status": "Released fold-2 test slides and hospitals; pilot/final/previously inspected slides disjoint.",
                       "window": "64 × 64 native-pixel central windows; clipped edges retained; 64-pixel surrounding context is not measured.",
                       "endpoint_definitions": ENDPOINT_DEFINITIONS, "audit_methods": METHODS,
                       "evaluation_budgets": [48, 96, 160], "annotation_mode": "Simulated review of pathologist-approved manual region masks",
                       "frame_sha256": final.frame["sha256"], "prediction_sha256": final.predictions["sha256"],
                       "design_sha256": design["sha256"], "physical_scale_verified": False})
    write(root / "collection.json", collection)
    return collection


class ResearchWorkspace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.collection = read(self.directory / "collection.json")
        self.design = read(self.directory / "design.json")
        verify(self.collection)
        verify(self.design)
        self.source = SpatialStudy(self.directory / "collection" / "final")
        if (self.collection["frame_sha256"] != self.source.frame["sha256"]
                or self.collection["prediction_sha256"] != self.source.predictions["sha256"]
                or self.collection["design_sha256"] != self.design["sha256"]):
            raise ValueError("Research source bindings differ")
        self.lock = threading.RLock()

    def folder(self, study_id):
        return self.directory / "studies" / valid_id(study_id)

    def study(self, study_id):
        study = read(self.folder(study_id) / "study.json")
        verify(study)
        if study["collection_sha256"] != self.collection["sha256"] or study["id"] != study_id:
            raise ValueError("Study is bound to a different collection")
        return study

    def create_study(self, collection_id, endpoint_ids, name=None):
        available = {row["id"] for row in ENDPOINT_DEFINITIONS}
        if collection_id != self.collection["id"] or not endpoint_ids or len(set(endpoint_ids)) != len(endpoint_ids) or not set(endpoint_ids) <= available:
            raise ValueError("Choose this collection and one or both predefined measurements")
        identifier = uuid.uuid4().hex
        if name is not None and (not isinstance(name, str) or not name.strip() or len(name.strip()) > 120):
            raise ValueError("Use a study name between 1 and 120 characters")
        study = seal({"schema": "research-study-v1", "id": identifier, "name": name.strip() if name is not None else self.collection["name"] + " · " + identifier[:6],
                      "created_at": now(), "collection_id": collection_id, "collection_sha256": self.collection["sha256"],
                      "endpoint_ids": endpoint_ids, "measurement_definitions": [d for d in ENDPOINT_DEFINITIONS if d["id"] in endpoint_ids],
                      "prediction_sha256": self.source.predictions["sha256"], "design_sha256": self.design["sha256"],
                      "interval_family": "Both predefined endpoints remain in the simultaneous family even when only one is displayed.",
                      "predictor_frozen": True, "allocation_pilot_cost": self.collection["pilot_window_count"]})
        write(self.folder(study["id"]) / "study.json", study)
        return study

    def studies(self):
        return sorted([self.study(path.parent.name) for path in (self.directory / "studies").glob("*/study.json")],
                      key=lambda value: value["created_at"], reverse=True)

    def initial(self, study):
        totals = self.source.predicted.sum(axis=0)
        return [{"id": definition["id"], "name": definition["name"], "estimate": float(totals[column] / totals[3]) if totals[3] else None,
                 "numerator": int(totals[column]), "denominator": int(totals[3])}
                for definition, column in zip(ENDPOINT_DEFINITIONS, [0, 2], strict=True) if definition["id"] in study["endpoint_ids"]]

    def new_audit(self, study_id, budget, method, seed):
        from .spatial_audit import sample_plan
        study = self.study(study_id)
        engine = sample_plan(self.design, method, budget, seed)
        identifier = uuid.uuid4().hex
        units = []
        for row in engine["selected"]:
            unit = self.source.rows[row["index"]]
            units.append({**unit, "frame_index": row["index"], "stratum": row["stratum"],
                          "inclusion_probability": row["inclusion_probability"],
                          "reason": ("Random draw from the full window frame." if method in ("manual", "uniform") else
                                     "Random draw within a prediction-defined stratum; allocation was frozen using separate pilot errors.")})
        plan = seal({"schema": "research-audit-plan-v1", "id": identifier, "created_at": now(),
                     "study_sha256": study["sha256"], "frame_sha256": self.source.frame["sha256"],
                     "prediction_sha256": self.source.predictions["sha256"], "engine_plan": engine,
                     "method": method, "budget": budget, "seed": seed, "alpha": engine.get("alpha", .05),
                     "units": units, "allocation": engine.get("allocation", engine.get("strata", [])),
                     "pilot_cost": self.collection["pilot_window_count"], "total_review_units": budget + self.collection["pilot_window_count"],
                     "design_description": "Fixed-size probability sample without replacement. Every reviewed window supplies both area numerators and their shared eligible-area denominator. No stopping on results or model retraining."})
        folder = self.folder(study_id) / "audits" / identifier
        with self.lock:
            write(folder / "plan.json", plan)
            write(folder / "annotations.json", seal({"schema": "research-annotations-v1", "plan_sha256": plan["sha256"], "rows": []}))
        return plan

    def plan(self, study_id, audit_id):
        from .spatial_audit import sample_plan
        study = self.study(study_id)
        plan = read(self.folder(study_id) / "audits" / valid_id(audit_id) / "plan.json")
        verify(plan)
        verify(plan["engine_plan"])
        if plan["id"] != audit_id or plan["study_sha256"] != study["sha256"]:
            raise ValueError("Audit belongs to another frozen study")
        if (plan["frame_sha256"] != self.source.frame["sha256"] or plan["prediction_sha256"] != self.source.predictions["sha256"]
                or plan["engine_plan"] != sample_plan(self.design, plan["method"], plan["budget"], plan["seed"])):
            raise ValueError("Audit differs from its frozen design")
        if [row["frame_index"] for row in plan["units"]] != plan["engine_plan"]["indices"]:
            raise ValueError("Audit units differ from frozen probability sample")
        for unit, selected in zip(plan["units"], plan["engine_plan"]["selected"], strict=True):
            if (unit["id"] != self.source.rows[selected["index"]]["id"] or unit["inclusion_probability"] != selected["inclusion_probability"]
                    or unit["stratum"] != selected["stratum"]):
                raise ValueError("Audit unit or inclusion probability differs")
        return plan

    def annotation_artifact(self, study_id, audit_id):
        plan = self.plan(study_id, audit_id)
        annotations = read(self.folder(study_id) / "audits" / audit_id / "annotations.json")
        verify(annotations)
        allowed = {u["id"] for u in plan["units"]}
        ids = [row["id"] for row in annotations["rows"]]
        if annotations["plan_sha256"] != plan["sha256"] or len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError("Annotations differ from selected audit windows")
        if ids:
            _validate_components([r["components"] for r in annotations["rows"]], [self.source.by_id[i]["area_pixels"] for i in ids], "reviewed")
        return annotations

    def review(self, study_id, audit_id, unit_id=None, *, simulate_all=False):
        with self.lock:
            plan = self.plan(study_id, audit_id)
            artifact = self.annotation_artifact(study_id, audit_id)
            previous = {r["id"] for r in artifact["rows"]}
            allowed = {r["id"]: r for r in plan["units"]}
            if not simulate_all and unit_id not in allowed:
                raise ValueError("Only selected windows may be reviewed")
            targets = [r for r in plan["units"] if r["id"] not in previous] if simulate_all else ([] if unit_id in previous else [allowed[unit_id]])
            references = reference_components(self.source, targets) if targets else []
            rows = list(artifact["rows"])
            for unit, values in zip(targets, references, strict=True):
                rows.append({"id": unit["id"], "components": values.tolist(), "role": "audit", "created_at": now(),
                             "source": "simulated_manual_reference", "plan_sha256": plan["sha256"],
                             "reference_mask_sha256": self.source.reference_by_roi[unit["roi_id"]]["sha256"]})
            if targets:
                artifact = seal({"schema": "research-annotations-v1", "plan_sha256": plan["sha256"], "rows": rows})
                write(self.folder(study_id) / "audits" / audit_id / "annotations.json", artifact, overwrite=True)
            if len(rows) == plan["budget"]:
                path = self.folder(study_id) / "audits" / audit_id / "result.json"
                if not path.exists():
                    write(path, self.calculate(study_id, plan, artifact))

    def calculate(self, study_id, plan, annotations):
        from .spatial_audit import estimate
        by_id = {row["id"]: row["components"] for row in annotations["rows"]}
        references = np.asarray([by_id[row["id"]] for row in plan["units"]])
        calculated = estimate(self.source.predicted[:, [0, 2, 3]], self.source.areas,
                              plan["engine_plan"], references[:, [0, 2, 3]])
        evidence = []
        for unit, values in zip(plan["units"], references, strict=True):
            predicted = self.source.predicted[unit["frame_index"]]
            baseline = np.zeros(3) if plan["method"] == "manual" else predicted[[0, 2, 3]]
            correction = (values[[0, 2, 3]] - baseline) / unit["inclusion_probability"]
            evidence.append({"unit_id": unit["id"], "roi_id": unit["roi_id"], "predicted_components": predicted.astype(int).tolist(),
                             "reference_components": values.tolist(), "correction": correction.tolist(),
                             "ranking_score": float(np.max(np.abs(correction))), "units": "native pixels",
                             "description": ("Ranked by largest inverse-probability-weighted reference component contribution (epithelial, dense tissue, eligible)." if plan["method"] == "manual" else
                                             "Ranked by largest absolute inverse-probability-weighted component correction (epithelial, dense tissue, eligible). This is not an isolated change in a ratio.")})
        evidence.sort(key=lambda r: (-r["ranking_score"], r["unit_id"]))
        return seal({"schema": "research-audit-result-v1", "created_at": now(), "study_sha256": self.study(study_id)["sha256"],
                     "plan_sha256": plan["sha256"], "annotation_sha256": annotations["sha256"],
                     "statistics": calculated, "evidence": evidence, "annotation_mode": "Simulated review; not measured pathologist time"})

    def result(self, study_id, audit_id):
        plan = self.plan(study_id, audit_id)
        path = self.folder(study_id) / "audits" / audit_id / "result.json"
        if not path.exists():
            return None
        result = read(path)
        verify(result)
        if result["plan_sha256"] != plan["sha256"] or result["annotation_sha256"] != self.annotation_artifact(study_id, audit_id)["sha256"]:
            raise ValueError("Stored result bindings differ")
        return result

    def audits(self, study_id):
        rows = []
        for path in (self.folder(study_id) / "audits").glob("*/plan.json"):
            plan = self.plan(study_id, path.parent.name)
            rows.append({"id": plan["id"], "method": plan["method"], "budget": plan["budget"], "created_at": plan["created_at"],
                         "reviewed": len(self.annotation_artifact(study_id, plan["id"])["rows"])})
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    def reference_crop(self, study_id, audit_id, unit_id):
        annotations = self.annotation_artifact(study_id, audit_id)
        if unit_id not in {row["id"] for row in annotations["rows"]}:
            raise ValueError("Reveal this selected window before viewing its reference")
        unit = self.source.by_id[unit_id]
        reference = self.source.reference_by_roi[unit["roi_id"]]
        check_file(reference["path"], reference["sha256"])
        b = unit["bounds"]
        with Image.open(reference["path"]) as image:
            return np.array(image)[b["top"]:b["bottom"], b["left"]:b["right"], reference["channel"]]

    def export(self, study_id, audit_id):
        result = self.result(study_id, audit_id)
        if result is None:
            raise ValueError("Complete the fixed audit before export")
        return seal({"schema": "research-export-v1", "study": self.study(study_id), "collection": self.collection,
                     "frame": self.source.frame, "predictions": self.source.predictions, "design": self.design,
                     "plan": self.plan(study_id, audit_id), "annotations": self.annotation_artifact(study_id, audit_id), "result": result,
                     "reference_policy": "Only sampled references are included. Full final truth is confined to the separate evaluation workspace."})


def recompute_export(artifact):
    from .spatial_audit import estimate, sample_plan
    verify(artifact)
    for name in ("study", "collection", "frame", "predictions", "design", "plan", "annotations", "result"):
        verify(artifact[name])
    frame, plan, predictions = artifact["frame"], artifact["plan"], artifact["predictions"]
    if (predictions["frame_sha256"] != frame["sha256"] or plan["frame_sha256"] != frame["sha256"]
            or plan["prediction_sha256"] != predictions["sha256"] or plan["study_sha256"] != artifact["study"]["sha256"]
            or artifact["annotations"]["plan_sha256"] != plan["sha256"]
            or artifact["result"]["annotation_sha256"] != artifact["annotations"]["sha256"]
            or artifact["result"]["plan_sha256"] != plan["sha256"]
            or artifact["study"]["collection_sha256"] != artifact["collection"]["sha256"]
            or artifact["collection"]["design_sha256"] != artifact["design"]["sha256"]
            or plan["engine_plan"] != sample_plan(artifact["design"], plan["method"], plan["budget"], plan["seed"])):
        raise ValueError("Export provenance mismatch")
    by_id = {r["id"]: r["components"] for r in predictions["units"]}
    ref = {r["id"]: r["components"] for r in artifact["annotations"]["rows"]}
    if (len(by_id) != len(predictions["units"]) or len(by_id) != len(frame["units"])
            or set(by_id) != {r["id"] for r in frame["units"]}):
        raise ValueError("Export predictions must cover each frame unit exactly once")
    if (len(ref) != len(artifact["annotations"]["rows"]) or len(ref) != plan["budget"]
            or len(plan["units"]) != plan["budget"] or set(ref) != {r["id"] for r in plan["units"]}):
        raise ValueError("Export needs exactly the complete sampled references")
    if [u["frame_index"] for u in plan["units"]] != plan["engine_plan"]["indices"]:
        raise ValueError("Export sample indices differ")
    for unit, selected in zip(plan["units"], plan["engine_plan"]["selected"], strict=True):
        original = frame["units"][selected["index"]]
        if (any(unit.get(key) != value for key, value in original.items())
                or unit["inclusion_probability"] != selected["inclusion_probability"] or unit["stratum"] != selected["stratum"]):
            raise ValueError("Export sample identity or geometry differs")
    calculated = estimate(np.asarray([by_id[r["id"]] for r in frame["units"]])[:, [0, 2, 3]],
                          [r["area_pixels"] for r in frame["units"]], plan["engine_plan"],
                          np.asarray([ref[r["id"]] for r in plan["units"]])[:, [0, 2, 3]])
    if calculated != artifact["result"]["statistics"]:
        raise ValueError("Export statistics do not reproduce")
    return calculated
