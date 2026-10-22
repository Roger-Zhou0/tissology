"""Validated study artifacts and auditable annotation state."""
from pathlib import Path
import threading

import numpy as np

from .audit import estimate, make_plan
from .prepare import CLASSES
from .storage import digest, now, read, write


def verify(artifact):
    if artifact.get("sha256") != digest({k: v for k, v in artifact.items() if k != "sha256"}):
        raise ValueError("Frozen artifact hash mismatch")


class Study:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.lock = threading.RLock()
        self.config = read(self.directory / "study.json")
        self.manifest = read(self.directory / "manifest.json")
        self.predictions = read(self.directory / "predictions.json")
        verify(self.manifest)
        verify(self.predictions)
        if (self.predictions["provenance"]["manifest_sha256"] != self.manifest["sha256"] or
                self.config["manifest_sha256"] != self.manifest["sha256"]):
            raise ValueError("Study and prediction manifest hashes differ")
        if self.predictions["provenance"]["prompts_sha256"] != digest(read(self.directory / "prompts.json")):
            raise ValueError("Prompts changed after predictions were frozen")
        self.images = {r["id"]: r for r in self.manifest["images"]}
        self.by_id = {r["id"]: r for r in self.predictions["images"]}
        if len(self.by_id) != len(self.predictions["images"]) or set(self.by_id) != set(self.images):
            raise ValueError("Prediction IDs do not match the manifest exactly")
        self.rows = [r for r in self.predictions["images"] if r["role"] == "study"]
        self.pilot = [r for r in self.predictions["images"] if r["role"] == "pilot"]
        self.reference_artifact = read(self.directory / "references.json")
        refs = self.reference_artifact["labels"]
        if len(refs) != len(self.images) or {r["id"] for r in refs} != set(self.images):
            raise ValueError("Reference IDs do not match the manifest")
        self.references = {r["id"]: r["label"] for r in refs}
        for row in refs:
            if row["label"] not in CLASSES or row["role"] != self.images[row["id"]]["role"]:
                raise ValueError("Invalid reference role or class")
        for row in self.by_id.values():
            if row["role"] != self.images[row["id"]]["role"] or row["predicted_label"] not in CLASSES:
                raise ValueError("Prediction role or class mismatch")
        self.pilot_labels = {r["id"]: self.references[r["id"]] for r in self.pilot}
        self.reference_hash = digest(self.reference_artifact)

    def new_plan(self, budget, method, seed):
        with self.lock:
            plan = make_plan(self.rows, budget, method, seed,
                             prediction_hash=self.predictions["sha256"],
                             manifest_hash=self.manifest["sha256"],
                             pilot_rows=self.pilot, pilot_labels=self.pilot_labels)
            plan["study_sha256"] = digest(self.config)
            plan["sha256"] = digest({k: v for k, v in plan.items() if k != "sha256"})
            folder = self.directory / "audits" / plan["id"]
            write(folder / "plan.json", plan)
            write(folder / "annotations.json", [])
            return plan

    def get_plan(self, plan_id):
        if len(plan_id) != 32 or any(c not in "0123456789abcdef" for c in plan_id):
            raise ValueError("Invalid audit ID")
        plan = read(self.directory / "audits" / plan_id / "plan.json")
        verify(plan)
        if (plan["prediction_sha256"] != self.predictions["sha256"] or
                plan["manifest_sha256"] != self.manifest["sha256"] or
                plan.get("study_sha256") != digest(self.config)):
            raise ValueError("Audit belongs to a different frozen predictor or population")
        return plan

    def annotations(self, plan_id):
        self.get_plan(plan_id)
        return read(self.directory / "audits" / plan_id / "annotations.json")

    def annotate(self, plan_id, image_id=None, label=None, *, simulate=False, simulate_all=False):
        with self.lock:
            plan = self.get_plan(plan_id)
            folder = self.directory / "audits" / plan_id
            annotations = read(folder / "annotations.json")
            previous = {a["id"]: a for a in annotations}
            allowed = {unit["id"] for unit in plan["units"]}
            if simulate_all:
                targets = [unit["id"] for unit in plan["units"] if unit["id"] not in previous]
                simulate = True
            else:
                if image_id not in allowed:
                    raise ValueError("Only sampled audit units can be annotated")
                targets = [image_id]
            for identifier in targets:
                value = self.references[identifier] if simulate else label
                if value not in CLASSES:
                    raise ValueError("Choose one of the nine reference classes")
                if identifier in previous:
                    if previous[identifier]["label"] != value:
                        raise ValueError("Completed annotations are immutable; create a new audit")
                    continue
                annotations.append({"id": identifier, "label": value, "role": "audit",
                                    "source": "simulated_dataset_reference" if simulate else "human_entered",
                                    "created_at": now(), "plan_id": plan_id,
                                    "reference_sha256": self.reference_hash if simulate else None})
            write(folder / "annotations.json", annotations, overwrite=True)
            if len(annotations) == plan["budget"]:
                result = estimate(self.rows, plan, {a["id"]: a["label"] for a in annotations})
                result.update(created_at=now(), annotation_sha256=digest(annotations),
                              annotation_sources=sorted({a["source"] for a in annotations}),
                              study_sha256=digest(self.config), reference_sha256=self.reference_hash)
                if not (folder / "result.json").exists():
                    write(folder / "result.json", result)
            return annotations

    def result(self, plan_id):
        self.get_plan(plan_id)
        path = self.directory / "audits" / plan_id / "result.json"
        if not path.exists():
            return None
        result = read(path)
        if result["annotation_sha256"] != digest(self.annotations(plan_id)):
            raise ValueError("Annotations changed after result creation")
        return result

    def audit_summaries(self):
        root = self.directory / "audits"
        summaries = []
        for path in sorted(root.glob("*/plan.json")):
            plan = self.get_plan(path.parent.name)
            annotations = self.annotations(plan["id"])
            summaries.append({"id": plan["id"], "method": plan["method"], "budget": plan["budget"],
                              "created_at": plan["created_at"], "reviewed": len(annotations)})
        return sorted(summaries, key=lambda r: r["created_at"], reverse=True)

    def export(self, plan_id):
        plan = self.get_plan(plan_id)
        result = self.result(plan_id)
        if result is None:
            raise ValueError("Complete the planned audit before exporting results")
        return {"study": self.config, "provenance": self.predictions["provenance"],
                "plan": plan, "annotations": self.annotations(plan_id), "result": result,
                "population_manifest": [{k: row[k] for k in ("id", "sha256", "role", "size_px") if k in row}
                                        for row in self.manifest["images"] if row["role"] == "study"],
                "population_predictions": self.rows,
                "allocation_pilot": {"predictions": self.pilot, "labels": self.pilot_labels,
                                     "role": "pilot"},
                "dataset": {"name": "NCT-CRC-HE-100K", "url": "https://zenodo.org/records/1214456"},
                "evidence": [{"id": u["id"], "image_sha256": self.images[u["id"]]["sha256"],
                              "relative_image_url": f"/api/images/{u['id']}",
                              "prediction": self.by_id[u["id"]]} for u in plan["units"]]}


def build_atlas(directory):
    """Compute a two-dimensional PCA of frozen embeddings on a compute node."""
    directory = Path(directory)
    with np.load(directory / "predictions-embeddings.npz", allow_pickle=False) as cache:
        features, ids = cache["embeddings"], cache["ids"].tolist()
    centered = features - features.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = left[:, :2] * singular[:2]
    explained = singular[:2] ** 2 / (singular ** 2).sum()
    write(directory / "atlas.json", {
        "method": "PCA of normalized image embeddings; morphology similarity, not tissue coordinates",
        "explained_variance_fraction": explained.tolist(),
        "points": {identifier: xy.tolist() for identifier, xy in zip(ids, coordinates)}})
