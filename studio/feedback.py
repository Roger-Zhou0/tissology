"""CRC development feedback. No study, audit, or evaluation labels are accepted.

Append-only events and candidate plans live outside every frozen study. The web
process can retrieve/annotate/queue; only a Slurm worker can fit a classifier.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
import subprocess
import uuid

import numpy as np

from .conch_infer import file_hash
from .feedback_storage import freeze
from .prepare import CLASSES
from .storage import now, read
from .study import verify

BLOCKED = "Fresh evaluation required: no provenance-verified, previously uninspected collection has evaluated this frozen candidate. New studies are blocked."


TRAINING_SOURCES = ("studio/feedback.py", "studio/feedback_train.py", "studio/feedback_storage.py",
                    "studio/improve.py", "studio/quality.py", "studio/prepare.py", "studio/storage.py",
                    "studio/study.py", "studio/conch_infer.py", "scripts/feedback_train.sbatch")


def training_source_identity():
    root = Path(__file__).resolve().parents[1]
    return {name: file_hash(root / name) for name in TRAINING_SOURCES}


def identifier(value):
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Invalid feedback or candidate ID")
    return value


class FeedbackStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.catalog = read(self.directory / "catalog.json")
        verify(self.catalog)
        self.rows = {row["id"]: row for row in self.catalog["examples"]}
        if len(self.rows) != len(self.catalog["examples"]) or any(r["role"] != "train" for r in self.rows.values()):
            raise ValueError("Feedback catalog must contain unique training-role examples only")
        feature_path = self.directory / "retrieval.npz"
        if file_hash(feature_path) != self.catalog["retrieval_sha256"]:
            raise ValueError("Feedback retrieval cache changed")
        with np.load(feature_path, allow_pickle=False) as cache:
            self.features, ids = cache["features"].copy(), cache["ids"].tolist()
        if ids != list(self.rows) or not np.isfinite(self.features).all():
            raise ValueError("Feedback feature identities differ")
        self.indices = {key: i for i, key in enumerate(ids)}

    @contextmanager
    def lock(self):
        with (self.directory / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def row(self, image_id):
        if image_id not in self.rows:
            raise ValueError("Only registered training-role development examples are eligible")
        return self.rows[image_id]

    def public(self, row):
        return {k: v for k, v in row.items() if k != "path"}

    def examples(self, offset=0, limit=24, disagreements=False):
        rows = list(self.rows.values())
        if disagreements:
            rows = [r for r in rows if r["predicted_label"] != r["development_label"]]
        return {"total": len(rows), "examples": [self.public(r) for r in rows[offset:offset + limit]]}

    def neighbors(self, image_id):
        self.row(image_id)
        scores = self.features @ self.features[self.indices[image_id]]
        order = np.argsort(-scores, kind="stable")
        ids = list(self.rows)
        return [{**self.public(self.rows[ids[i]]), "similarity": float(scores[i])}
                for i in order if ids[i] != image_id][:12]

    def events(self):
        values = [read(p) for p in sorted((self.directory / "events").glob("*.json"))]
        for value in values:
            verify(value)
            if value["catalog_sha256"] != self.catalog["sha256"]:
                raise ValueError("Feedback event belongs to another catalog")
        return sorted(values, key=lambda e: (e["created_at"], e["id"]))

    def append(self, body):
        event = {"id": uuid.uuid4().hex, "created_at": now(),
                 "catalog_sha256": self.catalog["sha256"],
                 "task": "crc_tile_classification", "taxonomy": "NCT-CRC nine-class",
                 "base_classifier_sha256": self.catalog["base_classifier_sha256"], **body}
        return freeze(self.directory / "events" / (event["id"] + ".json"), event)

    def flag(self, image_id, reviewer, note):
        row = self.row(image_id)
        with self.lock():
            return self.append({"kind": "flag", "image_id": image_id, "image_sha256": row["image_sha256"],
                                "reviewer": reviewer, "note": note, "role": "train", "label": None,
                                "annotation_extent": "whole_tile"})

    def decide(self, flag_id, reviewer, disposition, label, note, source="manual_review"):
        identifier(flag_id)
        if source not in ("manual_review", "dataset_simulation"):
            raise ValueError("Unknown correction source")
        if disposition not in ("approved", "uncertain", "dismissed"):
            raise ValueError("Unknown review disposition")
        if (disposition == "approved" and label not in CLASSES) or (disposition != "approved" and label is not None):
            raise ValueError("An approved correction requires a tissue class; unresolved flags cannot supply a label")
        with self.lock():
            flag = next((e for e in self.events() if e["id"] == flag_id and e["kind"] == "flag"), None)
            if flag is None:
                raise ValueError("Unknown flag")
            return self.append({"kind": "decision", "flag_id": flag_id, "image_id": flag["image_id"],
                                "image_sha256": flag["image_sha256"], "reviewer": reviewer, "note": note,
                                "role": "train", "annotation_extent": "whole_tile", "disposition": disposition,
                                "label": label, "source": source, "reviewer_expertise": "not independently verified"})

    def corrections(self):
        # The latest decision for each image wins, including uncertainty/retraction.
        latest = {}
        for event in self.events():
            if event["kind"] == "decision":
                latest[event["image_id"]] = event
        return [e for e in latest.values() if e["disposition"] == "approved"]

    def candidate(self, candidate_id):
        folder = self.directory / "candidates" / identifier(candidate_id)
        plan = read(folder / "plan.json")
        verify(plan)
        if plan["id"] != candidate_id or plan["catalog_sha256"] != self.catalog["sha256"]:
            raise ValueError("Candidate belongs to another catalog")
        value = {"id": candidate_id, "plan": plan, "status": "queued", "eligible_for_new_study": False,
                 "promotion_block": BLOCKED}
        for name in ("submission", "started", "result", "failure"):
            if (folder / (name + ".json")).exists():
                artifact = read(folder / (name + ".json"))
                verify(artifact)
                if artifact["plan_sha256"] != plan["sha256"]:
                    raise ValueError("Candidate artifact belongs to another plan")
                value[name] = artifact
                value["status"] = {"submission": "queued", "started": "training", "result": "evaluated_development_only", "failure": "failed"}[name]
        if "result" in value:
            model = read(folder / "model.json")
            verify(model)
            prediction = read(folder / "validation-predictions.json")
            verify(prediction)
            if (model["plan_sha256"] != plan["sha256"] or
                    model["sha256"] != value["result"]["model_sha256"] or
                    file_hash(folder / "coefficients.npz") != model["coefficients_sha256"] or
                    prediction["model_sha256"] != model["sha256"] or
                    prediction["sha256"] != value["result"]["validation_predictions_sha256"]):
                raise ValueError("Candidate model or evaluation binding differs")
        return value

    def candidates(self):
        return [self.candidate(p.parent.name) for p in sorted((self.directory / "candidates").glob("*/plan.json"))]

    def new_candidate(self, submit=True):
        with self.lock():
            corrections = self.corrections()
            if not corrections:
                raise ValueError("Approve at least one development correction before training")
            candidate_id = uuid.uuid4().hex
            folder = self.directory / "candidates" / candidate_id
            plan = freeze(folder / "plan.json", {
                "id": candidate_id, "created_at": now(), "catalog_sha256": self.catalog["sha256"],
                "base_classifier_sha256": self.catalog["base_classifier_sha256"],
                "source_files": training_source_identity(),
                "corrections": corrections, "algorithm": "L2 logistic regression on frozen CONCH pooled features",
                "C": 10.0, "max_iter": 2000, "seed": 20261017, "correction_weight": 5.0,
                "training_role": "train", "evaluation_role": "validation",
                "evaluation_scope": "Reused development validation only; previously inspected and used in original model selection. Not fresh generalization.",
                "control": "Same C=10 classifier fitted on original train-role labels only; not the deployed 5400-label classifier",
                "development_gate": "Candidate accuracy and balanced accuracy must each be at least the train-only control; no per-class recall may decrease by more than 0.05",
                "promotion_rule": BLOCKED})
            if submit:
                try:
                    runner = Path(__file__).resolve().parents[1] / "scripts" / "feedback_train.sbatch"
                    completed = subprocess.run(["sbatch", "--parsable", str(runner), str(self.directory), candidate_id],
                                               capture_output=True, text=True, check=True, timeout=20, cwd=runner.parent.parent)
                    job_id = completed.stdout.strip().split(";")[0]
                    if not job_id.isdigit():
                        raise ValueError("Scheduler returned an invalid job identifier")
                    freeze(folder / "submission.json", {"plan_sha256": plan["sha256"], "job_id": job_id, "created_at": now()})
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    freeze(folder / "failure.json", {"plan_sha256": plan["sha256"], "created_at": now(),
                                                      "stage": "submission", "reason": type(exc).__name__ + ": scheduler submission failed"})
            return self.candidate(candidate_id)
