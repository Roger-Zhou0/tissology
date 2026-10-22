"""Descriptive model agreement on a frozen, fully reference-labelled collection.

This separate evaluation artifact never supplies labels to a predictor or audit
allocator. Its exact finite-collection accuracy is not a patient-generalization
estimate or the match rate of an unequally sampled audit.
"""
import argparse
import os
from pathlib import Path

from .prepare import CLASSES
from .storage import digest, now, read, write
from .study import Study, verify


def classification_metrics(reference, predicted, name):
    if len(reference) != len(predicted) or len(reference) == 0:
        raise ValueError("Reference and predicted labels must have equal nonzero length")
    index = {label: i for i, label in enumerate(CLASSES)}
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for actual, prediction in zip(reference, predicted):
        if actual not in index or prediction not in index:
            raise ValueError("Unknown tissue class")
        matrix[index[actual]][index[prediction]] += 1
    per_class = []
    for i, label in enumerate(CLASSES):
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        correct = matrix[i][i]
        per_class.append({"label": label, "support": support, "correct": correct,
                          "recall": correct / support if support else None,
                          "precision": correct / predicted_count if predicted_count else None})
    correct = sum(matrix[i][i] for i in range(len(CLASSES)))
    return {"name": name, "accuracy": correct / len(reference), "correct": correct,
            "total": len(reference), "labels": CLASSES, "confusion_matrix": matrix,
            "per_class": per_class}


def write_quality(directory):
    """Add an evaluation artifact without modifying any frozen study inputs."""
    study = Study(directory)
    provenance = study.predictions["provenance"]
    encoder = provenance.get("encoder_name") or provenance.get("model", "Image encoder").rsplit("/", 1)[-1]
    if encoder.lower() in {"plip", "conch"}:
        encoder = encoder.upper()
    name = provenance.get("classifier", {}).get("name") or f"{encoder} with text prompts"
    metrics = classification_metrics([study.references[r["id"]] for r in study.rows],
                                     [r["predicted_label"] for r in study.rows], name)
    output = {"schema_version": 1, "created_at": now(),
              "manifest_sha256": study.manifest["sha256"],
              "prediction_sha256": study.predictions["sha256"],
              "reference_sha256": study.reference_hash,
              "active_model": metrics, "comparison": None,
              "scope": "Exact agreement with dataset class labels across this fixed image collection. "
                       "This collection has already been inspected; it is an exploratory baseline. "
                       "Patient identities and foundation-model pretraining overlap are unknown. "
                       "No patient-level, clinical, or independent external-validation claim."}
    output["sha256"] = digest(output)
    write(Path(directory) / "model-quality.json", output)
    return output


def load_quality(study):
    path = study.directory / "model-quality.json"
    if not path.exists():
        return {"available": False}
    artifact = read(path)
    verify(artifact)
    for field, expected in (("manifest_sha256", study.manifest["sha256"]),
                            ("prediction_sha256", study.predictions["sha256"]),
                            ("reference_sha256", study.reference_hash)):
        if artifact[field] != expected:
            raise ValueError("Model quality belongs to a different frozen study")
    return {**artifact, "available": True}


def load_test_evaluations(study):
    """Read aggregate tests bound to the exact frozen classifier serving this study."""
    path = study.directory / "test-evaluations.json"
    if not path.exists():
        return {"available": False, "evaluations": []}
    artifact = read(path)
    verify(artifact)
    classifier = study.predictions["provenance"].get("classifier_sha256")
    if not classifier or artifact.get("classifier_sha256") != classifier:
        raise ValueError("Test evaluations belong to a different frozen classifier")
    evaluations = list(artifact["evaluations"])
    seen = {row["id"] for row in evaluations}
    seen_hashes = {row["evaluation_sha256"] for row in evaluations if "evaluation_sha256" in row}
    addenda = []
    for path in sorted(study.directory.glob("test-evaluations-addendum-*.json")):
        addition = read(path)
        verify(addition)
        if (addition.get("classifier_sha256") != classifier or
                addition.get("base_aggregate_sha256") != artifact["sha256"]):
            raise ValueError("Test addendum belongs to a different classifier or base aggregate")
        for row in addition["evaluations"]:
            if (row.get("classifier_sha256") != classifier or row["id"] in seen or
                    row.get("evaluation_sha256") in seen_hashes):
                raise ValueError("Test addendum contains duplicate or mismatched evaluations")
            seen.add(row["id"])
            if "evaluation_sha256" in row:
                seen_hashes.add(row["evaluation_sha256"])
            evaluations.append(row)
        addenda.append({"path": path.name, "sha256": addition["sha256"], "created_at": addition["created_at"]})
    if not addenda:
        return {**artifact, "available": True}
    combined = {**{key: value for key, value in artifact.items() if key != "sha256"},
                "base_aggregate_sha256": artifact["sha256"], "addenda": addenda, "evaluations": evaluations}
    combined["sha256"] = digest(combined)
    return {**combined, "available": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run model evaluation on a Slurm compute node")
    result = write_quality(args.study_dir)
    print({k: result["active_model"][k] for k in ("name", "correct", "total", "accuracy")})
