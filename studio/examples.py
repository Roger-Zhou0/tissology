"""Frozen development examples for comparing tissue classes during review."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from .improve import _validate_roles, encoder_identity
from .prepare import CLASSES
from .storage import digest, now, read, write
from .study import verify

SELECTION = "Two development images per class nearest its arithmetic feature centroid by squared Euclidean distance; ties use image ID"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(experiment_directory, study_manifest, study_predictions):
    """Only development references are opened; held-out labels are never read."""
    directory = Path(experiment_directory).resolve()
    source = {name: read(directory / f"{name}.json") for name in
              ("classifier", "experiment", "manifest", "development-labels", "predictions")}
    for artifact in (*source.values(), study_manifest, study_predictions):
        verify(artifact)
    classifier, experiment, manifest, labels, predictions = (
        source[name] for name in ("classifier", "experiment", "manifest", "development-labels", "predictions"))
    if (study_predictions["provenance"].get("classifier_sha256") != classifier["sha256"] or
            study_predictions["provenance"]["manifest_sha256"] != study_manifest["sha256"] or
            classifier["experiment_sha256"] != experiment["sha256"] or
            experiment["manifest_sha256"] != manifest["sha256"] or
            predictions["provenance"]["manifest_sha256"] != manifest["sha256"] or
            classifier["development_labels_sha256"] != labels["sha256"] or
            experiment["development_labels_sha256"] != labels["sha256"] or
            classifier["embedding_file_sha256"] != predictions["provenance"]["embedding_file_sha256"] or
            encoder_identity(study_predictions["provenance"]) != encoder_identity(predictions["provenance"])):
        raise ValueError("Development example provenance differs from the frozen classifier or study")
    _validate_roles(manifest, labels)
    index = {row["id"]: row for row in manifest["images"]}
    development = [index[row["id"]] for row in labels["labels"]]
    for key in ("id", "sha256"):
        if ({row[key] for row in development} & {row[key] for row in study_manifest["images"]}):
            raise ValueError("Development examples overlap the study or pilot population")
    cache_path = directory / "predictions-embeddings.npz"
    if file_hash(cache_path) != classifier["embedding_file_sha256"]:
        raise ValueError("Development example feature cache hash changed")
    feature_key = classifier.get("feature_key", "embeddings")
    if feature_key not in ("embeddings", "classifier_embeddings"):
        raise ValueError("Unsupported classifier feature key")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["ids"].tolist()
        features = cache[feature_key].copy()
    if (ids != [row["id"] for row in manifest["images"]] or
            features.shape != (len(ids), classifier.get("feature_dimension", 512)) or
            not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Development feature IDs, dimensions, or normalization differ")
    positions = {identifier: position for position, identifier in enumerate(ids)}
    chosen = []
    for label in CLASSES:
        members = [row for row in labels["labels"] if row["label"] == label]
        if len(members) < 2:
            raise ValueError("Each tissue class needs two distinct development examples")
        vectors = features[[positions[row["id"]] for row in members]]
        distances = np.sum((vectors - vectors.mean(axis=0)) ** 2, axis=1)
        ranked = sorted(zip(members, distances), key=lambda pair: (float(pair[1]), pair[0]["id"]))
        for rank, (reference, distance) in enumerate(ranked[:2], 1):
            image = index[reference["id"]]
            chosen.append({"id": image["id"], "path": image["path"], "sha256": image["sha256"],
                           "reference_label": label, "role": "development", "development_role": reference["role"],
                           "rank": rank, "centroid_squared_distance": float(distance)})
    if ({str(Path(row["path"]).resolve()) for row in chosen} &
            {str(Path(row["path"]).resolve()) for row in study_manifest["images"]}):
        raise ValueError("Development examples overlap the study or pilot population")
    return {"source_directory": str(directory), "classifier_sha256": classifier["sha256"],
            "source_manifest_sha256": manifest["sha256"], "development_labels_sha256": labels["sha256"],
            "embedding_file_sha256": classifier["embedding_file_sha256"],
            "study_manifest_sha256": study_manifest["sha256"], "feature_key": feature_key,
            "feature_representation": classifier["representation"], "selection": SELECTION, "examples": chosen}


def build(experiment_directory, study_directory):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Build development examples inside a Slurm CPU allocation")
    directory = Path(study_directory).resolve()
    content = inputs(experiment_directory, read(directory / "manifest.json"), read(directory / "predictions.json"))
    path = directory / "reference-examples.json"
    if path.exists():
        artifact = read(path)
        verify(artifact)
        if {key: value for key, value in artifact.items() if key not in ("created_at", "sha256", "schema_version")} != content:
            raise FileExistsError("Development examples are frozen and cannot be replaced")
        return artifact
    for row in content["examples"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Selected development image bytes changed")
    artifact = {"schema_version": 1, "created_at": now(), **content}
    artifact["sha256"] = digest(artifact)
    write(path, artifact)
    return artifact


class ExampleStore:
    def __init__(self, directory, study_manifest, study_predictions):
        path = Path(directory) / "reference-examples.json"
        self.artifact = None
        self.by_id = {}
        if not path.exists():
            return
        artifact = read(path)
        verify(artifact)
        expected = inputs(artifact["source_directory"], study_manifest, study_predictions)
        if {key: value for key, value in artifact.items() if key not in ("created_at", "sha256", "schema_version")} != expected:
            raise ValueError("Frozen development examples differ from their source artifacts")
        self.artifact = artifact
        self.by_id = {row["id"]: row for row in artifact["examples"]}

    def public(self, label):
        if label not in CLASSES:
            raise KeyError(label)
        if self.artifact is None:
            return {"available": False, "examples": []}
        return {"available": True, "source": "Dataset-labeled development examples",
                "classifier_sha256": self.artifact["classifier_sha256"],
                "artifact_sha256": self.artifact["sha256"],
                "examples": [{key: row[key] for key in ("id", "reference_label", "role", "rank")}
                             for row in self.artifact["examples"] if row["reference_label"] == label]}

    def image_path(self, identifier):
        if identifier not in self.by_id:
            raise KeyError(identifier)
        row = self.by_id[identifier]
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Development example image bytes changed")
        return row["path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_directory")
    parser.add_argument("study_directory")
    args = parser.parse_args()
    result = build(args.experiment_directory, args.study_directory)
    print({"examples": len(result["examples"]), "sha256": result["sha256"]})
