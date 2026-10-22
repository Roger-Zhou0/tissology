"""Trusted command-line preparation and Slurm-only training for CRC feedback."""
import argparse
import os
import importlib.metadata
from pathlib import Path
import socket
import warnings

import numpy as np

from .conch_infer import file_hash
from .feedback import BLOCKED, FeedbackStore, training_source_identity
from .improve import _validate_roles, encoder_identity, metrics
from .feedback_storage import freeze
from .prepare import CLASSES
from .storage import now, read
from .study import verify


def source(experiment):
    root = Path(experiment).resolve()
    names = ("experiment", "manifest", "development-labels", "classifier", "predictions")
    config, manifest, labels, classifier, predictions = [read(root / (n + ".json")) for n in names]
    for artifact in (config, manifest, labels, classifier, predictions):
        verify(artifact)
    _validate_roles(manifest, labels)
    if (config["manifest_sha256"] != manifest["sha256"] or
            config["development_labels_sha256"] != labels["sha256"] or
            classifier["experiment_sha256"] != config["sha256"] or
            classifier["development_labels_sha256"] != labels["sha256"] or
            predictions["provenance"]["manifest_sha256"] != manifest["sha256"] or
            classifier["feature_key"] != "classifier_embeddings" or
            classifier["encoder_identity"]["model"] != "MahmoodLab/CONCH" or
            encoder_identity(classifier["encoder_identity"]) != encoder_identity(predictions["provenance"])):
        raise ValueError("CONCH development provenance does not match")
    feature_hash = file_hash(root / "predictions-embeddings.npz")
    if feature_hash != classifier["embedding_file_sha256"] or feature_hash != predictions["provenance"]["embedding_file_sha256"]:
        raise ValueError("Source feature cache changed")
    if file_hash(root / "classifier-coefficients.npz") != classifier["coefficients_sha256"]:
        raise ValueError("Base coefficients changed")
    with np.load(root / "predictions-embeddings.npz", allow_pickle=False) as cache:
        features, ids = cache["classifier_embeddings"].copy(), cache["ids"].tolist()
    if (ids != [r["id"] for r in manifest["images"]] or features.ndim != 2 or
            features.shape != (len(ids), classifier["feature_dimension"]) or not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Feature identities, shape or normalization differ")
    return root, manifest, labels, classifier, features


def prepare(experiment, workspace):
    require_slurm()
    output = Path(workspace).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty feedback workspace")
    root, manifest, labels, classifier, features = source(experiment)
    labels = {r["id"]: r["label"] for r in labels["labels"]}
    with np.load(root / "classifier-coefficients.npz", allow_pickle=False) as model:
        predictions = model["classes"][np.argmax(features @ model["coefficients"].T + model["intercept"], axis=1)]
    rows, indices = [], []
    for i, row in enumerate(manifest["images"]):
        if row["role"] != "train":
            continue
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Development image changed")
        rows.append({"id": row["id"], "role": "train", "path": row["path"], "image_sha256": row["sha256"],
                     "development_label": labels[row["id"]], "predicted_label": str(predictions[i]),
                     "label_source": "NCT-CRC-HE-100K supplied development label", "annotation_extent": "whole_tile"})
        indices.append(i)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "retrieval.npz").open("xb") as handle:
        np.savez_compressed(handle, ids=np.array([r["id"] for r in rows]), features=features[indices])
    freeze(output / "source.json", {"experiment_directory": str(root), "manifest_sha256": manifest["sha256"],
                                   "base_classifier_sha256": classifier["sha256"]})
    return freeze(output / "catalog.json", {
        "created_at": now(), "task": "CRC whole-tile classification", "classes": list(CLASSES),
        "base_classifier_sha256": classifier["sha256"], "manifest_sha256": manifest["sha256"],
        "development_labels_sha256": classifier["development_labels_sha256"],
        "source_sha256": read(output / "source.json")["sha256"],
        "retrieval_sha256": file_hash(output / "retrieval.npz"),
        "representation": classifier["representation"], "examples": rows})


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Feedback preparation and fitting must run inside Slurm")


def train(workspace, candidate_id):
    require_slurm()
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    store = FeedbackStore(workspace)
    plan = store.candidate(candidate_id)["plan"]
    folder = store.directory / "candidates" / candidate_id
    # Exclusive creation prevents duplicate workers from overwriting a version.
    freeze(folder / "started.json", {"created_at": now(), "plan_sha256": plan["sha256"],
                                      "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname()})
    try:
        if plan["source_files"] != training_source_identity():
            raise ValueError("Training source changed after candidate was queued; create a new version")
        private = read(store.directory / "source.json")
        verify(private)
        if private["sha256"] != store.catalog["source_sha256"]:
            raise ValueError("Workspace source changed")
        _, manifest, labels, classifier, features = source(private["experiment_directory"])
        if (manifest["sha256"] != store.catalog["manifest_sha256"] or classifier["sha256"] != plan["base_classifier_sha256"] or
                labels["sha256"] != store.catalog["development_labels_sha256"]):
            raise ValueError("Candidate source changed")
        rows = manifest["images"]
        roles = np.array([r["role"] for r in rows])
        labels = {r["id"]: r["label"] for r in labels["labels"]}
        train_rows = [r for r in rows if r["role"] == "train"]
        train_index = {r["id"]: i for i, r in enumerate(train_rows)}
        original = np.array([labels[r["id"]] for r in train_rows])
        corrected = original.copy()
        weights = np.ones(len(original))
        for correction in plan["corrections"]:
            verify(correction)
            row = store.row(correction["image_id"])
            if (correction["disposition"] != "approved" or correction["role"] != "train" or
                    correction["annotation_extent"] != "whole_tile" or correction["label"] not in CLASSES or
                    correction["catalog_sha256"] != store.catalog["sha256"] or
                    correction["base_classifier_sha256"] != classifier["sha256"] or
                    correction["image_sha256"] != row["image_sha256"]):
                raise ValueError("Invalid approved development correction")
            i = train_index[correction["image_id"]]
            corrected[i], weights[i] = correction["label"], plan["correction_weight"]
        if set(original) != set(CLASSES) or set(corrected) != set(CLASSES):
            raise ValueError("All classes must remain represented")
        models = []
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            for target, sample_weights in ((original, np.ones(len(original))), (corrected, weights)):
                model = LogisticRegression(C=plan["C"], max_iter=plan["max_iter"], solver="lbfgs", random_state=plan["seed"])
                model.fit(features[roles == "train"], target, sample_weight=sample_weights)
                models.append(model)
        candidate = models[1]
        with (folder / "coefficients.npz").open("xb") as handle:
            np.savez_compressed(handle, classes=candidate.classes_, coefficients=candidate.coef_, intercept=candidate.intercept_)
        frozen = freeze(folder / "model.json", {"created_at": now(), "plan_sha256": plan["sha256"],
            "coefficients_sha256": file_hash(folder / "coefficients.npz"), "classes": candidate.classes_.tolist(),
            "encoder_identity": classifier["encoder_identity"], "feature_key": "classifier_embeddings",
            "source_features_sha256": classifier["embedding_file_sha256"], "training_count": len(original),
            "approved_corrections": len(plan["corrections"]), "correction_weight": plan["correction_weight"],
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
            "slurm_job_id": os.environ["SLURM_JOB_ID"]})
        # No heldout-labels.json, study references, final audit, or test store is opened.
        predicted = [m.predict(features[roles == "validation"]) for m in models]
        evaluation_predictions = freeze(folder / "validation-predictions.json", {
            "model_sha256": frozen["sha256"], "ids": [r["id"] for r in rows if r["role"] == "validation"],
            "control": predicted[0].tolist(), "candidate": predicted[1].tolist()})
        truth = [labels[r["id"]] for r in rows if r["role"] == "validation"]
        scores = [metrics(truth, values, "reused development validation") for values in predicted]
        control, adapted = scores
        gate = (adapted["accuracy"] >= control["accuracy"] and adapted["balanced_accuracy"] >= control["balanced_accuracy"] and
                all(a["recall"] >= c["recall"] - 0.05 for a, c in zip(adapted["per_class"], control["per_class"])))
        return freeze(folder / "result.json", {"created_at": now(), "plan_sha256": plan["sha256"],
            "model_sha256": frozen["sha256"], "validation_predictions_sha256": evaluation_predictions["sha256"], "control": control, "candidate": adapted, "development_gate_passed": bool(gate),
            "fresh_evaluation": None, "eligible_for_new_study": False, "promotion_block": BLOCKED,
            "scope": plan["evaluation_scope"], "existing_studies_modified": False})
    except Exception as exc:
        freeze(folder / "failure.json", {"created_at": now(), "plan_sha256": plan["sha256"],
            "stage": "training", "reason": str(exc)[:1000]})
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("experiment")
    p.add_argument("workspace")
    p = sub.add_parser("train")
    p.add_argument("workspace")
    p.add_argument("candidate_id")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.experiment, args.workspace)
    else:
        train(args.workspace, args.candidate_id)


if __name__ == "__main__":
    main()
