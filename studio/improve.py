"""Fit a small classifier on frozen encoder features, then open fresh evaluation labels.

The development labels, final audit labels, and model comparison have distinct
artifacts. Selection freezes all roles before encoding. No original study image
or duplicate bytes may enter the new experiment.
"""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import re
import socket

import numpy as np
from PIL import Image

from .prepare import CLASSES, ENDPOINTS, PROMPTS
from .quality import classification_metrics
from .storage import digest, now, read, write
from .study import verify

GRID = [0.1, 1.0, 10.0, 100.0]
ENCODER_KEYS = ("model", "model_revision", "representation", "preprocessing")
ENCODER_EXTRA_KEYS = ("code_revision", "checkpoint_sha256", "classifier_representation",
                      "classifier_feature_key", "feature_transform")
SCOPE = ("Exact agreement with supplied class labels on this finite, newly frozen tile collection. "
         "Patient IDs are unavailable; this does not establish patient-level generalization or "
         "independence from foundation-model pretraining. Dataset labels are not a new expert review.")


def freeze(path, artifact):
    artifact["sha256"] = digest(artifact)
    write(path, artifact)
    return artifact


def encoder_identity(provenance):
    """Feature compatibility includes checkpoint, implementation and preprocessing."""
    if any(not provenance.get(key) for key in ENCODER_KEYS):
        raise ValueError("Encoder model, revision, representation and preprocessing are required")
    identity = {key: provenance[key] for key in (*ENCODER_KEYS, *ENCODER_EXTRA_KEYS)
                if key in provenance}
    preprocessing = dict(identity["preprocessing"])
    if isinstance(preprocessing.get("transform"), str):
        # Function repr addresses identify a Python process, not preprocessing.
        # Preserve function names and every other transform/configuration byte.
        preprocessing["transform"] = re.sub(
            r"(<function [^<>\n]+?) at 0x[0-9a-fA-F]+>", r"\1>", preprocessing["transform"])
    identity["preprocessing"] = preprocessing
    return identity


def feature_representation(provenance, feature_key):
    if feature_key not in ("embeddings", "classifier_embeddings"):
        raise ValueError("Unsupported classifier feature key")
    if provenance.get("classifier_feature_key", feature_key) != feature_key:
        raise ValueError("Declared classifier feature key differs from selected features")
    key = "classifier_representation" if feature_key == "classifier_embeddings" else "representation"
    representation = provenance.get(key)
    if not representation or "l2" not in representation.lower():
        raise ValueError("Classifier feature representation must explicitly identify L2 normalization")
    return representation


def prepare(source, output, exclude_studies, *, train_per_class=100, validation_per_class=25,
            n=600, pilot=90, seed=20261020):
    """Select an unbalanced study/pilot first, then class-balanced development sets."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if min(train_per_class, validation_per_class, n) < 1 or pilot < 0:
        raise ValueError("Positive train, validation and study counts are required")
    if not exclude_studies:
        raise ValueError("Explicitly exclude the previously inspected study")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new experiment directory; frozen artifacts cannot be overwritten")
    old_ids, old_hashes, old_paths, exclusions = set(), set(), set(), []
    for study in exclude_studies:
        artifact = read(Path(study) / "manifest.json")
        verify(artifact)
        for row in artifact["images"]:
            old_ids.add(row["id"])
            old_hashes.add(row["sha256"])
            old_paths.add(str(Path(row["path"]).resolve()))
        exclusions.append({"manifest_sha256": artifact["sha256"], "images": len(artifact["images"])})
    paths = []
    for label in CLASSES:
        with os.scandir(source / label) as entries:
            paths.extend(Path(entry.path) for entry in entries
                         if entry.name.endswith(".tif") and entry.is_file())
    paths.sort()
    random.Random(seed).shuffle(paths)
    rows, development, heldout, rejected = [], [], [], []
    used_hashes, used_ids = set(old_hashes), set(old_ids)
    per_class = {role: {label: 0 for label in CLASSES} for role in ("train", "validation")}
    total = n + pilot + len(CLASSES) * (train_per_class + validation_per_class)
    for path in paths:
        if len(rows) == total:
            break
        if str(path.resolve()) in old_paths:
            continue
        label = path.parent.name
        if len(heldout) < n + pilot:
            role = "study" if len(heldout) < n else "pilot"
        elif per_class["train"][label] < train_per_class:
            role = "train"
        elif per_class["validation"][label] < validation_per_class:
            role = "validation"
        else:
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        identifier = sha[:20]
        if sha in used_hashes or identifier in used_ids:
            rejected.append({"image_sha256": sha, "reason": "duplicate or previously inspected image"})
            continue
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError("Expected native 224x224 RGB tile")
        except Exception as error:
            rejected.append({"path": str(path), "reason": str(error)})
            continue
        used_hashes.add(sha)
        used_ids.add(identifier)
        rows.append({"id": identifier, "path": str(path), "sha256": sha,
                     "size_px": [224, 224], "mode": "RGB", "role": role})
        reference = {"id": identifier, "label": label, "role": role,
                     "source": "NCT-CRC-HE-100K supplied class folder"}
        if role in per_class:
            development.append(reference)
            per_class[role][label] += 1
        else:
            heldout.append(reference)
    if len(rows) != total:
        raise ValueError("Insufficient readable unique images for the requested frozen roles")
    manifest = freeze(output / "manifest.json", {
        "schema_version": 1, "created_at": now(), "source_root": str(source),
        "source_file_count": len(paths), "seed": seed, "images": rows,
        "n_study": n, "n_pilot": pilot,
        "selection": "Seeded uniform study/pilot permutation before class-balanced development sampling",
        "excluded_prior_studies": exclusions, "exclusions": rejected})
    dev = freeze(output / "development-labels.json", {
        "purpose": "Training and validation only; never final audit labels", "labels": development})
    hold = freeze(output / "heldout-labels.json", {
        "purpose": "Opened only after classifier and predictions freeze; study evaluation and allocation pilot",
        "labels": heldout})
    write(output / "prompts.json", PROMPTS)
    freeze(output / "experiment.json", {
        "schema_version": 1, "created_at": now(), "manifest_sha256": manifest["sha256"],
        "development_labels_sha256": dev["sha256"], "heldout_labels_sha256": hold["sha256"],
        "training_labels": len(CLASSES) * train_per_class,
        "validation_labels": len(CLASSES) * validation_per_class,
        "selection_metric": "Validation balanced accuracy; smallest C breaks ties",
        "regularization_grid": GRID, "solver": "lbfgs", "max_iter": 2000,
        "feature_transform": "L2-normalized frozen encoder features; feature key and representation frozen in classifier metadata",
        "refit": "Refit once using training plus validation labels, before opening held-out labels",
        "seed": seed, "scope": SCOPE,
        "model_training_overlap": "Unknown foundation-model pretraining overlap; no patient identifiers"})
    return {"experiment_directory": str(output), "images": total,
            "study": n, "pilot": pilot, "train": len(CLASSES) * train_per_class,
            "validation": len(CLASSES) * validation_per_class,
            "manifest_sha256": manifest["sha256"]}


def metrics(truth, predictions, name):
    """Finite-collection label agreement; no binomial CI for a fully labeled census."""
    result = classification_metrics(list(truth), list(predictions), name)
    result["balanced_accuracy"] = float(np.mean([row["recall"] for row in result["per_class"]
                                                if row["recall"] is not None]))
    return result


def _validate_roles(manifest, development):
    rows = manifest["images"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate experiment image IDs or bytes")
    if any(row["role"] not in ("train", "validation", "study", "pilot") for row in rows):
        raise ValueError("Unknown experiment role")
    expected = {row["id"]: row["role"] for row in rows if row["role"] in ("train", "validation")}
    labels = development["labels"]
    if (len(labels) != len(expected) or {row["id"]: row["role"] for row in labels} != expected or
            any(row["label"] not in CLASSES for row in labels)):
        raise ValueError("Development labels must match train/validation roles exactly")


def train(experiment_dir, study_dir):
    """Tune with validation only, freeze classifier, and compare on a fresh study."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Classifier fitting must run inside a Slurm CPU or GPU allocation")
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    import warnings

    experiment_dir, study_dir = Path(experiment_dir), Path(study_dir)
    if (experiment_dir / "classifier.json").exists():
        raise FileExistsError("Classifier is frozen; never retune against an opened test collection")
    if study_dir.exists() and any(study_dir.iterdir()):
        raise ValueError("Promote to a new study directory")
    config = read(experiment_dir / "experiment.json")
    manifest = read(experiment_dir / "manifest.json")
    development = read(experiment_dir / "development-labels.json")
    baseline = read(experiment_dir / "predictions.json")
    prompts = read(experiment_dir / "prompts.json")
    for artifact in (config, manifest, development, baseline):
        verify(artifact)
    if (config["manifest_sha256"] != manifest["sha256"] or
            config["development_labels_sha256"] != development["sha256"] or
            baseline["provenance"]["manifest_sha256"] != manifest["sha256"] or
            baseline["provenance"]["prompts_sha256"] != digest(prompts)):
        raise ValueError("Experiment manifests or development label hashes differ")
    _validate_roles(manifest, development)
    cache_path = experiment_dir / "predictions-embeddings.npz"
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if cache_hash != baseline["provenance"]["embedding_file_sha256"]:
        raise ValueError("Embedding cache changed after inference")
    with np.load(cache_path, allow_pickle=False) as cache:
        retrieval_features, ids = cache["embeddings"].copy(), cache["ids"].tolist()
        feature_key = "classifier_embeddings" if "classifier_embeddings" in cache.files else "embeddings"
        features = cache[feature_key].copy()
        text_embeddings, text_labels = cache["text_embeddings"].copy(), cache["labels"].copy()
    identity = encoder_identity(baseline["provenance"])
    representation = feature_representation(baseline["provenance"], feature_key)
    encoder_name = baseline["provenance"]["model"].rsplit("/", 1)[-1].upper()
    classifier_name = f"{encoder_name} + linear classifier"
    rows = manifest["images"]
    if (ids != [row["id"] for row in rows] or
            [row["id"] for row in baseline["images"]] != ids or
            features.ndim != 2 or features.shape[0] != len(rows) or features.shape[1] < 1 or
            retrieval_features.ndim != 2 or retrieval_features.shape[0] != len(rows) or
            not np.isfinite(features).all() or not np.isfinite(retrieval_features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)):
        raise ValueError("Embedding order or content differs from the frozen manifest")
    roles = np.array([row["role"] for row in rows])
    labels = {row["id"]: row["label"] for row in development["labels"]}
    training, validation = roles == "train", roles == "validation"
    train_y = np.array([labels[identifier] for identifier, selected in zip(ids, training) if selected])
    validation_y = np.array([labels[identifier] for identifier, selected in zip(ids, validation) if selected])
    if set(train_y) != set(CLASSES) or set(validation_y) != set(CLASSES):
        raise ValueError("All nine classes are required in training and validation")
    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for c in config["regularization_grid"]:
            model = LogisticRegression(C=c, solver=config["solver"], max_iter=config["max_iter"],
                                       random_state=config["seed"])
            model.fit(features[training], train_y)
            score = metrics(validation_y, model.predict(features[validation]), "validation")
            candidates.append({"C": c, "balanced_accuracy": score["balanced_accuracy"],
                               "accuracy": score["accuracy"]})
        selected = max(candidates, key=lambda row: (row["balanced_accuracy"], -row["C"]))
        fit = training | validation
        fit_y = np.array([labels[identifier] for identifier, use in zip(ids, fit) if use])
        model = LogisticRegression(C=selected["C"], solver=config["solver"], max_iter=config["max_iter"],
                                   random_state=config["seed"])
        model.fit(features[fit], fit_y)
    # Persist coefficients and all predictions before any held-out reference is opened.
    coefficient_file = experiment_dir / "classifier-coefficients.npz"
    with coefficient_file.open("xb") as handle:
        np.savez_compressed(handle, classes=model.classes_, coefficients=model.coef_, intercept=model.intercept_)
    classifier = freeze(experiment_dir / "classifier.json", {
        "schema_version": 1, "name": classifier_name, "created_at": now(),
        "selected_C": selected["C"], "validation_candidates": candidates,
        "selection_metric": config["selection_metric"], "refit_labels": len(fit_y),
        "training_labels": len(train_y), "validation_labels": len(validation_y),
        "experiment_sha256": config["sha256"], "development_labels_sha256": development["sha256"],
        "embedding_file_sha256": cache_hash,
        "coefficients_sha256": hashlib.sha256(coefficient_file.read_bytes()).hexdigest(),
        "classes": model.classes_.tolist(), "representation": representation,
        "feature_key": feature_key, "feature_dimension": features.shape[1],
        "encoder_identity": identity,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
        "scope": SCOPE})
    probabilities = model.predict_proba(features)
    predicted_rows = []
    for row, original, scores in zip(rows, baseline["images"], probabilities):
        order = np.argsort(scores)[::-1]
        predicted_rows.append({"id": row["id"], "role": row["role"],
                               "predicted_label": str(model.classes_[order[0]]),
                               "cosine_scores": original["cosine_scores"],
                               "classification_scores": dict(zip(model.classes_.tolist(), scores.tolist())),
                               "score_kind": "linear_classifier_probability",
                               "margin": float(scores[order[0]] - scores[order[1]])})
    linear_predictions = freeze(experiment_dir / "linear-predictions.json", {
        "classifier_sha256": classifier["sha256"], "manifest_sha256": manifest["sha256"],
        "images": predicted_rows, "created_at": now()})
    heldout = read(experiment_dir / "heldout-labels.json")
    verify(heldout)
    if heldout["sha256"] != config["heldout_labels_sha256"]:
        raise ValueError("Held-out references changed after experiment freeze")
    keep = [i for i, row in enumerate(rows) if row["role"] in ("study", "pilot")]
    kept_rows = [rows[i] for i in keep]
    expected = {row["id"]: row["role"] for row in kept_rows}
    if (len(heldout["labels"]) != len(expected) or
            {row["id"]: row["role"] for row in heldout["labels"]} != expected or
            any(row["label"] not in CLASSES for row in heldout["labels"])):
        raise ValueError("Held-out references must match study/pilot roles exactly")
    references = {"purpose": "Simulated final audit and isolated frozen-model evaluation", "labels": heldout["labels"]}
    new_manifest = freeze(study_dir / "manifest.json", {
        "schema_version": 1, "created_at": now(), "source_root": manifest["source_root"],
        "source_file_count": manifest["source_file_count"], "seed": manifest["seed"],
        "selection": manifest["selection"], "n_study": manifest["n_study"], "n_pilot": manifest["n_pilot"],
        "parent_experiment_manifest_sha256": manifest["sha256"],
        "excluded_prior_studies": manifest["excluded_prior_studies"],
        "exclusions": manifest["exclusions"], "images": kept_rows})
    write(study_dir / "references.json", references)
    write(study_dir / "prompts.json", prompts)
    new_cache = study_dir / "predictions-embeddings.npz"
    with new_cache.open("xb") as handle:
        arrays = {"ids": np.array(ids)[keep], "embeddings": retrieval_features[keep],
                  "text_embeddings": text_embeddings, "labels": text_labels}
        if feature_key != "embeddings":
            arrays[feature_key] = features[keep]
        np.savez_compressed(handle, **arrays)
    provenance = {**baseline["provenance"], "manifest_sha256": new_manifest["sha256"],
                  "embedding_file_sha256": hashlib.sha256(new_cache.read_bytes()).hexdigest(),
                  "classifier": classifier, "classifier_sha256": classifier["sha256"],
                  "baseline_predictions_sha256": baseline["sha256"],
                  "experiment_predictions_sha256": linear_predictions["sha256"],
                  "classification": f"Multinomial logistic regression on frozen {encoder_name} image features",
                  "classifier_feature_key": feature_key,
                  "classifier_representation": representation,
                  "score_interpretation": "Linear classifier probabilities; calibration has not been validated. Original text cosine scores retained separately.",
                  "prediction_created_at": now()}
    predictions = freeze(study_dir / "predictions.json", {
        "provenance": provenance, "images": [predicted_rows[i] for i in keep]})
    write(study_dir / "study.json", {
        "schema_version": 1, "title": "CRC tissue composition — improved classifier", "created_at": now(),
        "manifest_sha256": new_manifest["sha256"], "classes": CLASSES, "endpoints": ENDPOINTS,
        "target_population": f"The {manifest['n_study']} unique, readable tiles in this frozen manifest",
        "measurement": {"numerator": "Number of tiles with the endpoint reference class",
                        "denominator": "All retained study tiles, including BACK and DEB",
                        "aggregation": "Equal weight per tile; ratio of counts with known denominator",
                        "eligible_tissue": "Entire native tile; no pixel-level mask",
                        "exclusions": "Previously inspected, unreadable, non-224x224 RGB, or duplicate-byte images before freeze",
                        "physical_scale": "Dataset documents 0.5 microns/pixel; no physical area or distance estimated"},
        "scope": "Finite tile collection; no patient, whole-slide, spatial or clinical-outcome inference",
        "reference_limit": "Agreement with supplied dataset class labels; simulated review is not expert validation",
        "split": f"Classifier uses {len(train_y)} training and {len(validation_y)} validation labels; all study/pilot images excluded from fitting; final audit labels never train the predictor",
        "model_training_overlap": "Unknown overlap with foundation-model pretraining; no independent patient-generalization claim",
        "default_audit_budget": min(120, manifest["n_study"]), "familywise_alpha": 0.05,
        "pilot_labels": manifest["n_pilot"], "endpoint_precision_scales": {key: 1.0 for key in ENDPOINTS},
        "development_labels": len(fit_y), "classifier_sha256": classifier["sha256"]})
    labels_by_id = {row["id"]: row["label"] for row in heldout["labels"]}
    study_indices = [i for i, row in enumerate(rows) if row["role"] == "study"]
    truth = [labels_by_id[rows[i]["id"]] for i in study_indices]
    before_labels = [baseline["images"][i]["predicted_label"] for i in study_indices]
    after_labels = [predicted_rows[i]["predicted_label"] for i in study_indices]
    before = metrics(truth, before_labels, f"{encoder_name} with text prompts")
    after = metrics(truth, after_labels, classifier_name)
    quality = freeze(study_dir / "model-quality.json", {
        "schema_version": 1, "created_at": now(), "manifest_sha256": new_manifest["sha256"],
        "prediction_sha256": predictions["sha256"], "reference_sha256": digest(references),
        "active_model": after, "comparison": {
            "baseline": before, "improved": after, "same_test_population": True,
            "development_labels": len(fit_y), "training_labels": len(train_y),
            "validation_labels": len(validation_y), "scope": SCOPE,
            "newly_correct": sum(b != y and a == y for y, b, a in zip(truth, before_labels, after_labels)),
            "newly_incorrect": sum(b == y and a != y for y, b, a in zip(truth, before_labels, after_labels)),
            "accuracy_difference": after["accuracy"] - before["accuracy"]},
        "scope": SCOPE})
    write(experiment_dir / "comparison.json", quality)
    return {"study_directory": str(study_dir), "baseline_accuracy": before["accuracy"],
            "improved_accuracy": after["accuracy"], "selected_C": selected["C"],
            "model_quality_sha256": quality["sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", default="/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--exclude-study", action="append", required=True)
    prepare_parser.add_argument("--train-per-class", type=int, default=100)
    prepare_parser.add_argument("--validation-per-class", type=int, default=25)
    prepare_parser.add_argument("--n", type=int, default=600)
    prepare_parser.add_argument("--pilot", type=int, default=90)
    prepare_parser.add_argument("--seed", type=int, default=20261020)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("experiment_dir")
    train_parser.add_argument("study_dir")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.output, args.exclude_study, train_per_class=args.train_per_class,
                         validation_per_class=args.validation_per_class, n=args.n, pilot=args.pilot, seed=args.seed)
    else:
        result = train(args.experiment_dir, args.study_dir)
    print(result, flush=True)
