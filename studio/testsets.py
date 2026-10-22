"""Evaluate the frozen tissue classifier on new, disjoint image collections.

Preparation freezes image selection and references separately. Inference uses
the existing pinned encoder. Evaluation applies saved coefficients, freezes
predictions, and only then opens test references. No fitting occurs here.
"""
import argparse
import hashlib
import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image

from .infer import MODEL, REVISION
from .paths import cached_paths, resolved
from .improve import encoder_identity, feature_representation
from .prepare import CLASSES, PROMPTS
from .quality import classification_metrics
from .storage import digest, now, read, write
from .study import verify

FEATURES = "L2-normalized frozen PLIP projected features; no fitted transform"
ENCODER_FEATURES = "L2-normalized projected CLIP image/text features"
FEATURE_KEYS = ("model", "model_revision", "representation", "preprocessing")


def freeze(path, artifact):
    result = {**artifact, "sha256": digest(artifact)}
    write(path, result)
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classifier_scores(features, classifier, coefficients, intercept):
    """Keep signed SVM decisions distinct from multinomial probabilities."""
    family = classifier.get("family", "multinomial_logistic")
    if family == "rbf_svc":
        from .rbf_predict import score
        votes, _ = score(features, coefficients)
        return votes, "class_vote_counts", "rbf_svm_votes"
    logits = features @ coefficients.T + intercept
    if family == "linear_svm":
        return logits, "decision_scores", "linear_svm_decision"
    if family != "multinomial_logistic":
        raise ValueError("Unsupported frozen classifier family")
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, "classification_scores", "linear_classifier_probability"


def _classifier(directory):
    """Verify the frozen classifier and its original encoder lineage."""
    directory = resolved(directory)
    classifier = read(directory / "classifier.json")
    experiment = read(directory / "experiment.json")
    training_manifest = read(directory / "manifest.json")
    encoder_predictions = read(directory / "predictions.json")
    for artifact in (classifier, experiment, training_manifest, encoder_predictions):
        verify(artifact)
    encoder = encoder_predictions["provenance"]
    if (classifier["experiment_sha256"] != experiment["sha256"] or
            classifier["development_labels_sha256"] != experiment["development_labels_sha256"] or
            experiment["manifest_sha256"] != training_manifest["sha256"] or
            encoder["manifest_sha256"] != training_manifest["sha256"] or
            classifier["embedding_file_sha256"] != encoder["embedding_file_sha256"]):
        raise ValueError("Classifier training lineage differs from its frozen artifacts")
    identity = encoder_identity(encoder)
    feature_key = classifier.get("feature_key", "embeddings")
    if "feature_key" in classifier:
        if (encoder_identity(classifier.get("encoder_identity", {})) != identity or
                classifier["representation"] != feature_representation(encoder, feature_key)):
            raise ValueError("Classifier feature representation differs from its training encoder")
    elif (classifier["representation"] != FEATURES or encoder.get("model") != MODEL or
          encoder.get("model_revision") != REVISION or
          encoder.get("representation") != ENCODER_FEATURES):
        # Original PLIP artifacts predate explicit feature routing. Keep their
        # exact provenance contract, without rewriting historical metadata.
        raise ValueError("Unsupported legacy classifier feature representation or encoder revision")
    coefficient_path = directory / "classifier-coefficients.npz"
    if file_hash(coefficient_path) != classifier["coefficients_sha256"]:
        raise ValueError("Frozen classifier coefficients changed")
    with np.load(coefficient_path, allow_pickle=False) as saved:
        classes = saved["classes"].tolist()
        if classifier.get("family") == "rbf_svc":
            coefficients = {key: saved[key].copy() for key in saved.files}
            intercept = None
        else:
            coefficients = saved["coefficients"].copy()
            intercept = saved["intercept"].copy()
    dimension = classifier.get("feature_dimension", 512)
    if classifier.get("family") == "rbf_svc":
        from .rbf_predict import validate
        if classes != CLASSES or classes != classifier["classes"] or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Invalid RBF classifier classes or feature dimension")
        validate(coefficients, classes, dimension)
        return classifier, identity, coefficients, intercept
    if (not isinstance(dimension, int) or dimension < 1 or
            classifier.get("family", "multinomial_logistic") not in ("multinomial_logistic", "linear_svm") or
            classes != classifier["classes"] or classes != CLASSES or
            coefficients.shape != (len(CLASSES), dimension) or intercept.shape != (len(CLASSES),) or
            not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()):
        raise ValueError("Invalid classifier classes or coefficient dimensions/content")
    return classifier, identity, coefficients, intercept


@cached_paths
def reuse_embeddings(source_directory, destination_directory):
    """Reuse one frozen encoding across predictors with identical image inputs."""
    source, destination = resolved(source_directory), resolved(destination_directory)
    if (destination / "predictions.json").exists() or (destination / "predictions-embeddings.npz").exists():
        raise FileExistsError("Destination inference is already frozen")
    source_manifest, destination_manifest = read(source / "manifest.json"), read(destination / "manifest.json")
    destination_config, inference = read(destination / "testset.json"), read(source / "predictions.json")
    for artifact in (source_manifest, destination_manifest, destination_config, inference):
        verify(artifact)
    if (source_manifest["images"] != destination_manifest["images"] or
            inference["provenance"]["manifest_sha256"] != source_manifest["sha256"] or
            destination_config["manifest_sha256"] != destination_manifest["sha256"] or
            encoder_identity(destination_config["encoder"]) != encoder_identity(inference["provenance"])):
        raise ValueError("Embedding reuse requires identical ordered images and encoder/preprocessing")
    source_prompts, destination_prompts = read(source / "prompts.json"), read(destination / "prompts.json")
    if source_prompts != destination_prompts or inference["provenance"]["prompts_sha256"] != digest(source_prompts):
        raise ValueError("Embedding reuse prompt artifacts differ")
    cache_path = source / "predictions-embeddings.npz"
    cache_hash = file_hash(cache_path)
    if cache_hash != inference["provenance"]["embedding_file_sha256"]:
        raise ValueError("Source embedding cache changed")
    with np.load(cache_path, allow_pickle=False) as cache:
        if (cache["ids"].tolist() != [row["id"] for row in destination_manifest["images"]] or
                [row["id"] for row in inference["images"]] != cache["ids"].tolist()):
            raise ValueError("Source embedding order differs from the frozen population")
    with cache_path.open("rb") as original, (destination / "predictions-embeddings.npz").open("xb") as copied:
        shutil.copyfileobj(original, copied)
    if file_hash(destination / "predictions-embeddings.npz") != cache_hash:
        raise ValueError("Copied embedding bytes differ")
    return freeze(destination / "predictions.json", {
        "provenance": {**inference["provenance"], "manifest_sha256": destination_manifest["sha256"],
                       "embedding_reuse": {"source_directory": str(source), "source_manifest_sha256": source_manifest["sha256"],
                                           "source_inference_sha256": inference["sha256"],
                                           "source_embedding_file_sha256": cache_hash, "reused_at": now()}},
        "images": inference["images"]})


def _excluded(directories):
    ids, hashes, paths, records = set(), set(), set(), []
    seen = set()
    for directory in directories:
        path = (resolved(directory) / "manifest.json")
        if path in seen:
            continue
        seen.add(path)
        manifest = read(path)
        verify(manifest)
        for row in manifest["images"]:
            ids.add(row["id"])
            hashes.add(row["sha256"])
            paths.add(str(resolved(row["path"])))
        records.append({"manifest_path": str(path), "manifest_sha256": manifest["sha256"],
                        "images": len(manifest["images"])})
    return ids, hashes, paths, records


def _previous_benchmark(directory, *, open_references=False):
    """Bind a named completed benchmark; never silently waive overlap checks."""
    directory = resolved(directory)
    config = read(directory / "testset.json")
    manifest = read(directory / "manifest.json")
    evaluation = read(directory / "test-evaluation.json")
    for artifact in (config, manifest, evaluation):
        verify(artifact)
    if (config["manifest_sha256"] != manifest["sha256"] or
            evaluation["manifest_sha256"] != manifest["sha256"] or
            evaluation["testset_sha256"] != config["sha256"] or
            evaluation["reference_sha256"] != config["reference_sha256"] or
            evaluation.get("model_refit") is not False or
            any(row["role"] != "test" for row in manifest["images"])):
        raise ValueError("Reused benchmark must be a completed, consistently frozen test collection")
    references = None
    if open_references:
        references = read(directory / "references.json")
        verify(references)
        if (references["sha256"] != config["reference_sha256"] or
                [row["id"] for row in references["labels"]] != [row["id"] for row in manifest["images"]] or
                any(row["role"] != "test" or row["label"] not in CLASSES for row in references["labels"])):
            raise ValueError("Reused benchmark references differ from its original test population")
    binding = {"test_directory": str(directory), "manifest_path": str(directory / "manifest.json"),
               "manifest_sha256": manifest["sha256"], "reference_sha256": config["reference_sha256"],
               "reference_file_sha256": file_hash(directory / "references.json"),
               "testset_sha256": config["sha256"], "evaluation_sha256": evaluation["sha256"]}
    return binding, config, manifest, references


@cached_paths
def verified_exclusions(config, manifest):
    """Return forbidden identities, exempting only an exact declared benchmark."""
    excluded_paths = [str(Path(row["manifest_path"]).parent) for row in manifest["excluded_manifests"]]
    ids, hashes, paths, excluded = _excluded(excluded_paths)
    if excluded != manifest["excluded_manifests"]:
        raise ValueError("An excluded manifest changed after test freeze")
    reused = config.get("reused_benchmark")
    if reused is None:
        if config.get("already_inspected", False) or manifest.get("reused_benchmark"):
            raise ValueError("Already inspected benchmark requires an explicit reuse binding")
        return ids, hashes, paths, excluded
    binding, _, previous, _ = _previous_benchmark(reused["test_directory"])
    if (not config.get("already_inspected") or binding != reused or
            manifest.get("reused_benchmark") != binding or
            manifest["images"] != previous["images"] or
            config["reference_sha256"] != binding["reference_sha256"]):
        raise ValueError("Reused benchmark must retain the exact declared image/reference population")
    allowed_path = Path(binding["manifest_path"])
    if not any(Path(row["manifest_path"]) == allowed_path and
               row["manifest_sha256"] == binding["manifest_sha256"] for row in excluded):
        raise ValueError("Reused benchmark must appear explicitly among excluded prior collections")
    forbidden = [directory for directory in excluded_paths
                 if Path(directory) / "manifest.json" != allowed_path]
    ids, hashes, paths, _ = _excluded(forbidden)
    return ids, hashes, paths, excluded


@cached_paths
def prepare(source, output, classifier_dir, exclude_studies, *, name, dataset_id,
            n=None, seed=20261021, scope, source_url=None):
    """Freeze a new test collection without using class labels for selection."""
    source, output = resolved(source), resolved(output)
    if n is not None and n < 1:
        raise ValueError("Test size must be positive, or omitted for a census")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new test directory; frozen artifacts cannot be overwritten")
    if not exclude_studies:
        raise ValueError("Explicitly exclude previously inspected studies")
    if not all((source / label).is_dir() for label in CLASSES):
        raise ValueError("Source must contain all nine CRC class folders")
    classifier, encoder, _, _ = _classifier(classifier_dir)
    old_ids, old_hashes, old_paths, excluded = _excluded([classifier_dir, *exclude_studies])
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    # DirEntry uses directory-provided file types where available. Avoid a
    # separate filesystem metadata request for every image on shared storage.
    paths = []
    for label in CLASSES:
        with os.scandir(source / label) as entries:
            paths.extend(Path(entry.path) for entry in entries
                         if Path(entry.name).suffix.lower() in suffixes and entry.is_file())
    paths.sort()
    random.Random(seed).shuffle(paths)
    rows, labels, rejections = [], [], []
    used_ids, used_hashes = set(old_ids), set(old_hashes)
    for path in paths:
        if n is not None and len(rows) == n:
            break
        if str(resolved(path)) in old_paths:
            rejections.append({"path": str(path), "reason": "previously inspected path"})
            continue
        sha = file_hash(path)
        identifier = sha[:20]
        if sha in used_hashes or identifier in used_ids:
            rejections.append({"path": str(path), "sha256": sha,
                               "reason": "duplicate bytes or previously inspected image"})
            continue
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError("Expected native 224x224 RGB tile")
        except Exception as error:
            rejections.append({"path": str(path), "reason": str(error)})
            continue
        used_ids.add(identifier)
        used_hashes.add(sha)
        rows.append({"id": identifier, "path": str(path), "sha256": sha,
                     "size_px": [224, 224], "mode": "RGB", "role": "test"})
        labels.append({"id": identifier, "label": path.parent.name, "role": "test",
                       "source": f"{dataset_id} supplied class folder"})
    if not rows or (n is not None and len(rows) != n):
        raise ValueError("Insufficient readable, unique, previously unused test images")
    selection = ("Census of eligible unique image bytes" if n is None else
                 "Seeded uniform permutation of image paths, retaining first eligible unique images; no class balancing")
    manifest = freeze(output / "manifest.json", {
        "schema_version": 1, "created_at": now(), "source_root": str(source),
        "source_file_count": len(paths), "seed": seed, "selection": selection,
        "n_test": len(rows), "requested_n": n, "images": rows,
        "excluded_manifests": excluded, "exclusions": rejections})
    references = freeze(output / "references.json", {
        "purpose": "Test evaluation only; never model training, selection or tuning", "labels": labels})
    write(output / "prompts.json", PROMPTS)
    config = freeze(output / "testset.json", {
        "schema_version": 1, "created_at": now(), "name": name, "dataset_id": dataset_id,
        "source_url": source_url, "scope": scope, "manifest_sha256": manifest["sha256"],
        "reference_sha256": references["sha256"], "classifier_sha256": classifier["sha256"],
        "coefficients_sha256": classifier["coefficients_sha256"], "encoder": encoder,
        "feature_key": classifier.get("feature_key", "embeddings"),
        "feature_dimension": classifier.get("feature_dimension", 512),
        "classifier_representation": classifier["representation"],
        "already_inspected": False, "reused_benchmark": None,
        "predictor_frozen_before_test_selection": True,
        "development_labels": classifier["refit_labels"],
        "overlap_checks": "Resolved paths, full file SHA-256, and image IDs against all excluded manifests; exact-byte deduplication within this collection",
        "overlap_limitations": "Near-duplicate images and foundation-model pretraining overlap are not assessed. Patient metadata are not inferred from filenames.",
        "target_population": f"The {len(rows)} retained tiles in this frozen test manifest"})
    return {"test_directory": str(output), "images": len(rows), "exclusions": len(rejections),
            "manifest_sha256": manifest["sha256"], "testset_sha256": config["sha256"]}


@cached_paths
def reuse_benchmark(previous_directory, output, classifier_dir, exclude_studies, *, name=None, scope=None):
    """Re-evaluate one explicitly named, previously inspected benchmark."""
    output = resolved(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new test directory; frozen artifacts cannot be overwritten")
    if not exclude_studies:
        raise ValueError("Explicitly exclude previously inspected studies")
    classifier, encoder, _, _ = _classifier(classifier_dir)
    binding, previous_config, previous, references = _previous_benchmark(previous_directory, open_references=True)
    # The whitelist contains one entire prior test manifest. An overlapping
    # development or study manifest still blocks reuse of any of its images.
    _, _, _, excluded = _excluded([classifier_dir, *exclude_studies, previous_directory])
    rows = previous["images"]
    if (not rows or len({row["id"] for row in rows}) != len(rows) or
            len({row["sha256"] for row in rows}) != len(rows)):
        raise ValueError("Reused benchmark image identities must be unique")
    warning = ("Previously inspected benchmark, reused with this frozen classifier; not a new untouched test. "
               "These labels do not enter fitting, but prior benchmark results may have informed development decisions.")
    inherited_scope = (scope or previous_config["scope"]).replace(
        "PLIP pretraining overlap is unknown", "Encoder pretraining overlap is unknown")
    manifest_data = {
        "schema_version": 1, "created_at": now(), "source_root": previous["source_root"],
        "source_file_count": previous["source_file_count"], "seed": previous["seed"],
        "selection": "Exact ordered population of an explicitly named, previously inspected benchmark",
        "n_test": len(rows), "requested_n": len(rows), "images": rows,
        "excluded_manifests": excluded, "exclusions": previous["exclusions"],
        "reused_benchmark": binding}
    config_data = {
        "schema_version": 1, "created_at": now(),
        "name": name or f"{previous_config['name']} (reused benchmark)",
        "dataset_id": previous_config["dataset_id"], "source_url": previous_config["source_url"],
        "scope": f"{warning} {inherited_scope}",
        "manifest_sha256": digest(manifest_data), "reference_sha256": references["sha256"],
        "classifier_sha256": classifier["sha256"], "coefficients_sha256": classifier["coefficients_sha256"],
        "encoder": encoder, "feature_key": classifier.get("feature_key", "embeddings"),
        "feature_dimension": classifier.get("feature_dimension", 512),
        "classifier_representation": classifier["representation"],
        "already_inspected": True, "reused_benchmark": binding,
        "predictor_frozen_before_test_selection": False,
        "predictor_frozen_before_benchmark_reuse": True,
        "development_labels": classifier["refit_labels"],
        "overlap_checks": "Exact named benchmark population reused; image IDs, full file SHA-256 and resolved paths remain disjoint from current development/study and all other excluded manifests",
        "overlap_limitations": "Previously inspected benchmark; near-duplicate images and foundation-model pretraining overlap are not assessed. Patient metadata are not inferred from filenames.",
        "target_population": f"The same {len(rows)} retained tiles as the declared previous benchmark"}
    ids, hashes, paths, _ = verified_exclusions(config_data, manifest_data)
    if (any(row["id"] in ids or row["sha256"] in hashes or str(resolved(row["path"])) in paths for row in rows)):
        raise ValueError("Reused benchmark overlaps current development/study or another excluded collection")
    manifest = freeze(output / "manifest.json", manifest_data)
    write(output / "references.json", references)
    write(output / "prompts.json", PROMPTS)
    config = freeze(output / "testset.json", config_data)
    return {"test_directory": str(output), "images": len(rows), "already_inspected": True,
            "manifest_sha256": manifest["sha256"], "testset_sha256": config["sha256"]}


def metrics(truth, predicted, name):
    result = classification_metrics(truth, predicted, name)
    recalls, f1s = [], []
    for row in result["per_class"]:
        predicted_count = sum(values[CLASSES.index(row["label"])] for values in result["confusion_matrix"])
        denominator = row["support"] + predicted_count
        row["f1"] = 2 * row["correct"] / denominator if denominator else 0.0
        f1s.append(row["f1"])
        if row["recall"] is not None:
            recalls.append(row["recall"])
    result["balanced_accuracy"] = float(np.mean(recalls))
    result["macro_f1"] = float(np.mean(f1s))
    result["metric_definitions"] = {
        "balanced_accuracy": "Mean recall over reference classes with at least one image",
        "macro_f1": "Mean F1 over all nine predefined classes; zero when both support and predicted count are zero",
        "confusion_matrix": "Rows are reference classes; columns are predicted classes"}
    return result


@cached_paths
def evaluate(directory, classifier_dir):
    """Apply the existing coefficients and evaluate once, without model fitting."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Test evaluation must run inside a Slurm allocation")
    directory = resolved(directory)
    if (directory / "test-evaluation.json").exists():
        raise FileExistsError("Test evaluation is frozen; do not overwrite an opened test result")
    config = read(directory / "testset.json")
    manifest = read(directory / "manifest.json")
    inference = read(directory / "predictions.json")
    for artifact in (config, manifest, inference):
        verify(artifact)
    classifier, encoder, coefficients, intercept = _classifier(classifier_dir)
    provenance = inference["provenance"]
    dimension = classifier.get("feature_dimension", 512)
    if (config["manifest_sha256"] != manifest["sha256"] or
            provenance["manifest_sha256"] != manifest["sha256"] or
            config["classifier_sha256"] != classifier["sha256"] or
            config["coefficients_sha256"] != classifier["coefficients_sha256"]):
        raise ValueError("Test artifacts belong to a different frozen predictor or collection")
    feature_key = classifier.get("feature_key", "embeddings")
    if (encoder_identity(config["encoder"]) != encoder or encoder_identity(provenance) != encoder or
            config.get("feature_key", "embeddings") != feature_key or
            config.get("feature_dimension", 512) != dimension or
            config.get("classifier_representation", classifier["representation"]) != classifier["representation"]):
        raise ValueError("Test encoder/preprocessing differs from classifier training representation")
    if provenance["prompts_sha256"] != digest(read(directory / "prompts.json")):
        raise ValueError("Inference prompt artifact changed")
    old_ids, old_hashes, old_paths, excluded = verified_exclusions(config, manifest)
    if not any(Path(row["manifest_path"]).parent == Path(classifier_dir).resolve() for row in excluded):
        raise ValueError("The full classifier experiment must be excluded")
    rows = manifest["images"]
    ids = [row["id"] for row in rows]
    hashes = [row["sha256"] for row in rows]
    if (not rows or len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes) or
            any(row["role"] != "test" for row in rows) or set(ids) & old_ids or
            set(hashes) & old_hashes or any(str(resolved(row["path"])) in old_paths for row in rows)):
        raise ValueError("Test collection overlaps prior images or has duplicate/invalid image roles")
    cache_path = directory / "predictions-embeddings.npz"
    cache_hash = file_hash(cache_path)
    if cache_hash != provenance["embedding_file_sha256"]:
        raise ValueError("Test embedding cache changed after inference")
    with np.load(cache_path, allow_pickle=False) as cache:
        if feature_key not in cache.files:
            raise ValueError("Test cache lacks the frozen classifier feature representation")
        cached_ids, features = cache["ids"].tolist(), cache[feature_key].copy()
    if (cached_ids != ids or [row["id"] for row in inference["images"]] != ids or
            features.shape != (len(rows), dimension) or
            not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)):
        raise ValueError("Test feature IDs/order, dimensions, or L2-normalized representation differ")
    scores_array, score_field, score_kind = classifier_scores(features, classifier, coefficients, intercept)
    prediction_rows = [{"id": row["id"], "role": "test",
                        "predicted_label": CLASSES[int(np.argmax(scores) if score_kind == "rbf_svm_votes" else np.argsort(scores)[-1])],
                        score_field: dict(zip(CLASSES, scores.tolist())),
                        "margin": float(np.sort(scores)[-1] - np.sort(scores)[-2]),
                        "score_kind": score_kind}
                       for row, scores in zip(rows, scores_array)]
    if score_kind == "rbf_svm_votes":
        from .rbf_predict import row_scores
        for row, scores in zip(prediction_rows, scores_array):
            row.update(row_scores(scores))
    prediction_path = directory / "classifier-predictions.json"
    bindings = {"testset_sha256": config["sha256"], "manifest_sha256": manifest["sha256"],
                "classifier_sha256": classifier["sha256"],
                "coefficients_sha256": classifier["coefficients_sha256"],
                "embedding_file_sha256": cache_hash, "inference_sha256": inference["sha256"],
                "feature_key": feature_key, "feature_dimension": dimension,
                "classifier_representation": classifier["representation"]}
    if prediction_path.exists():
        predictions = read(prediction_path)
        verify(predictions)
        if any(predictions[key] != value for key, value in bindings.items()) or predictions["images"] != prediction_rows:
            raise ValueError("Previously frozen test predictions differ")
    else:
        predictions = freeze(prediction_path, {"schema_version": 1, "created_at": now(),
                                              **bindings, "images": prediction_rows})
    # This is deliberately the first reference read during evaluation.
    references = read(directory / "references.json")
    verify(references)
    if references["sha256"] != config["reference_sha256"]:
        raise ValueError("Test references changed after test selection")
    if ([row["id"] for row in references["labels"]] != ids or
            any(row["role"] != "test" or row["label"] not in CLASSES for row in references["labels"])):
        raise ValueError("Test references must match every ordered test image exactly")
    result = metrics([row["label"] for row in references["labels"]],
                     [row["predicted_label"] for row in prediction_rows], classifier["name"])
    evaluation = freeze(directory / "test-evaluation.json", {
        "schema_version": 1, "created_at": now(), "name": config["name"],
        "dataset_id": config["dataset_id"], "source_url": config["source_url"],
        "active_model": result, **bindings, "prediction_sha256": predictions["sha256"],
        "reference_sha256": references["sha256"], "encoder": encoder,
        "already_inspected": config.get("already_inspected", False),
        "reused_benchmark": config.get("reused_benchmark"),
        "scope": config["scope"], "target_population": config["target_population"],
        "selection": manifest["selection"], "seed": manifest["seed"],
        "source_file_count": manifest["source_file_count"], "exclusions": len(manifest["exclusions"]),
        "excluded_manifests": excluded, "overlap_checks": config["overlap_checks"],
        "overlap_limitations": config["overlap_limitations"],
        "development_labels": config["development_labels"],
        "prediction_frozen_before_reference_read": True, "model_refit": False,
        "uncertainty": "Exact descriptive metrics for this fully labelled finite collection; no tile-independence confidence interval or patient-generalization estimate",
        "score_interpretation": (classifier["score_interpretation"] if score_kind == "rbf_svm_votes" else
                                 "Signed one-versus-rest linear SVM decision scores; not probabilities or calibrated confidence"
                                 if score_kind == "linear_svm_decision" else
                                 "Linear classifier probabilities; calibration has not been validated"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("prepare")
    selection.add_argument("source")
    selection.add_argument("output")
    selection.add_argument("--classifier-dir", required=True)
    selection.add_argument("--exclude-study", action="append", required=True)
    selection.add_argument("--name", required=True)
    selection.add_argument("--dataset-id", required=True)
    selection.add_argument("--n", type=int)
    selection.add_argument("--seed", type=int, default=20261021)
    selection.add_argument("--scope", required=True)
    selection.add_argument("--source-url")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("directory")
    evaluation.add_argument("--classifier-dir", required=True)
    reuse = commands.add_parser("reuse-benchmark")
    reuse.add_argument("previous_directory")
    reuse.add_argument("output")
    reuse.add_argument("--classifier-dir", required=True)
    reuse.add_argument("--exclude-study", action="append", required=True)
    reuse.add_argument("--name")
    reuse.add_argument("--scope")
    cached = commands.add_parser("reuse-embeddings")
    cached.add_argument("source_directory")
    cached.add_argument("destination_directory")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.source, args.output, args.classifier_dir, args.exclude_study,
                      name=args.name, dataset_id=args.dataset_id, n=args.n, seed=args.seed,
                      scope=args.scope, source_url=args.source_url))
    elif args.command == "reuse-benchmark":
        print(reuse_benchmark(args.previous_directory, args.output, args.classifier_dir,
                             args.exclude_study, name=args.name, scope=args.scope))
    elif args.command == "reuse-embeddings":
        result = reuse_embeddings(args.source_directory, args.destination_directory)
        print({"destination": args.destination_directory, "inference_sha256": result["sha256"]})
    else:
        result = evaluate(args.directory, args.classifier_dir)
        print({"name": result["name"], **{key: result["active_model"][key] for key in
               ("correct", "total", "accuracy", "balanced_accuracy", "macro_f1")},
               "sha256": result["sha256"]})
