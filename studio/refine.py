"""A preregistered, bounded CPU search using development labels only.

Existing study, pilot and test reference artifacts are never opened. Results
are development diagnostics, not a new independent performance estimate.
"""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time
import warnings

import numpy as np

from .improve import _validate_roles, encoder_identity, metrics
from .prepare import CLASSES
from .storage import digest, now, read, write
from .study import verify
from .testsets import _classifier, file_hash

BASELINE = "logistic-C10"
CANDIDATES = [
    {"id": BASELINE, "family": "logistic", "C": 10.0},
    {"id": "logistic-C3", "family": "logistic", "C": 3.0},
    {"id": "logistic-C30", "family": "logistic", "C": 30.0},
    {"id": "linear-svm-C0.1", "family": "linear_svm", "C": 0.1},
    {"id": "linear-svm-C1", "family": "linear_svm", "C": 1.0},
    {"id": "linear-svm-C10", "family": "linear_svm", "C": 10.0},
    {"id": "cosine-knn-k15", "family": "knn", "k": 15},
    {"id": "cosine-knn-k5", "family": "knn", "k": 5},
]
THRESHOLD = 2 / 900


def freeze(path, artifact):
    result = {**artifact, "sha256": digest(artifact)}
    write(path, result)
    return result


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ("refine.py", "improve.py", "quality.py", "testsets.py")}


def load_development(directory):
    """Read only development labels and frozen encoder/model metadata."""
    directory = Path(directory).resolve()
    classifier, encoder, _, _ = _classifier(directory)
    config = read(directory / "experiment.json")
    manifest = read(directory / "manifest.json")
    development = read(directory / "development-labels.json")
    inference = read(directory / "predictions.json")
    for artifact in (config, manifest, development, inference):
        verify(artifact)
    if (config["development_labels_sha256"] != development["sha256"] or
            config["manifest_sha256"] != manifest["sha256"] or
            encoder_identity(inference["provenance"]) != encoder or
            classifier.get("feature_key") != "classifier_embeddings" or
            encoder["model"] != "MahmoodLab/CONCH"):
        raise ValueError("Development artifacts must match the frozen CONCH pooled-feature classifier")
    _validate_roles(manifest, development)
    cache_path = directory / "predictions-embeddings.npz"
    cache_hash = file_hash(cache_path)
    if cache_hash != classifier["embedding_file_sha256"]:
        raise ValueError("Development feature cache changed")
    with np.load(cache_path, allow_pickle=False) as saved:
        ids = saved["ids"].tolist()
        features = saved[classifier["feature_key"]].copy()
    rows = manifest["images"]
    if (ids != [row["id"] for row in rows] or
            ids != [row["id"] for row in inference["images"]] or
            features.shape != (len(rows), classifier["feature_dimension"]) or
            not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Development feature identity/order/normalization differs")
    labels = {row["id"]: row["label"] for row in development["labels"]}
    blocks = {}
    for role in ("train", "validation"):
        indices = [i for i, row in enumerate(rows) if row["role"] == role]
        block_ids = [ids[i] for i in indices]
        block_labels = np.array([labels[identifier] for identifier in block_ids])
        if set(block_labels) != set(CLASSES):
            raise ValueError("Every development split must contain all nine tissue classes")
        blocks[role] = {"ids": block_ids, "features": features[indices], "labels": block_labels}
    bindings = {"experiment_sha256": config["sha256"], "manifest_sha256": manifest["sha256"],
                "development_labels_sha256": development["sha256"],
                "classifier_sha256": classifier["sha256"],
                "coefficients_sha256": classifier["coefficients_sha256"],
                "embedding_file_sha256": cache_hash, "encoder": encoder,
                "feature_key": classifier["feature_key"], "feature_dimension": features.shape[1]}
    return blocks, bindings


def prepare(directory, output, seed=20261024):
    """Write the protocol before fitting any candidate."""
    from sklearn.model_selection import StratifiedKFold

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new refinement directory; the development protocol is immutable")
    blocks, bindings = load_development(directory)
    training = blocks["train"]
    if min(np.count_nonzero(training["labels"] == label) for label in CLASSES) < 3:
        raise ValueError("Three-fold development CV requires at least three training examples per class")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = [{"fold": i, "fit_ids": [training["ids"][j] for j in fit],
              "score_ids": [training["ids"][j] for j in score]}
             for i, (fit, score) in enumerate(splitter.split(training["features"], training["labels"]))]
    plan = freeze(output / "refinement-plan.json", {
        "schema_version": 1, "created_at": now(), "source_directory": str(Path(directory).resolve()),
        **bindings, "implementation_sha256": implementation_hashes(), "seed": seed,
        "candidates": CANDIDATES, "baseline": BASELINE, "folds": folds,
        "training_ids": training["ids"], "validation_ids": blocks["validation"]["ids"],
        "training_labels": len(training["ids"]), "validation_labels": len(blocks["validation"]["ids"]),
        "selection": "Highest mean three-fold training-only CV balanced accuracy, rounded to 12 decimal places; ties follow candidate order (existing logistic C10 first)",
        "validation": "Fit selected candidate and baseline using original training labels only, then score each once on existing development validation labels",
        "gate": {"minimum_cv_gain": THRESHOLD, "minimum_validation_gain": THRESHOLD,
                 "rule": "Both gains versus logistic C10 must be at least 2/900; otherwise retain current model. No automatic promotion."},
        "solver": {"logistic": "lbfgs, max_iter=2000", "linear_svm": "LinearSVC dual=False, max_iter=20000",
                   "knn": "cosine distance, distance weights, brute force, n_jobs=1"},
        "failed_candidates": "Candidates with any failed/nonconverged CV fold are ineligible; baseline failure aborts comparison",
        "budget": {"cpus": 1, "memory_gb": 4, "time_minutes": 15, "gpus": 0},
        "scope": "Development-only search. Existing validation labels have already been used for regularization selection. No study, pilot or test reference labels are read. No untouched-test performance claim.",
        "raw_norm_experiments": "Excluded: the existing cache stores L2-normalized features, not their original norms",
        "automatic_promotion": False})
    return {"plan_sha256": plan["sha256"], "output": str(output), "candidates": len(CANDIDATES)}


def estimator(candidate, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    if candidate["family"] == "logistic":
        return LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=2000, random_state=seed)
    if candidate["family"] == "linear_svm":
        return LinearSVC(C=candidate["C"], dual=False, max_iter=20000, random_state=seed)
    if candidate["family"] == "knn":
        return KNeighborsClassifier(n_neighbors=candidate["k"], weights="distance", metric="cosine",
                                    algorithm="brute", n_jobs=1)
    raise ValueError("Unknown preregistered candidate family")


def fit_candidate(candidate, features, labels, seed):
    from sklearn.exceptions import ConvergenceWarning

    model = estimator(candidate, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    return model


def select_candidate(results, candidates):
    valid = {row["id"]: row for row in results if row["status"] == "complete"}
    if BASELINE not in valid:
        raise ValueError("Baseline CV failed; no valid development comparison")
    selected = max((row for row in candidates if row["id"] in valid),
                   key=lambda row: (round(valid[row["id"]]["mean_balanced_accuracy"], 12),
                                    -candidates.index(row)))
    return selected


def run(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Development search must run in a Slurm CPU allocation")
    started = time.monotonic()
    output = Path(output).resolve()
    if (output / "refinement-results.json").exists():
        raise FileExistsError("Development results are frozen; do not overwrite them")
    plan = read(output / "refinement-plan.json")
    verify(plan)
    if plan["implementation_sha256"] != implementation_hashes() or plan["candidates"] != CANDIDATES:
        raise ValueError("Implementation or preregistered candidate list changed after protocol freeze")
    blocks, bindings = load_development(plan["source_directory"])
    if any(plan[key] != value for key, value in bindings.items()):
        raise ValueError("Frozen development inputs changed after protocol freeze")
    training, validation = blocks["train"], blocks["validation"]
    if plan["training_ids"] != training["ids"] or plan["validation_ids"] != validation["ids"]:
        raise ValueError("Development split IDs changed")
    indices = {identifier: i for i, identifier in enumerate(training["ids"])}
    scored = []
    for fold in plan["folds"]:
        if (set(fold["fit_ids"]) & set(fold["score_ids"]) or
                set(fold["fit_ids"]) | set(fold["score_ids"]) != set(training["ids"])):
            raise ValueError("CV folds must partition training IDs without validation leakage")
        scored.extend(fold["score_ids"])
    if len(scored) != len(indices) or set(scored) != set(indices):
        raise ValueError("Every training image must be scored in exactly one CV fold")
    results = []
    for candidate in plan["candidates"]:
        result = {"id": candidate["id"], "candidate": candidate, "status": "complete", "folds": []}
        for fold in plan["folds"]:
            fit = [indices[identifier] for identifier in fold["fit_ids"]]
            score = [indices[identifier] for identifier in fold["score_ids"]]
            try:
                model = fit_candidate(candidate, training["features"][fit], training["labels"][fit], plan["seed"])
                scored_metrics = metrics(training["labels"][score], model.predict(training["features"][score]), candidate["id"])
                result["folds"].append({"fold": fold["fold"], "fit_count": len(fit), **scored_metrics})
            except Exception as error:
                result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                break
        if result["status"] == "complete":
            result["mean_balanced_accuracy"] = float(np.mean([row["balanced_accuracy"] for row in result["folds"]]))
        results.append(result)
        print({"candidate": result["id"], "status": result["status"],
               "mean_cv_balanced_accuracy": result.get("mean_balanced_accuracy")}, flush=True)
    # Freeze all CV decisions before consulting validation outcomes.
    selected = select_candidate(results, plan["candidates"])
    selection = freeze(output / "development-selection.json", {
        "created_at": now(), "plan_sha256": plan["sha256"], "candidates": results,
        "selected_candidate": selected, "validation_not_yet_scored": True})
    validation_results, models = {}, {}
    for candidate in (next(row for row in CANDIDATES if row["id"] == BASELINE), selected):
        if candidate["id"] in models:
            continue
        model = fit_candidate(candidate, training["features"], training["labels"], plan["seed"])
        models[candidate["id"]] = model
        validation_results[candidate["id"]] = metrics(
            validation["labels"], model.predict(validation["features"]), candidate["id"])
    by_id = {row["id"]: row for row in results}
    cv_gain = by_id[selected["id"]]["mean_balanced_accuracy"] - by_id[BASELINE]["mean_balanced_accuracy"]
    validation_gain = (validation_results[selected["id"]]["balanced_accuracy"] -
                       validation_results[BASELINE]["balanced_accuracy"])
    qualifies = (cv_gain + 1e-12 >= plan["gate"]["minimum_cv_gain"] and
                 validation_gain + 1e-12 >= plan["gate"]["minimum_validation_gain"])
    candidate_path = output / "selected-development-candidate.npz"
    model = models[selected["id"]]
    arrays = {"classes": model.classes_}
    if selected["family"] == "knn":
        arrays.update({"training_features": training["features"], "training_labels": training["labels"]})
    else:
        arrays.update({"coefficients": model.coef_, "intercept": model.intercept_})
    with candidate_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    # Verify that the original frozen classifier still matches its bindings.
    _, final_bindings = load_development(plan["source_directory"])
    if final_bindings != bindings:
        raise ValueError("Original model/development artifacts changed during search")
    result = freeze(output / "refinement-results.json", {
        "schema_version": 1, "created_at": now(), "plan_sha256": plan["sha256"],
        "selection_sha256": selection["sha256"], **bindings, "candidates": results,
        "selected_candidate": selected, "validation_results": validation_results,
        "gate": {**plan["gate"], "observed_cv_gain": cv_gain,
                 "observed_validation_gain": validation_gain, "qualifies_for_fresh_testing": qualifies},
        "decision": ("Candidate qualifies for a new untouched evaluation; current model remains unchanged"
                     if qualifies else "Bounded search did not meet both improvement thresholds; retain current model"),
        "candidate_file": str(candidate_path), "candidate_file_sha256": file_hash(candidate_path),
        "candidate_fit_labels": len(training["ids"]), "refit_on_validation": False,
        "development_only": True, "external_evaluation_performed": False,
        "current_model_changed": False, "automatic_promotion": False, "scope": plan["scope"],
        "elapsed_seconds": time.monotonic() - started, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("source_directory")
    preparation.add_argument("output")
    preparation.add_argument("--seed", type=int, default=20261024)
    execution = commands.add_parser("run")
    execution.add_argument("output")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.source_directory, args.output, args.seed), flush=True)
    else:
        result = run(args.output)
        print({key: result[key] for key in ("decision", "selected_candidate", "gate", "elapsed_seconds", "sha256")}, flush=True)
