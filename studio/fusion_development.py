"""Bounded development-only coordinate concatenation of two frozen encoders."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np

from .improve import _validate_roles, encoder_identity, metrics
from .paths import cached_paths
from .prepare import CLASSES
from .refine import fit_candidate
from .storage import now, read, write
from .study import verify
from .testsets import _classifier, file_hash, freeze

THRESHOLD = 2 / 900
CANDIDATES = [{"id": f"fusion-w{weight:g}-C{c:g}", "family": "logistic", "C": c,
               "musk_weight": weight}
              for weight in (.25, .5, .75) for c in (1., 10., 100.)]
CONTROLS = [{"id": "CONCH-C10", "family": "logistic", "C": 10., "encoder": "CONCH"},
            {"id": "MUSK-C100", "family": "logistic", "C": 100., "encoder": "MUSK"}]
SCOPE = ("Development-only search on existing training folds and previously used development-validation labels. "
         "No study, pilot, audit or test reference labels are read. Reused development scores are selection "
         "evidence, not an independent generalization estimate. No class-specific routing or automatic promotion.")


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ("fusion_development.py", "refine.py", "improve.py", "quality.py", "testsets.py", "storage.py", "study.py", "paths.py")}


def load_encoder(directory, model):
    """Verify all cached features, but return labels/features only for development roles."""
    directory = Path(directory).resolve()
    classifier, identity, _, _ = _classifier(directory)
    config, manifest, labels, inference = (read(directory / name) for name in
        ("experiment.json", "manifest.json", "development-labels.json", "predictions.json"))
    for artifact in (config, manifest, labels, inference):
        verify(artifact)
    if (identity["model"] != model or classifier["feature_key"] != "classifier_embeddings" or
            config["development_labels_sha256"] != labels["sha256"] or
            config["manifest_sha256"] != manifest["sha256"] or
            encoder_identity(inference["provenance"]) != identity):
        raise ValueError("Frozen encoder/development binding changed")
    _validate_roles(manifest, labels)
    cache = directory / "predictions-embeddings.npz"
    if file_hash(cache) != classifier["embedding_file_sha256"]:
        raise ValueError("Development feature cache changed")
    with np.load(cache, allow_pickle=False) as saved:
        ids, features = saved["ids"].tolist(), saved["classifier_embeddings"].copy()
    if (ids != [row["id"] for row in manifest["images"]] or
            ids != [row["id"] for row in inference["images"]] or len(set(ids)) != len(ids) or
            features.shape != (len(ids), classifier["feature_dimension"]) or
            not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Development feature IDs/order/dimensions/normalization changed")
    by_id = {row["id"]: row["label"] for row in labels["labels"]}
    blocks = {}
    for role in ("train", "validation"):
        indices = [i for i, row in enumerate(manifest["images"]) if row["role"] == role]
        block_ids = [ids[i] for i in indices]
        targets = np.array([by_id[item] for item in block_ids])
        if set(targets) != set(CLASSES):
            raise ValueError("Each development role requires all nine classes")
        blocks[role] = {"ids": block_ids, "features": features[indices], "labels": targets}
    binding = {"directory": str(directory), "classifier_sha256": classifier["sha256"],
               "coefficients_sha256": classifier["coefficients_sha256"], "experiment_sha256": config["sha256"],
               "manifest_sha256": manifest["sha256"], "development_labels_sha256": labels["sha256"],
               "encoder_predictions_sha256": inference["sha256"], "embedding_file_sha256": classifier["embedding_file_sha256"],
               "feature_key": "classifier_embeddings", "feature_dimension": features.shape[1], "encoder": identity}
    return blocks, binding


def load_pair(conch, musk):
    a, a_binding = load_encoder(conch, "MahmoodLab/CONCH")
    b, b_binding = load_encoder(musk, "xiangjx/musk")
    if (a_binding["manifest_sha256"] != b_binding["manifest_sha256"] or
            a_binding["development_labels_sha256"] != b_binding["development_labels_sha256"] or
            any(a[role]["ids"] != b[role]["ids"] or not np.array_equal(a[role]["labels"], b[role]["labels"])
                for role in ("train", "validation"))):
        raise ValueError("Paired encoders must have identical image identities, development roles and labels")
    return {"CONCH": a, "MUSK": b}, {"CONCH": a_binding, "MUSK": b_binding}


def concatenate(conch, musk, weight):
    """Block weights are squared-norm shares, not cross-space vector averages."""
    if not 0 < weight < 1 or len(conch) != len(musk) or conch.ndim != 2 or musk.ndim != 2:
        raise ValueError("Concatenation requires paired matrices and an interior block weight")
    return np.concatenate((conch.astype(np.float32) * np.float32(np.sqrt(1 - weight)),
                           musk.astype(np.float32) * np.float32(np.sqrt(weight))), axis=1)


def validate_folds(folds, training, validation):
    training, validation = set(training), set(validation)
    if training & validation or len(folds) != 3:
        raise ValueError("Training and validation roles must be disjoint with exactly three folds")
    scored = []
    for fold in folds:
        fit, score = fold["fit_ids"], fold["score_ids"]
        if (len(set(fit)) != len(fit) or len(set(score)) != len(score) or set(fit) & set(score) or
                set(fit) | set(score) != training):
            raise ValueError("CV must partition training IDs only, without validation leakage")
        scored.extend(score)
    if len(scored) != len(training) or set(scored) != training:
        raise ValueError("Each training image must be scored exactly once")


@cached_paths
def prepare(conch, musk, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new immutable fusion development directory")
    blocks, bindings = load_pair(conch, musk)
    source_plan = read(Path(musk) / "musk-development-plan.json")
    source_result = read(Path(musk) / "musk-development-results.json")
    for artifact in (source_plan, source_result):
        verify(artifact)
    if (source_result["plan_sha256"] != source_plan["sha256"] or
            source_plan["training_ids"] != blocks["CONCH"]["train"]["ids"] or
            source_plan["validation_ids"] != blocks["CONCH"]["validation"]["ids"] or
            source_result["selected_candidate"]["C"] != 100. or
            source_plan["baseline_bindings"]["classifier_sha256"] != bindings["CONCH"]["classifier_sha256"] or
            source_plan["baseline_bindings"]["embedding_file_sha256"] != bindings["CONCH"]["embedding_file_sha256"] or
            source_result["encoder"]["embedding_file_sha256"] != bindings["MUSK"]["embedding_file_sha256"] or
            source_result["encoder"]["manifest_sha256"] != bindings["MUSK"]["manifest_sha256"] or
            any(source_result["baseline_bindings"][key] != bindings["CONCH"][key] for key in
                ("classifier_sha256", "manifest_sha256", "development_labels_sha256", "embedding_file_sha256")) or
            encoder_identity(source_result["encoder"]) != bindings["MUSK"]["encoder"]):
        raise ValueError("Original development folds or fixed single-encoder controls changed")
    validate_folds(source_plan["folds"], source_plan["training_ids"], source_plan["validation_ids"])
    musk_cv = next(row for row in source_result["candidates"]
                   if row["candidate"] == source_result["selected_candidate"])["mean_balanced_accuracy"]
    controls = {"CONCH": {"cv_balanced_accuracy": source_result["baseline_cv"]["mean_balanced_accuracy"],
                           "validation": source_result["baseline_validation"]},
                "MUSK": {"cv_balanced_accuracy": musk_cv, "validation": source_result["validation"]}}
    return freeze(output / "fusion-development-plan.json", {
        "schema_version": 1, "created_at": now(), "bindings": bindings, "source_plan_sha256": source_plan["sha256"],
        "source_result_sha256": source_result["sha256"], "folds": source_plan["folds"], "seed": source_plan["seed"],
        "training_ids": source_plan["training_ids"], "validation_ids": source_plan["validation_ids"],
        "candidates": CANDIDATES, "controls": CONTROLS, "control_scores": controls,
        "representation": "Coordinate concatenation [sqrt(1-w)*L2 CONCH pooled features, sqrt(w)*L2 MUSK CLS features]; w is MUSK squared-norm share. Float32 weighting; no averaging unrelated coordinates or fitted transform.",
        "feature_dimension": sum(item["feature_dimension"] for item in bindings.values()),
        "selection": "Highest mean training-only three-fold CV balanced accuracy rounded to12 decimals; ties lowerC, then weight nearest0.5, then lowerweight. Seal selection before scoring it once on existing development-validation labels.",
        "gate": {"minimum_cv_gain": THRESHOLD, "minimum_validation_gain": THRESHOLD,
                 "cv_control": max(item["cv_balanced_accuracy"] for item in controls.values()),
                 "validation_control": max(item["validation"]["balanced_accuracy"] for item in controls.values()),
                 "rule": "At least2/900 improvement over the stronger frozen single-encoder control for BOTH CV and existing development-validation balanced accuracy. Otherwise no new test is authorized by this result."},
        "control_diagnostic": "Reconstruct fixed CONCH C10 and MUSK C100 OOF predictions on original folds. Report paired correctness and oracle-union ceiling descriptively only; controls were previously selected on these development data.",
        "solver": "Multinomial LogisticRegression lbfgs,max_iter2000; nonconvergence invalidates candidate, control failure aborts",
        "implementation_sha256": implementation_hashes(), "scope": SCOPE,
        "budget": {"cpus": 2, "memory_gb": 4, "minutes": 15, "gpus": 0},
        "compute_cost": "Development reuses both existing caches. Future deployment requires BOTH encoder forwards and1536 classifier coordinates instead of512 for CONCH alone; retrieval spaces remain separate.",
        "automatic_promotion": False, "test_protocol_not_yet_defined": True})


def select(results):
    eligible = [row for row in results if row["status"] == "complete"]
    if not eligible:
        raise ValueError("Every fusion candidate failed")
    return max(eligible, key=lambda row: (round(row["mean_balanced_accuracy"], 12),
        -row["candidate"]["C"], -abs(row["candidate"]["musk_weight"] - .5), -row["candidate"]["musk_weight"]))


def paired_diagnostic(labels, conch, musk):
    a, b = np.asarray(conch) == labels, np.asarray(musk) == labels
    return {"images": len(labels), "both_correct": int(np.sum(a & b)), "both_wrong": int(np.sum(~a & ~b)),
            "musk_only_correct": int(np.sum(~a & b)), "conch_only_correct": int(np.sum(a & ~b)),
            "oracle_union_correct": int(np.sum(a | b)), "oracle_union_accuracy": float(np.mean(a | b)),
            "interpretation": "Descriptive development-only complementarity; oracle uses true labels and is not a deployable classifier"}


def verify_control_sources(plan):
    for filename, expected in (("musk-development-plan.json", plan["source_plan_sha256"]),
                               ("musk-development-results.json", plan["source_result_sha256"])):
        source = read(Path(plan["bindings"]["MUSK"]["directory"]) / filename)
        verify(source)
        if source["sha256"] != expected:
            raise ValueError("Frozen single-encoder control source changed")


@cached_paths
def run(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Fusion development requires a scheduled Slurm CPU allocation")
    output, started = Path(output).resolve(), time.monotonic()
    if (output / "fusion-development-results.json").exists():
        raise FileExistsError("Fusion development results are frozen")
    plan = read(output / "fusion-development-plan.json")
    verify(plan)
    if plan["implementation_sha256"] != implementation_hashes() or plan["candidates"] != CANDIDATES or plan["controls"] != CONTROLS:
        raise ValueError("Frozen fusion implementation or candidate protocol changed")
    verify_control_sources(plan)
    blocks, bindings = load_pair(*(plan["bindings"][name]["directory"] for name in ("CONCH", "MUSK")))
    if bindings != plan["bindings"] or any(plan[f"{role}_ids"] != blocks["CONCH"][block]["ids"] for role, block in
                                           (("training", "train"), ("validation", "validation"))):
        raise ValueError("Frozen development feature bindings or roles changed")
    validate_folds(plan["folds"], plan["training_ids"], plan["validation_ids"])
    labels = blocks["CONCH"]["train"]["labels"]
    index = {item: i for i, item in enumerate(plan["training_ids"])}

    def cv(candidate, features):
        folds, predictions = [], np.empty(len(labels), dtype="<U4")
        for fold in plan["folds"]:
            fit, score = ([index[item] for item in fold[key]] for key in ("fit_ids", "score_ids"))
            model = fit_candidate(candidate, features[fit], labels[fit], plan["seed"])
            predictions[score] = model.predict(features[score])
            folds.append({"fold": fold["fold"], **metrics(labels[score], predictions[score], candidate["id"])})
        return {"candidate": candidate, "status": "complete", "folds": folds,
                "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in folds]))}, predictions

    controls, oof = {}, {}
    for control in CONTROLS:
        name = control["encoder"]
        controls[name], oof[name] = cv(control, blocks[name]["train"]["features"])
        if not np.isclose(controls[name]["mean_balanced_accuracy"], plan["control_scores"][name]["cv_balanced_accuracy"], atol=1e-12, rtol=0):
            raise ValueError("Reconstructed single-encoder control differs from frozen original CV")
    diagnostic = paired_diagnostic(labels, oof["CONCH"], oof["MUSK"])
    freeze(output / "control-oof-diagnostic.json", {"plan_sha256": plan["sha256"], "controls": controls,
           "paired_correctness": diagnostic, "rows": [{"id": item, "label": str(labels[i]),
           "CONCH": str(oof["CONCH"][i]), "MUSK": str(oof["MUSK"][i])} for i, item in enumerate(plan["training_ids"])]})
    results = []
    for candidate in CANDIDATES:
        features = concatenate(blocks["CONCH"]["train"]["features"], blocks["MUSK"]["train"]["features"], candidate["musk_weight"])
        try:
            result, _ = cv(candidate, features)
        except Exception as error:
            result = {"candidate": candidate, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        results.append(result)
        print({"candidate": candidate["id"], "status": result["status"], "cv_balanced_accuracy": result.get("mean_balanced_accuracy")}, flush=True)
    chosen = select(results)
    selection = freeze(output / "fusion-development-selection.json", {"created_at": now(), "plan_sha256": plan["sha256"],
        "candidates": results, "selected_candidate": chosen["candidate"], "validation_not_yet_scored": True})
    weight = chosen["candidate"]["musk_weight"]
    train, validation = (concatenate(blocks["CONCH"][role]["features"], blocks["MUSK"][role]["features"], weight)
                         for role in ("train", "validation"))
    model = fit_candidate(chosen["candidate"], train, labels, plan["seed"])
    score = metrics(blocks["CONCH"]["validation"]["labels"], model.predict(validation), chosen["candidate"]["id"])
    cv_gain = chosen["mean_balanced_accuracy"] - plan["gate"]["cv_control"]
    validation_gain = score["balanced_accuracy"] - plan["gate"]["validation_control"]
    qualifies = cv_gain + 1e-12 >= THRESHOLD and validation_gain + 1e-12 >= THRESHOLD
    path = output / "selected-development-candidate.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, classes=model.classes_, coefficients=model.coef_, intercept=model.intercept_,
                            musk_weight=np.array(weight), conch_dimension=np.array(bindings["CONCH"]["feature_dimension"]),
                            musk_dimension=np.array(bindings["MUSK"]["feature_dimension"]))
    _, after = load_pair(*(bindings[name]["directory"] for name in ("CONCH", "MUSK")))
    if after != bindings:
        raise ValueError("An original classifier or encoder cache changed during development")
    verify_control_sources(plan)
    result = freeze(output / "fusion-development-results.json", {
        "schema_version": 1, "created_at": now(), "plan_sha256": plan["sha256"], "selection_sha256": selection["sha256"],
        "bindings": bindings, "selected_candidate": chosen["candidate"], "candidates": results, "validation": score,
        "controls": controls, "control_scores": plan["control_scores"], "paired_oof_correctness": diagnostic,
        "gate": {**plan["gate"], "observed_cv_gain": cv_gain, "observed_validation_gain": validation_gain,
                 "qualifies_for_fresh_testing": qualifies},
        "decision": "Qualifies for a separately frozen fresh-test proposal; current CONCH remains unchanged" if qualifies else
                    "Fusion did not meet both stronger-control development thresholds; current CONCH remains unchanged",
        "candidate_file_sha256": file_hash(path), "candidate_fit_labels": len(labels), "refit_on_validation": False,
        "feature_dimension": train.shape[1], "cached_feature_bytes": train.nbytes + validation.nbytes,
        "development_only": True, "test_references_read": False, "current_model_changed": False, "automatic_promotion": False,
        "scope": SCOPE, "compute_cost": plan["compute_cost"], "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}})
    return result


def publish(output, prefix):
    output, prefix = Path(output), Path(prefix)
    result, plan, selection = (read(output / name) for name in
        ("fusion-development-results.json", "fusion-development-plan.json", "fusion-development-selection.json"))
    for artifact in (result, plan, selection):
        verify(artifact)
    if (result["plan_sha256"] != plan["sha256"] or result["selection_sha256"] != selection["sha256"] or
            selection["selected_candidate"] != result["selected_candidate"] or
            file_hash(output / "selected-development-candidate.npz") != result["candidate_file_sha256"]):
        raise ValueError("Frozen fusion report lineage changed")
    text = ["# Development-only CONCH/MUSK feature concatenation", "", result["decision"], "", SCOPE, "",
            "| Procedure | CV balanced accuracy | Validation balanced accuracy |", "| --- | ---: | ---: |"]
    for name, row in result["control_scores"].items():
        text.append(f"| {name} control | {row['cv_balanced_accuracy']:.6f} | {row['validation']['balanced_accuracy']:.6f} |")
    chosen = next(row for row in result["candidates"] if row["candidate"] == result["selected_candidate"])
    text.append(f"| Selected fusion | {chosen['mean_balanced_accuracy']:.6f} | {result['validation']['balanced_accuracy']:.6f} |")
    diagnostic = result["paired_oof_correctness"]
    text.extend(["", f"Selected C={result['selected_candidate']['C']:g}, MUSK squared-norm share={result['selected_candidate']['musk_weight']:g}. "
                 f"Validation correct: {result['validation']['correct']}/{result['validation']['total']}. "
                 "Both gains must be at least2/900 above the stronger single-encoder control.", "",
                 f"Fixed-control OOF complementarity: {diagnostic['musk_only_correct']} MUSK-only correct, "
                 f"{diagnostic['conch_only_correct']} CONCH-only correct, {diagnostic['both_wrong']} both wrong. "
                 f"Oracle union {diagnostic['oracle_union_correct']}/{diagnostic['images']} is descriptive and cannot be deployed.", "",
                 plan["representation"], "", plan["compute_cost"], "",
                 f"Scheduled CPU elapsed: {result['elapsed_seconds']:.1f}s; no GPU or external reference labels used.", "",
                 f"Plan SHA-256: `{plan['sha256']}`. Result SHA-256: `{result['sha256']}`.", ""])
    markdown = "\n".join(text)
    if prefix.with_suffix(".json").exists():
        if read(prefix.with_suffix(".json")) != result or prefix.with_suffix(".md").read_text() != markdown:
            raise FileExistsError("Published fusion result is immutable")
    else:
        write(prefix.with_suffix(".json"), result)
        with prefix.with_suffix(".md").open("x") as handle:
            handle.write(markdown)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "publish"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    value = {"prepare": prepare, "run": run, "publish": publish}[args.command](*args.paths)
    print({key: value[key] for key in ("sha256", "decision", "selected_candidate", "gate") if key in value}, flush=True)
