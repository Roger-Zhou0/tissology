"""Trusted, one-shot fresh evaluation and isolated study publication.

HTTP supplies only an existing opaque candidate ID. An administrator freezes
paths/resource policy in fresh-config.json. Only Slurm workers select images,
encode or evaluate; public responses omit paths and unreviewed references.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import io
import os
from pathlib import Path
import random
import subprocess

import numpy as np
from PIL import Image

from .conch_infer import file_hash
from .feedback import identifier
from .feedback_storage import freeze
from .improve import encoder_identity
from .paths import resolved
from .paths_prefix import prefix_paths
from .prepare import CLASSES, ENDPOINTS, PROMPTS
from .storage import digest, now, read, write
from .study import verify
from .testsets import _classifier, metrics

ROOT = Path(os.environ.get("TISSUE_LEGACY_ARTIFACT_ROOT", "./artifacts")).resolve()
REPO = Path(__file__).resolve().parent.parent
CONTROL_SHA = "c55b4f70800235d2a7882099c877de8a0c76e61bceb523f2bb426196e73b7dce"
SCOPE = ("Previously unused same-source NCT-CRC tiles, exact supplied-label agreement on a finite collection. "
         "No patient IDs, no claim of new-patient independence, pretraining independence, clinical validation, "
         "or statistically established improvement. The acceptance gate checks observed non-regression only.")


def frozen(path):
    value = read(path)
    verify(value)
    return value


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Fresh selection, inference and evaluation require Slurm")


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def source_identity():
    names = ("studio/feedback_evaluation.py", "studio/feedback_storage.py", "studio/conch_infer.py",
             "studio/testsets.py", "studio/improve.py", "studio/quality.py", "studio/storage.py",
             "studio/study.py", "studio/paths.py", "studio/paths_prefix.py", "studio/prepare.py",
             "scripts/feedback_fresh_cpu.sbatch", "scripts/feedback_fresh_gpu.sbatch")
    return {name: file_hash(REPO / name) for name in names}


def configure(workspace, *, project_root=ROOT, source="/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples",
              control=None, test_count=2000, study_count=600, pilot_count=90, expected_control=CONTROL_SHA):
    """Administrator CLI only: no paths or budgets are accepted from HTTP."""
    require_slurm()
    workspace, project = Path(workspace).resolve(), Path(project_root).resolve()
    if workspace.parent != project:
        raise ValueError("Feedback workspace must be directly inside the configured project")
    catalog = frozen(workspace / "catalog.json")
    control = Path(control or project / "conch-experiment-001").resolve()
    classifier, identity, _, _ = _classifier(control)
    if (classifier["sha256"] != expected_control or catalog["base_classifier_sha256"] != expected_control
            or classifier.get("feature_key") != "classifier_embeddings"):
        raise ValueError("Fresh control must be the exact registered deployed classifier")
    if any(type(n) is not int or n < 1 for n in (test_count, study_count, pilot_count)):
        raise ValueError("Require positive fixed cohort counts")
    return freeze(workspace / "fresh-config.json", {
        "schema": "feedback-fresh-config-v1", "created_at": now(), "workspace": str(workspace),
        "project_root": str(project), "source_root": str(Path(source).resolve()),
        "control_directory": str(control), "control_classifier_sha256": expected_control,
        "catalog_sha256": catalog["sha256"], "encoder": identity,
        "test_count": test_count, "study_count": study_count, "pilot_count": pilot_count,
        "checkpoint": str(project / "hf-gated/hub/models--MahmoodLab--CONCH/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin"),
        "code_directory": str(project / "vendor/CONCH"), "dependency_directory": str(project / "conch-deps"),
        "gate": "Development gate passes; fresh candidate accuracy and balanced accuracy each >= deployed 5400-label control; every class supported and no recall drop exceeds 5 percentage points",
        "evaluation_kind": "one-shot finite same-source non-regression, not an improvement or patient-generalization claim",
        "seed_rule": "20261101 + integer(first8hex candidate ID)",
        "roles": "First test_count selected images are test; next study_count are study; last pilot_count are audit pilot",
        "gpu_budget": "One GPU,15minutes; only the newly selected images are encoded",
    })


def _context(workspace, candidate_id, *, require_model=True):
    workspace = Path(workspace).resolve()
    candidate_id = identifier(candidate_id)
    catalog = frozen(workspace / "catalog.json")
    config = frozen(workspace / "fresh-config.json")
    if config["workspace"] != str(workspace) or config["catalog_sha256"] != catalog["sha256"] or workspace.parent != Path(config["project_root"]):
        raise ValueError("Fresh configuration belongs to a different workspace")
    folder = workspace / "candidates" / candidate_id
    plan = frozen(folder / "plan.json")
    if (plan["id"] != candidate_id or plan["catalog_sha256"] != catalog["sha256"]
            or plan["base_classifier_sha256"] != config["control_classifier_sha256"]):
        raise ValueError("Candidate/catalog/control identities differ")
    context = {"workspace": workspace, "folder": folder, "fresh": folder / "fresh", "plan": plan,
               "catalog": catalog, "config": config, "id": candidate_id}
    if not require_model:
        return context
    if (folder / "failure.json").exists():
        raise ValueError("Candidate training has a failure receipt; trusted reconciliation is required")
    model, result = frozen(folder / "model.json"), frozen(folder / "result.json")
    validation = frozen(folder / "validation-predictions.json")
    if (model["plan_sha256"] != plan["sha256"] or result["plan_sha256"] != plan["sha256"]
            or result["model_sha256"] != model["sha256"]
            or result["validation_predictions_sha256"] != validation["sha256"]
            or validation["model_sha256"] != model["sha256"]
            or file_hash(folder / "coefficients.npz") != model["coefficients_sha256"]
            or model["feature_key"] != "classifier_embeddings"
            or encoder_identity(model["encoder_identity"]) != encoder_identity(config["encoder"])):
        raise ValueError("Candidate model/validation/encoder binding changed")
    control, identity, coefficients, intercept = _classifier(config["control_directory"])
    if control["sha256"] != config["control_classifier_sha256"] or identity != encoder_identity(config["encoder"]):
        raise ValueError("Deployed control changed")
    with np.load(folder / "coefficients.npz", allow_pickle=False) as cache:
        candidate_coef, candidate_intercept, classes = cache["coefficients"].copy(), cache["intercept"].copy(), cache["classes"].tolist()
    dimension = control["feature_dimension"]
    if (classes != CLASSES or model["classes"] != CLASSES or candidate_coef.shape != (9, dimension)
            or candidate_intercept.shape != (9,) or not np.isfinite(candidate_coef).all()
            or not np.isfinite(candidate_intercept).all()):
        raise ValueError("Invalid candidate numeric model")
    context.update(model=model, development=result, control=control, control_coefficients=coefficients,
                   control_intercept=intercept, coefficients=candidate_coef, intercept=candidate_intercept)
    return context


def _check_request(context, *, execution=False):
    request = frozen(context["fresh"] / "request.json")
    expected = {"candidate_plan_sha256": context["plan"]["sha256"], "candidate_model_sha256": context["model"]["sha256"],
                "candidate_coefficients_sha256": context["model"]["coefficients_sha256"],
                "control_classifier_sha256": context["control"]["sha256"],
                "control_coefficients_sha256": context["control"]["coefficients_sha256"],
                "config_sha256": context["config"]["sha256"], "candidate_development_result_sha256": context["development"]["sha256"]}
    if any(request.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen fresh request no longer matches its model or configuration")
    if any(file_hash(context["fresh"] / "source" / name) != expected for name, expected in request["source_files"].items()):
        raise ValueError("Frozen execution source snapshot changed")
    if execution and request["source_files"] != source_identity():
        raise ValueError("Fresh execution source changed after protocol freeze")
    return request


def _submit(script, workspace, candidate_id, stage=None, dependency=None):
    args = ["sbatch", "--parsable"]
    if dependency:
        args.append("--dependency=" + dependency)
    args.extend([str(REPO / "scripts" / script), str(workspace), candidate_id])
    if stage:
        args.append(stage)
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True, timeout=20)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Invalid scheduler job identifier")
    return job


def request_evaluation(workspace, candidate_id, *, submit=True):
    context = _context(workspace, candidate_id)
    if not context["development"]["development_gate_passed"]:
        raise ValueError("Development gate failed; do not consume a fresh test for this candidate")
    with locked(context["workspace"] / ".fresh.lock"):
        if (context["fresh"] / "request.json").exists():
            return eligibility(workspace, candidate_id)
        sources = source_identity()
        for name, expected in sources.items():
            path = context["fresh"] / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write((REPO / name).read_bytes())
            if file_hash(path) != expected:
                raise ValueError("Execution source changed while freezing the protocol")
        request = freeze(context["fresh"] / "request.json", {
            "schema": "feedback-fresh-request-v1", "created_at": now(), "candidate_id": context["id"],
            "candidate_plan_sha256": context["plan"]["sha256"], "candidate_model_sha256": context["model"]["sha256"],
            "candidate_coefficients_sha256": context["model"]["coefficients_sha256"],
            "candidate_development_result_sha256": context["development"]["sha256"],
            "control_classifier_sha256": context["control"]["sha256"],
            "control_coefficients_sha256": context["control"]["coefficients_sha256"],
            "control_training_labels": context["control"]["refit_labels"], "candidate_training_labels": context["model"]["training_count"],
            "correction_sources": sorted({r["source"] for r in context["plan"]["corrections"]}),
            "config_sha256": context["config"]["sha256"], "source_files": sources,
            "seed": 20261101 + int(context["id"][:8], 16), "gate": context["config"]["gate"],
            "scope": SCOPE, "new_study_requires_gate_pass": True, "candidate_refit_after_fresh_test": False,
            "gate_parameters": {"schema": 1, "accuracy_min_difference": 0, "balanced_min_difference": 0,
                                "maximum_recall_drop": .05, "all_classes_required": True},
        })
        if submit:
            jobs = {}
            try:
                jobs["prepare"] = _submit("feedback_fresh_cpu.sbatch", workspace, candidate_id, "prepare")
                jobs["encode"] = _submit("feedback_fresh_gpu.sbatch", workspace, candidate_id,
                                         dependency=f"afterok:{jobs['prepare']},singleton")
                jobs["evaluate"] = _submit("feedback_fresh_cpu.sbatch", workspace, candidate_id, "evaluate",
                                           dependency=f"afterok:{jobs['encode']}")
                freeze(context["fresh"] / "jobs.json", {"request_sha256": request["sha256"], "jobs": jobs})
            except (OSError, subprocess.SubprocessError, ValueError):
                freeze(context["fresh"] / "failure.json", {"request_sha256": request["sha256"], "stage": "submission",
                                                            "reason": "Scheduler submission failed", "jobs": jobs})
    return eligibility(workspace, candidate_id)


def discover_manifests(project, current):
    """All canonical historical CRC manifests, plus every reserved feedback cohort."""
    project, current = Path(project), Path(current)
    paths = set(project.glob("*/manifest.json"))
    paths.update(project.glob("*/candidates/*/fresh/encoding/manifest.json"))
    # A failed preparation may have staged images before publishing its final
    # manifest. Reserve those exact bytes as well. The exclusive project lock
    # means no other preparation can be adding to this snapshot concurrently.
    for directory in project.glob("*/candidates/*/fresh/reservations"):
        if directory.is_relative_to(current) or (directory.parent / "encoding/manifest.json").exists():
            continue
        partial = directory.parent / "partial-reservation-manifest.json"
        if not partial.exists():
            rows = []
            for entry in sorted(directory.glob("*.json")):
                row = frozen(entry)
                if entry.stem != row["sha256"][:20] or row["image_sha256"][:20] != row["id"]:
                    raise ValueError("Unexpected partial reserved image identity")
                rows.append({"id": row["id"], "sha256": row["image_sha256"], "path": row["original_path"], "role": "reserved"})
            if rows:
                freeze(partial, {"images": rows, "purpose": "Reserve partially staged images from an interrupted preparation"})
        if partial.exists():
            paths.add(partial)
    return sorted(p.resolve() for p in paths if not p.resolve().is_relative_to(current.resolve()))


def _exclusions(paths):
    ids, hashes, canonical, records = set(), set(), set(), []
    with prefix_paths():
        for path in paths:
            artifact = frozen(path)
            for row in artifact["images"]:
                ids.add(row["id"])
                hashes.add(row["sha256"])
                canonical.add(str(resolved(row["path"])))
            record = {"path": str(path), "sha256": artifact["sha256"], "file_sha256": file_hash(path),
                      "images": len(artifact["images"])}
            if artifact.get("source_lineage_sha256"):
                lineage_path = path.parent.parent / "source-lineage.json"
                lineage = frozen(lineage_path)
                if lineage["sha256"] != artifact["source_lineage_sha256"]:
                    raise ValueError("Prior original-path lineage differs")
                canonical.update(str(resolved(r["original_path"])) for r in lineage["images"])
                record["source_lineage"] = {"path": str(lineage_path), "sha256": lineage["sha256"], "file_sha256": file_hash(lineage_path)}
            records.append(record)
    return ids, hashes, canonical, records


def _verify_selection(context, request):
    selection = frozen(context["fresh"] / "selection-plan.json")
    if selection["request_sha256"] != request["sha256"]:
        raise ValueError("Selection belongs to another candidate request")
    for row in selection["excluded_manifests"]:
        if file_hash(row["path"]) != row["file_sha256"] or frozen(row["path"])["sha256"] != row["sha256"]:
            raise ValueError("An excluded prior collection changed")
        if row.get("source_lineage"):
            item = row["source_lineage"]
            if file_hash(item["path"]) != item["file_sha256"] or frozen(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("An excluded original-path lineage changed")
    manifest = frozen(context["fresh"] / "encoding/manifest.json")
    prepared = frozen(context["fresh"] / "prepared.json")
    if (manifest["selection_plan_sha256"] != selection["sha256"] or prepared["manifest_sha256"] != manifest["sha256"]
            or prepared["request_sha256"] != request["sha256"]):
        raise ValueError("Prepared population binding differs")
    lineage = frozen(context["fresh"] / "source-lineage.json")
    if (lineage["sha256"] != prepared["source_lineage_sha256"] or lineage["sha256"] != manifest["source_lineage_sha256"]
            or lineage["selection_plan_sha256"] != selection["sha256"]
            or [(r["id"], r["sha256"]) for r in lineage["images"]] != [(r["id"], r["sha256"]) for r in manifest["images"]]):
        raise ValueError("Original source lineage changed")
    rows = manifest["images"]
    if (len({r["id"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows)
            or [r["role"] for r in rows] != ["test"] * context["config"]["test_count"] + ["study"] * context["config"]["study_count"] + ["pilot"] * context["config"]["pilot_count"]):
        raise ValueError("Fresh role order/counts or unique identities differ")
    # Revalidate recorded exclusion membership from immutable identity snapshots;
    # no source labels are opened by inference or prediction validation.
    excluded_ids, excluded_hashes = set(), set()
    for record in selection["excluded_manifests"]:
        previous = frozen(record["path"])
        excluded_ids.update(r["id"] for r in previous["images"])
        excluded_hashes.update(r["sha256"] for r in previous["images"])
    if any(r["id"] in excluded_ids or r["sha256"] in excluded_hashes for r in rows):
        raise ValueError("Fresh population overlaps prior development/study/test images")
    excluded_paths = set(selection["excluded_canonical_paths"])
    if any(r["original_path"] in excluded_paths for r in lineage["images"]):
        raise ValueError("Fresh original path overlaps an earlier collection")
    return selection, manifest, prepared


def prepare(workspace, candidate_id):
    require_slurm()
    context = _context(workspace, candidate_id)
    request = _check_request(context, execution=True)
    project = Path(context["config"]["project_root"])
    with locked(project / ".feedback-fresh-selection.lock"):
        if (context["fresh"] / "prepared.json").exists():
            raise FileExistsError("This candidate already has its one frozen fresh cohort")
        prior = discover_manifests(project, context["fresh"])
        if Path(context["config"]["control_directory"]) / "manifest.json" not in prior:
            raise ValueError("The full deployed/development experiment must be excluded")
        old_ids, old_hashes, old_paths, exclusions = _exclusions(prior)
        selection = freeze(context["fresh"] / "selection-plan.json", {
            "request_sha256": request["sha256"], "created_at": now(), "excluded_manifests": exclusions,
            "excluded_canonical_paths": sorted(old_paths),
            "selection": "Uniform seeded permutation of sorted TIFF paths, first eligible unique224x224RGB images; no class balancing",
            "seed": request["seed"], "test_count": context["config"]["test_count"],
            "study_count": context["config"]["study_count"], "pilot_count": context["config"]["pilot_count"],
            "original_path_checks": True, "byte_hash_checks": True,
            "concurrency": "Project-wide exclusive selection lock; completed cohort manifest reserves every chosen image",
        })
        source = Path(context["config"]["source_root"])
        paths = []
        for label in CLASSES:
            with os.scandir(source / label) as entries:
                paths.extend(Path(e.path) for e in entries if Path(e.name).suffix.lower() in (".tif", ".tiff") and e.is_file())
        paths.sort()
        random.Random(request["seed"]).shuffle(paths)
        counts = [context["config"][key] for key in ("test_count", "study_count", "pilot_count")]
        total = sum(counts)
        rows, refs, origins, rejected = [], [], [], 0
        image_dir = context["fresh"] / "encoding/images"
        image_dir.mkdir(parents=True, exist_ok=False)
        with prefix_paths():
            for path in paths:
                if len(rows) == total:
                    break
                original = str(resolved(path))
                if original in old_paths:
                    rejected += 1
                    continue
                data = path.read_bytes()
                import hashlib
                image_hash = hashlib.sha256(data).hexdigest()
                image_id = image_hash[:20]
                if image_hash in old_hashes or image_id in old_ids:
                    rejected += 1
                    continue
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        if image.size != (224, 224) or image.mode != "RGB":
                            raise ValueError("Invalid source geometry")
                except (OSError, ValueError):
                    rejected += 1
                    continue
                role = "test" if len(rows) < counts[0] else "study" if len(rows) < sum(counts[:2]) else "pilot"
                reservation = {"id": image_id, "image_sha256": image_hash, "original_path": original,
                               "selection_plan_sha256": selection["sha256"]}
                # A partial failed preparation still permanently reserves each
                # inspected eligible source path and its bytes for later calls.
                reservation_hash = digest(reservation)
                freeze(context["fresh"] / "reservations" / (reservation_hash[:20] + ".json"), reservation)
                staged = image_dir / (image_hash + ".tif")
                with staged.open("xb") as stream:
                    stream.write(data)
                rows.append({"id": image_id, "path": str(staged), "sha256": image_hash,
                             "size_px": [224, 224], "mode": "RGB", "role": role})
                refs.append({"id": image_id, "role": role, "label": path.parent.name,
                             "source": "NCT-CRC-HE-100K supplied class folder; not expert feedback"})
                origins.append({"id": image_id, "original_path": original, "sha256": image_hash})
                old_ids.add(image_id)
                old_hashes.add(image_hash)
        if len(rows) != total:
            raise ValueError("Insufficient disjoint readable source images")
        lineage = freeze(context["fresh"] / "source-lineage.json", {"selection_plan_sha256": selection["sha256"], "images": origins})
        manifest = freeze(context["fresh"] / "encoding/manifest.json", {
            "schema_version": 1, "selection_plan_sha256": selection["sha256"], "seed": request["seed"],
            "source_file_count": len(paths), "images": rows, "n_test": counts[0], "n_study": counts[1], "n_pilot": counts[2],
            "selection": selection["selection"], "source_lineage_sha256": lineage["sha256"], "staged_bytes_verified": True})
        test_refs = freeze(context["fresh"] / "test-references.json", {"role": "evaluation_only", "manifest_sha256": manifest["sha256"],
                           "labels": [r for r in refs if r["role"] == "test"]})
        study_refs = freeze(context["fresh"] / "study-references.json", {"role": "future_simulated_audit_only", "manifest_sha256": manifest["sha256"],
                            "labels": [r for r in refs if r["role"] != "test"]})
        write(context["fresh"] / "encoding/prompts.json", PROMPTS)
        return freeze(context["fresh"] / "prepared.json", {
            "request_sha256": request["sha256"], "manifest_sha256": manifest["sha256"],
            "selection_plan_sha256": selection["sha256"], "test_references_sha256": test_refs["sha256"],
            "source_lineage_sha256": lineage["sha256"],
            "study_references_sha256": study_refs["sha256"],
            "study_references_file_sha256": file_hash(context["fresh"] / "study-references.json"),
            "selected": len(rows), "rejected": rejected,
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "reference_values_used_for_selection": False})


def encode(workspace, candidate_id):
    require_slurm()
    context = _context(workspace, candidate_id)
    request = _check_request(context, execution=True)
    _verify_selection(context, request)
    from .conch_infer import CHECKPOINT_SHA256, infer, load_encoder
    encoder = load_encoder(context["config"]["checkpoint"], context["config"]["code_directory"], CHECKPOINT_SHA256)
    if encoder_identity(encoder[-1]) != encoder_identity(context["config"]["encoder"]):
        raise ValueError("Inference encoder differs from frozen classifier representation")
    return infer(context["fresh"] / "encoding", encoder)


def _features(context, manifest):
    directory = context["fresh"] / "encoding"
    inference = frozen(directory / "predictions.json")
    provenance = inference["provenance"]
    cache_hash = file_hash(directory / "predictions-embeddings.npz")
    if (provenance["manifest_sha256"] != manifest["sha256"] or provenance["embedding_file_sha256"] != cache_hash
            or provenance["prompts_sha256"] != digest(read(directory / "prompts.json"))
            or encoder_identity(provenance) != encoder_identity(context["config"]["encoder"])):
        raise ValueError("Fresh cache/encoder binding differs")
    ids = [row["id"] for row in manifest["images"]]
    with np.load(directory / "predictions-embeddings.npz", allow_pickle=False) as cache:
        arrays = {key: cache[key].copy() for key in cache.files}
    if (arrays["ids"].tolist() != ids or [r["id"] for r in inference["images"]] != ids
            or arrays["classifier_embeddings"].shape != (len(ids), context["control"]["feature_dimension"])
            or arrays["embeddings"].shape != arrays["classifier_embeddings"].shape
            or arrays["labels"].tolist() != CLASSES
            or arrays["text_embeddings"].shape != (9, context["control"]["feature_dimension"])
            or not np.isfinite(arrays["embeddings"]).all() or not np.isfinite(arrays["text_embeddings"]).all()
            or not np.allclose(np.linalg.norm(arrays["embeddings"], axis=1), 1, atol=1e-5)
            or not np.isfinite(arrays["classifier_embeddings"]).all()
            or not np.allclose(np.linalg.norm(arrays["classifier_embeddings"], axis=1), 1, atol=1e-5)):
        raise ValueError("Fresh feature key/order/normalization differs")
    return inference, arrays, cache_hash


def probabilities(features, coefficients, intercept):
    logits = features @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def gate(candidate, control, development_passed, policy=None):
    policy = policy or {"schema": 1, "accuracy_min_difference": 0, "balanced_min_difference": 0,
                        "maximum_recall_drop": .05, "all_classes_required": True}
    if policy != {"schema": 1, "accuracy_min_difference": 0, "balanced_min_difference": 0,
                  "maximum_recall_drop": .05, "all_classes_required": True}:
        raise ValueError("Unsupported frozen gate schema")
    if (candidate["total"] != control["total"] or candidate["total"] < 1
            or len(candidate["per_class"]) != len(CLASSES) or len(control["per_class"]) != len(CLASSES)
            or any(a["label"] != label or c["label"] != label or a["support"] != c["support"]
                   for label, a, c in zip(CLASSES, candidate["per_class"], control["per_class"], strict=True))):
        raise ValueError("Fresh gate requires identical populations and class order")
    supported = all(a["support"] > 0 and c["support"] > 0 for a, c in zip(candidate["per_class"], control["per_class"], strict=True))
    checks = {"development": bool(development_passed), "all_classes_supported": supported,
              "accuracy_nonregression": candidate["correct"] >= control["correct"],
              "balanced_accuracy_nonregression": candidate["balanced_accuracy"] >= control["balanced_accuracy"],
              "per_class_recall": supported and all(a["recall"] >= c["recall"] - policy["maximum_recall_drop"] - 1e-12
                                                       for a, c in zip(candidate["per_class"], control["per_class"], strict=True))}
    return {"passed": all(checks.values()), "checks": checks,
            "interpretation": "Observed finite-collection non-regression gate; not evidence of clinical or statistically reliable improvement"}


def evaluate(workspace, candidate_id):
    require_slurm()
    context = _context(workspace, candidate_id)
    request = _check_request(context, execution=True)
    _, manifest, prepared = _verify_selection(context, request)
    folder = context["fresh"]
    if (folder / "evaluation.json").exists():
        raise FileExistsError("This candidate's one-shot test has already been evaluated")
    inference, arrays, cache_hash = _features(context, manifest)
    for row in manifest["images"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Fresh image bytes changed after selection")
    candidate_scores = probabilities(arrays["classifier_embeddings"], context["coefficients"], context["intercept"])
    control_scores = probabilities(arrays["classifier_embeddings"], context["control_coefficients"], context["control_intercept"])
    outputs = []
    for row, candidate, control, original in zip(manifest["images"], candidate_scores, control_scores, inference["images"], strict=True):
        outputs.append({"id": row["id"], "role": row["role"],
                        "candidate_label": CLASSES[int(np.argmax(candidate))], "control_label": CLASSES[int(np.argmax(control))],
                        "candidate_probabilities": candidate.tolist(), "control_probabilities": control.tolist(),
                        "cosine_scores": original["cosine_scores"]})
    predictions = freeze(folder / "frozen-classifier-predictions.json", {
        "request_sha256": request["sha256"], "manifest_sha256": manifest["sha256"],
        "candidate_model_sha256": context["model"]["sha256"], "control_classifier_sha256": context["control"]["sha256"],
        "inference_sha256": inference["sha256"], "embedding_file_sha256": cache_hash,
        "classes": CLASSES, "images": outputs, "both_models_frozen_before_reference_read": True})
    # First reference-value access in this evaluation: only the test-role file.
    references = frozen(folder / "test-references.json")
    test_rows = [r for r in outputs if r["role"] == "test"]
    if (references["sha256"] != prepared["test_references_sha256"] or references["manifest_sha256"] != manifest["sha256"]
            or [r["id"] for r in references["labels"]] != [r["id"] for r in test_rows]
            or any(r["role"] != "test" or r["label"] not in CLASSES for r in references["labels"])):
        raise ValueError("Fresh test references differ from the frozen role population")
    truth = [r["label"] for r in references["labels"]]
    candidate = metrics(truth, [r["candidate_label"] for r in test_rows], "Frozen feedback candidate")
    control = metrics(truth, [r["control_label"] for r in test_rows], "Deployed CONCH classifier")
    decision = gate(candidate, control, context["development"]["development_gate_passed"], request["gate_parameters"])
    correct_c = np.array([r["candidate_label"] == y for r, y in zip(test_rows, truth, strict=True)])
    correct_b = np.array([r["control_label"] == y for r, y in zip(test_rows, truth, strict=True)])
    result = freeze(folder / "evaluation.json", {
        "schema": "feedback-fresh-evaluation-v1", "created_at": now(), "request_sha256": request["sha256"],
        "manifest_sha256": manifest["sha256"], "predictions_sha256": predictions["sha256"],
        "references_sha256": references["sha256"], "candidate_model_sha256": context["model"]["sha256"],
        "references_file_sha256": file_hash(folder / "test-references.json"),
        "candidate_coefficients_sha256": context["model"]["coefficients_sha256"],
        "control_classifier_sha256": context["control"]["sha256"], "embedding_file_sha256": cache_hash,
        "candidate": candidate, "control": control, "gate": decision,
        "paired": {"candidate_only_correct": int(np.count_nonzero(correct_c & ~correct_b)),
                   "control_only_correct": int(np.count_nonzero(correct_b & ~correct_c))},
        "candidate_training_labels": context["model"]["training_count"], "control_training_labels": context["control"]["refit_labels"],
        "correction_sources": request["correction_sources"],
        "scope": SCOPE, "study_labels_used_in_evaluation": False, "model_refit": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    if decision["passed"]:
        _build_study(context, request, manifest, prepared, inference, arrays, predictions, result)
    return result


def _build_study(context, request, manifest, prepared, inference, arrays, predictions, result):
    """Stage a complete isolated study only after passing the frozen test gate."""
    folder = context["fresh"] / "new-study"
    keep = [i for i, row in enumerate(manifest["images"]) if row["role"] in ("study", "pilot")]
    rows = [manifest["images"][i] for i in keep]
    new_manifest = freeze(folder / "manifest.json", {"schema_version": 1, "created_at": now(),
        "selection": "Reserved disjoint study/audit-pilot population frozen before fresh evaluation", "seed": request["seed"],
        "n_study": context["config"]["study_count"], "n_pilot": context["config"]["pilot_count"], "images": rows,
        "parent_manifest_sha256": manifest["sha256"], "source_file_count": manifest["source_file_count"]})
    frozen_rows = []
    for i in keep:
        saved = predictions["images"][i]
        scores = saved["candidate_probabilities"]
        frozen_rows.append({"id": saved["id"], "role": saved["role"], "predicted_label": saved["candidate_label"],
                            "classification_scores": dict(zip(CLASSES, scores, strict=True)), "cosine_scores": saved["cosine_scores"],
                            "score_kind": "linear_classifier_probability", "margin": float(np.sort(scores)[-1] - np.sort(scores)[-2])})
    with (folder / "predictions-embeddings.npz").open("xb") as stream:
        np.savez_compressed(stream, ids=arrays["ids"][keep], embeddings=arrays["embeddings"][keep],
                            classifier_embeddings=arrays["classifier_embeddings"][keep], text_embeddings=arrays["text_embeddings"], labels=arrays["labels"])
    provenance = {**inference["provenance"], "manifest_sha256": new_manifest["sha256"],
                  "embedding_file_sha256": file_hash(folder / "predictions-embeddings.npz"),
                  "classifier_sha256": context["model"]["sha256"],
                  "classifier": {"name": "CONCH with feedback classifier", "feature_key": "classifier_embeddings",
                                 "feature_dimension": context["control"]["feature_dimension"],
                                 "representation": context["control"]["representation"]},
                  "feedback_candidate_id": context["id"], "feedback_evaluation_sha256": result["sha256"]}
    pred = freeze(folder / "predictions.json", {"provenance": provenance, "images": frozen_rows})
    write(folder / "prompts.json", PROMPTS)
    # Reserved references remain excluded from fitting and test metrics; this
    # separate artifact supplies only the new study's simulated audit interface.
    refs = frozen(context["fresh"] / "study-references.json")
    if (refs["sha256"] != prepared["study_references_sha256"] or refs["manifest_sha256"] != manifest["sha256"]
            or [r["id"] for r in refs["labels"]] != [r["id"] for r in rows]
            or [r["role"] for r in refs["labels"]] != [r["role"] for r in rows]):
        raise ValueError("Reserved study references differ")
    write(folder / "references.json", {"purpose": "Separate simulated study/audit pilot; never fitted or used for fresh test", "labels": refs["labels"]})
    config = {"schema_version": 1, "title": "CRC tissue study — feedback candidate", "created_at": now(),
              "manifest_sha256": new_manifest["sha256"], "classes": CLASSES, "endpoints": ENDPOINTS,
              "target_population": f"The {context['config']['study_count']} reserved tiles in this isolated new study",
              "measurement": {"numerator": "Tiles carrying each specified tissue class", "denominator": "All study tiles including BACK/DEB",
                              "aggregation": "Equal weight per tile", "eligible_tissue": "Entire tile", "physical_scale": "No physical area estimated", "exclusions": "All prior image manifests and the disjoint fresh test"},
              "scope": "Finite selected tiles; no spatial, patient-level or clinical inference", "split": "Train/validation, fresh test, study and audit pilot are separate roles",
              "default_audit_budget": min(120, context["config"]["study_count"]), "familywise_alpha": .05,
              "pilot_labels": context["config"]["pilot_count"], "classifier_sha256": context["model"]["sha256"],
              "feedback_label_sources": request["correction_sources"],
              "reference_limit": "Dataset-supplied labels simulate review; expert feedback source is recorded separately"}
    write(folder / "study.json", config)
    # Atlas uses only projected image features, never labels.
    x = arrays["embeddings"][keep].astype(float)
    u, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    coordinates = u[:, :2] * s[:2]
    write(folder / "atlas.json", {"method": "PCA of CONCH projected retrieval features; similarity is not tissue adjacency",
        "explained_variance_fraction": (s[:2]**2 / np.sum(s**2)).tolist(),
        "points": {row["id"]: point.tolist() for row, point in zip(rows, coordinates, strict=True)}})
    aggregate = {"schema_version": 1, "classifier_sha256": context["model"]["sha256"], "evaluations": [{
        "id": "feedback-fresh-" + context["id"], "name": "Fresh same-source NCT test", "images": result["candidate"]["total"],
        "classifier_sha256": context["model"]["sha256"], "evaluation_sha256": result["sha256"],
        "accuracy": result["candidate"]["accuracy"], "balanced_accuracy": result["candidate"]["balanced_accuracy"],
        "macro_f1": result["candidate"]["macro_f1"], "correct": result["candidate"]["correct"], "total": result["candidate"]["total"],
        "per_class": result["candidate"]["per_class"], "confusion_matrix": result["candidate"]["confusion_matrix"],
        "scope": SCOPE, "target_population": f"Separate {result['candidate']['total']}-tile finite fresh test", "already_inspected": False}]}
    freeze(folder / "test-evaluations.json", aggregate)
    from .study import Study
    Study(folder)
    names = ("manifest.json", "predictions.json", "predictions-embeddings.npz", "prompts.json", "references.json", "study.json", "atlas.json", "test-evaluations.json")
    freeze(context["fresh"] / "ready-study.json", {"request_sha256": request["sha256"], "evaluation_sha256": result["sha256"],
        "candidate_model_sha256": context["model"]["sha256"], "study_prediction_sha256": pred["sha256"],
        "files": {name: file_hash(folder / name) for name in names}, "active_study_changed": False})


def _verified_evaluation(context, request):
    result = frozen(context["fresh"] / "evaluation.json")
    pred = frozen(context["fresh"] / "frozen-classifier-predictions.json")
    _, manifest, prepared = _verify_selection(context, request)
    inference, _, cache_hash = _features(context, manifest)
    if (result["request_sha256"] != request["sha256"] or result["candidate_model_sha256"] != context["model"]["sha256"]
            or result["candidate_coefficients_sha256"] != context["model"]["coefficients_sha256"]
            or result["control_classifier_sha256"] != context["control"]["sha256"]
            or result["predictions_sha256"] != pred["sha256"] or pred["request_sha256"] != request["sha256"]
            or result["manifest_sha256"] != manifest["sha256"] or pred["manifest_sha256"] != manifest["sha256"]
            or result["references_sha256"] != prepared["test_references_sha256"]
            or file_hash(context["fresh"] / "test-references.json") != result["references_file_sha256"]
            or file_hash(context["fresh"] / "study-references.json") != prepared["study_references_file_sha256"]
            or pred["candidate_model_sha256"] != context["model"]["sha256"]
            or pred["control_classifier_sha256"] != context["control"]["sha256"]
            or pred["inference_sha256"] != inference["sha256"]
            or pred["embedding_file_sha256"] != cache_hash or result["embedding_file_sha256"] != cache_hash
            or result["gate"] != gate(result["candidate"], result["control"], context["development"]["development_gate_passed"], request["gate_parameters"])):
        raise ValueError("Fresh evaluation does not bind this exact candidate and protocol")
    return result


def eligibility(workspace, candidate_id):
    """Only sanitized summaries are returned; no source image paths/labels."""
    value = {"available": False, "eligible_for_new_study": False, "status": "not_configured",
             "reason": "Trusted fresh-evaluation configuration is not installed"}
    identifier(candidate_id)
    workspace = Path(workspace).resolve()
    if not (workspace / "fresh-config.json").exists():
        return value
    value["available"] = True
    if not (workspace / "candidates" / candidate_id / "result.json").exists():
        return dict(value, status="awaiting_training", reason="Complete candidate development evaluation first")
    context = _context(workspace, candidate_id)
    if not context["development"]["development_gate_passed"]:
        return dict(value, status="development_gate_failed", reason="Candidate failed the reused-development gate; fresh test was not consumed")
    if not (context["fresh"] / "request.json").exists():
        return dict(value, status="ready_to_request", reason="One fresh same-source evaluation can be queued for this frozen candidate")
    request = _check_request(context)
    if (context["fresh"] / "failure.json").exists():
        failure = frozen(context["fresh"] / "failure.json")
        if failure["request_sha256"] != request["sha256"]:
            raise ValueError("Failure belongs to another fresh request")
        return dict(value, status="failed", reason="Fresh workflow failed at " + failure["stage"] + "; frozen artifacts retained")
    if not (context["fresh"] / "evaluation.json").exists():
        status = "encoding" if (context["fresh"] / "prepared.json").exists() else "queued"
        jobs = frozen(context["fresh"] / "jobs.json") if (context["fresh"] / "jobs.json").exists() else None
        if jobs and jobs["request_sha256"] != request["sha256"]:
            raise ValueError("Job registry belongs to another request")
        return dict(value, status=status, jobs=jobs["jobs"] if jobs else {}, reason="Scheduled fresh evaluation is pending")
    result = _verified_evaluation(context, request)
    public = {key: result[key] for key in ("candidate", "control", "gate", "paired", "scope", "candidate_training_labels", "control_training_labels", "correction_sources")}
    public["sha256"] = result["sha256"]
    if not result["gate"]["passed"]:
        return dict(value, status="fresh_gate_failed", evaluation=public,
                    reason="Fresh non-regression gate failed; this candidate cannot start a new study")
    if not (context["fresh"] / "ready-study.json").exists():
        return dict(value, status="building_study", evaluation=public, reason="Gate passed; isolated study artifacts are still being validated")
    _verify_ready(context, request, result)
    return dict(value, status="eligible", eligible_for_new_study=True, evaluation=public,
                reason="Frozen finite same-source gate passed; an isolated new study is available without replacing the active model")


def _verify_ready(context, request, result):
    ready = frozen(context["fresh"] / "ready-study.json")
    if (ready["request_sha256"] != request["sha256"] or ready["evaluation_sha256"] != result["sha256"]
            or ready["candidate_model_sha256"] != context["model"]["sha256"] or not result["gate"]["passed"]):
        raise ValueError("Staged study belongs to another candidate/evaluation")
    if set(ready["files"]) != {"manifest.json", "predictions.json", "predictions-embeddings.npz", "prompts.json", "references.json", "study.json", "atlas.json", "test-evaluations.json"}:
        raise ValueError("Staged study required artifact set differs")
    for name, expected in ready["files"].items():
        if Path(name).name != name or file_hash(context["fresh"] / "new-study" / name) != expected:
            raise ValueError("Staged study artifact changed")
    return ready


def create_study(workspace, candidate_id):
    if not eligibility(workspace, candidate_id)["eligible_for_new_study"]:
        raise ValueError("Candidate is not eligible for a new study")
    context = _context(workspace, candidate_id)
    request = _check_request(context)
    result = _verified_evaluation(context, request)
    ready = _verify_ready(context, request, result)
    study_id = candidate_id  # One immutable published study per frozen candidate.
    path = context["workspace"] / "studies" / (study_id + ".json")
    with locked(context["workspace"] / ".fresh.lock"):
        if not path.exists():
            freeze(path, {"id": study_id, "candidate_id": candidate_id, "request_sha256": request["sha256"],
                          "ready_study_sha256": ready["sha256"], "evaluation_sha256": result["sha256"], "created_at": now()})
    resolve_study(workspace, study_id)
    return {"id": study_id, "url": f"feedback-studies/{study_id}/", "candidate_id": candidate_id,
            "scope": SCOPE, "active_study_changed": False}


def resolve_study(workspace, study_id):
    study_id = identifier(study_id)
    record = frozen(Path(workspace) / "studies" / (study_id + ".json"))
    if record["id"] != study_id or record["candidate_id"] != study_id:
        raise ValueError("Invalid published study ID")
    context = _context(workspace, study_id)
    request = _check_request(context)
    result = _verified_evaluation(context, request)
    ready = _verify_ready(context, request, result)
    if (record["request_sha256"] != request["sha256"] or record["ready_study_sha256"] != ready["sha256"]
            or record["evaluation_sha256"] != result["sha256"]):
        raise ValueError("Published study registry binding changed")
    return context["fresh"] / "new-study"


def list_studies(workspace):
    ids = []
    for path in sorted((Path(workspace) / "studies").glob("*.json")):
        resolve_study(workspace, path.stem)
        ids.append(path.stem)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "prepare", "encode", "evaluate"))
    parser.add_argument("workspace")
    parser.add_argument("candidate_id", nargs="?")
    args = parser.parse_args()
    if args.command == "configure":
        print(configure(args.workspace), flush=True)
        return
    try:
        result = {"prepare": prepare, "encode": encode, "evaluate": evaluate}[args.command](args.workspace, args.candidate_id)
        print({"stage": args.command, "sha256": result.get("sha256")}, flush=True)
    except Exception as error:
        folder = Path(args.workspace) / "candidates" / identifier(args.candidate_id) / "fresh"
        # Rejecting a duplicate command must not turn a completed immutable
        # evaluation into a failed candidate. Preserve its original outcome.
        if (folder / "request.json").exists() and not (folder / "failure.json").exists() and not (folder / "evaluation.json").exists():
            freeze(folder / "failure.json", {"request_sha256": frozen(folder / "request.json")["sha256"],
                                             "stage": args.command, "reason": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
