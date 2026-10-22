#!/usr/bin/env python3
"""Pinned, full-ROI MuTILs smoke inference; separately reveal manual FOV labels.

This deliberately uses only the released network, not the whole-slide runner.
No physical calibration, nucleus object extraction, or clinical validation is claimed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE_REVISION = "038474be875e3d34b8b6821b7e8fc5d3f79f7523"
WEIGHTS_REVISION = "60730e8baba6ba5adf2f3db84e55c0edbd5c1d12"
WEIGHTS_SHA256 = "a3c7fcbdc2ada7ed2b1fdaeb681b55ec65a2a4c21a4676bbd38f8e815c5ba63e"
WEIGHTS_BYTES = 250392523
FOLD_SHA256 = {
    "fold_2_train.csv": "805aa5ef143d4492ee82820ec13c6d038ddfd07391555b757b03f4d512616027",
    "fold_2_test.csv": "7ce5d4b82afa2ce400aaa7440ff2d8212abb579ac00210f1fbb7def9b67e906d",
}
REGION_CODES = {"EXCLUDE": 0, "TUMOR": 1, "STROMA": 2, "TILS": 3,
                "NORMAL": 4, "JUNK": 5, "BLOOD": 6, "OTHER": 7, "WHITE": 8}
ENDPOINTS = {
    "epithelial_fraction": {"numerator_codes": [1, 4], "denominator_codes": [1, 2, 3, 4]},
    "stromal_fraction": {"numerator_codes": [2, 3], "denominator_codes": [1, 2, 3, 4]},
    "dense_lymphocytic_tissue_fraction": {"numerator_codes": [3], "denominator_codes": [1, 2, 3, 4]},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Exclusive creation preserves completed or partially completed experiments.
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_file(path, expected_sha256, expected_size=None):
    path = Path(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path}")
    if sha256(path) != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {path}")


def check_inputs(data_dir, limit):
    manifest = json.loads((data_dir / "sample-manifest.json").read_text())
    samples = manifest["samples"][:limit]
    if not samples:
        raise ValueError("No samples requested")
    folds = {}
    for name, expected in FOLD_SHA256.items():
        check_file(data_dir / name, expected)
        with (data_dir / name).open() as stream:
            folds[name] = list(csv.DictReader(stream))
    train_slides = {r["slide_name"] for r in folds["fold_2_train.csv"]}
    test_slides = {r["slide_name"] for r in folds["fold_2_test.csv"]}
    train_hospitals = {r["hospital"] for r in folds["fold_2_train.csv"]}
    test_hospitals = {r["hospital"] for r in folds["fold_2_test.csv"]}
    if train_slides & test_slides or train_hospitals & test_hospitals:
        raise ValueError("Released fold has train/test slide or hospital overlap")
    for sample in samples:
        if sample["slide"] not in test_slides or sample["slide"] in train_slides:
            raise ValueError("Sample is not held out for checkpoint fold 2")
        for kind, info in sample["files"].items():
            check_file(data_dir / kind / info["name"], info["sha256"], info["size"])
        fov = sample["manual_fov"]
        left, right, top, bottom = (int(fov[key]) for key in ("left", "right", "top", "bottom"))
        if not (0 <= left < right <= 1024 and 0 <= top < bottom <= 1024):
            raise ValueError("Invalid native-pixel FOV bounds")
    return manifest, samples


def infer(args):
    import numpy as np
    from PIL import Image
    import torch
    import torchvision

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Inference requires a Slurm allocation with CUDA")
    data_dir, code_dir, checkpoint, output_dir = map(
        lambda p: Path(p).resolve(), (args.data_dir, args.code_dir, args.checkpoint, args.output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {output_dir}")
    revision = subprocess.check_output(["git", "-C", str(code_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != CODE_REVISION:
        raise ValueError(f"Unpinned MuTILs source: {revision}")
    subprocess.run(["git", "-C", str(code_dir), "diff", "--exit-code", "HEAD", "--"], check=True)
    check_file(checkpoint, WEIGHTS_SHA256, WEIGHTS_BYTES)
    manifest, samples = check_inputs(data_dir, args.limit)
    sys.path[:0] = [str(code_dir.parent), str(code_dir / "mutils_panoptic")]
    from MuTILs_Panoptic.mutils_panoptic.MuTILs import MuTILs
    from MuTILs_Panoptic.configs.panoptic_model_configs import MuTILsParams, RegionCellCombination
    if RegionCellCombination.REGION_CODES != REGION_CODES:
        raise ValueError("Region label schema differs from frozen endpoint definitions")
    params = copy.deepcopy(MuTILsParams.model_params)
    if (params["hpf_mpp"], params["roi_mpp"], params["roi_side"]) != (0.5, 1.0, 256):
        raise ValueError("Unexpected released geometry configuration")
    params.update(training=False, roi_side=256, hpf_side=512, topk_hpf=1, random_topk_hpf=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "mutils-spatial-smoke-v1", "created_at_unix": time.time(),
        "purpose": "Engineering smoke with held-out manual ROIs; not a representative cohort or clinical validation",
        "source_data_dir": str(data_dir), "sample_manifest_sha256": sha256(data_dir / "sample-manifest.json"),
        "selection": manifest["selection"], "samples": samples,
        "code_revision": revision, "adapter_sha256": sha256(__file__),
        "checkpoint": {"path": str(checkpoint), "sha256": WEIGHTS_SHA256, "bytes": WEIGHTS_BYTES,
                       "repository": "mutils-panoptic/mutils", "revision": WEIGHTS_REVISION, "fold": 2},
        "folds": {"sha256": FOLD_SHA256, "held_out_slide_and_hospital_verified": True},
        "model_params": params, "region_codes": REGION_CODES,
        "preprocessing": {
            "input": "native RGB 1024x1024 uint8",
            "highres": "PIL BILINEAR resize to 512x512; adapter's explicit native-to-network resize",
            "lowres": "PIL LANCZOS resize highres to 256x256, matching official runner _prep_batchdata",
            "tensor": "float32 CHW / 255; official network applies ImageNet mean/std internally",
            "stain_normalization": False, "test_time_augmentation": False,
            "roi_mode": "Complete ROI via official runner _fix_mutils_params geometry, one HPF",
            "nominal_config_mpp": {"original": 0.25, "highres": 0.5, "lowres": 1.0},
            "physical_scale_verified": False,
            "scale_caveat": "Release/config nominal scale conflicts with paper wording; PNG has no calibration. Filename WSI coordinates are not native PNG pixels.",
            "native_mask_mapping": "argmax of official hpf_region_logits + 1, then nearest-neighbor 512 to 1024; official roi_align uses its default aligned=False",
        },
        "endpoints": ENDPOINTS,
        "measurement_rules": {
            "units": "native-pixel area fractions only; no mm2, density, distance, or cell counts",
            "window": "manual FOV CSV coordinates interpreted as [left,right) x [top,bottom), in native 1024 pixel coordinates",
            "fov_boundary_caveat": "Half-open convention is explicit; release CSV does not specify inclusive/exclusive right/bottom",
            "eligible": "TUMOR, NORMAL, STROMA, TILS; exclude EXCLUDE, JUNK, BLOOD, OTHER, WHITE",
            "epithelium": "TUMOR plus NORMAL; no benign/malignant discrimination claim",
            "stroma": "STROMA plus TILS (dense lymphocytic tissue label, not lymphocyte counts)",
            "dependence": "Epithelial and stromal fractions are complementary under this eligibility rule; dense-lymphocytic tissue is a subset of stroma",
            "aggregation": "sum numerator pixels / sum denominator pixels over selected FOVs; never mean of FOV ratios",
            "zero_denominator": "null estimate; report component counts",
            "prediction_vs_reference": "Each mask supplies its own joint numerator and denominator; retain covariance-ready per-FOV component totals",
        },
        "runtime": {"slurm_job_id": os.environ["SLURM_JOB_ID"], "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "python": sys.version},
    }
    write_json(output_dir / "frozen-plan.json", plan)
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = MuTILs(**copy.deepcopy(params))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint incompatibility: {incompatible}")
    del state
    model.requires_grad_(False).to("cuda").eval()
    rows = []
    for sample in samples:
        started = time.monotonic()
        with Image.open(data_dir / "rgbs" / sample["files"]["rgbs"]["name"]) as opened:
            if opened.mode != "RGB" or opened.size != (1024, 1024):
                raise ValueError("Unexpected ROI image geometry or channels")
            highres = opened.resize((512, 512), Image.Resampling.BILINEAR)
        lowres = highres.resize((256, 256), Image.Resampling.LANCZOS)
        def tensor(image):
            return torch.as_tensor(np.array(image)).permute(2, 0, 1).float().div(255)[None].to("cuda")
        with torch.inference_mode():
            result = model([{"highres_rgb": tensor(highres), "lowres_rgb": tensor(lowres)}])
        for key, shape in {"roi_region_logits": (1, 8, 256, 256),
                           "hpf_region_logits": (1, 8, 512, 512),
                           "hpf_nuclei": (1, 9, 512, 512)}.items():
            if tuple(result[key].shape) != shape or not torch.isfinite(result[key]).all().item():
                raise ValueError(f"Invalid output: {key}")
        for key, expected in {"hpf_lres_bounds": [[0, 0, 256, 256]],
                              "hpf_hres_bounds": [[0, 0, 512, 512]]}.items():
            if not np.array_equal(result[key].cpu().numpy(), expected):
                raise ValueError(f"Unexpected full-ROI bounds: {key}={result[key]}")
        region_512 = (result["hpf_region_logits"][0].argmax(0).cpu().numpy() + 1).astype(np.uint8)
        region_native = np.repeat(np.repeat(region_512, 2, axis=0), 2, axis=1)
        prefix = sample["roi_id"]
        npz_path = output_dir / f"{prefix}.npz"
        np.savez_compressed(npz_path, region_native=region_native, region_512=region_512,
                            roi_region_logits=result["roi_region_logits"][0].cpu().numpy(),
                            hpf_region_logits=result["hpf_region_logits"][0].cpu().numpy(),
                            # Semantic pixels only: explicitly not object detections.
                            nuclear_semantic_512=(result["hpf_nuclei"][0].argmax(0).cpu().numpy() + 1).astype(np.uint8))
        png_path = output_dir / f"{prefix}.regions.png"
        Image.fromarray(region_native).save(png_path)
        rows.append({"roi_id": prefix, "seconds": time.monotonic() - started,
                     "arrays": str(npz_path), "arrays_sha256": sha256(npz_path),
                     "region_png": str(png_path), "region_png_sha256": sha256(png_path),
                     "full_roi_bounds_verified": True, "output_shapes_finite_verified": True})
        print(json.dumps({"predicted": prefix, "seconds": rows[-1]["seconds"]}), flush=True)
        del result
    frozen = {"schema": "mutils-frozen-predictions-v1", "plan_sha256": sha256(output_dir / "frozen-plan.json"),
              "strict_state_dict": True, "missing_keys": [], "unexpected_keys": [], "predictions": rows,
              "reference_masks_opened_by_inference": False, "frozen_at_unix": time.time()}
    write_json(output_dir / "frozen-predictions.json", frozen)
    if args.evaluate:
        evaluate(output_dir)
    return frozen


def evaluate(output_dir):
    import numpy as np
    from PIL import Image
    output_dir = Path(output_dir).resolve()
    frozen = json.loads((output_dir / "frozen-predictions.json").read_text())
    check_file(output_dir / "frozen-plan.json", frozen["plan_sha256"])
    plan = json.loads((output_dir / "frozen-plan.json").read_text())
    data_dir = Path(plan["source_data_dir"])
    def components(labels):
        counts = np.bincount(labels.ravel(), minlength=9)
        return {name: {"numerator": int(counts[spec["numerator_codes"]].sum()),
                       "denominator": int(counts[spec["denominator_codes"]].sum())}
                for name, spec in plan["endpoints"].items()}, counts.tolist()
    rows = []
    confusion_total = np.zeros((9, 9), dtype=np.int64)
    for sample, artifact in zip(plan["samples"], frozen["predictions"], strict=True):
        if sample["roi_id"] != artifact["roi_id"]:
            raise ValueError("Prediction order mismatch")
        check_file(artifact["arrays"], artifact["arrays_sha256"])
        mask_info = sample["files"]["masks"]
        mask_path = data_dir / "masks" / mask_info["name"]
        check_file(mask_path, mask_info["sha256"], mask_info["size"])
        with Image.open(mask_path) as image:
            manual = np.array(image)
        if manual.shape != (1024, 1024, 3):
            raise ValueError("Unexpected manual mask geometry")
        with np.load(artifact["arrays"], allow_pickle=False) as cache:
            pred = cache["region_native"]
        fov = sample["manual_fov"]
        box = (slice(int(fov["top"]), int(fov["bottom"])), slice(int(fov["left"]), int(fov["right"])))
        ref, pred = manual[..., 0][box], pred[box]
        if np.any(ref > 8) or np.any(pred > 8):
            raise ValueError("Invalid region label code")
        ref_components, ref_counts = components(ref)
        pred_components, pred_counts = components(pred)
        valid = ref != 0
        confusion = np.bincount((9 * ref[valid].astype(int) + pred[valid]).ravel(), minlength=81).reshape(9, 9)
        confusion_total += confusion
        rows.append({"roi_id": sample["roi_id"], "native_fov_bounds": {k: int(fov[k]) for k in ("left", "top", "right", "bottom")},
                     "fov_pixels": int(ref.size), "predicted_components": pred_components,
                     "reference_components": ref_components, "predicted_class_pixels": pred_counts,
                     "reference_class_pixels": ref_counts, "confusion_true_rows_pred_columns": confusion.tolist()})
    totals = {}
    for name in plan["endpoints"]:
        totals[name] = {}
        for kind in ("predicted", "reference"):
            numerator = sum(row[f"{kind}_components"][name]["numerator"] for row in rows)
            denominator = sum(row[f"{kind}_components"][name]["denominator"] for row in rows)
            totals[name][kind] = {"numerator": numerator, "denominator": denominator,
                                  "fraction": numerator / denominator if denominator else None}
    report = {"schema": "mutils-smoke-reference-evaluation-v1", "evaluated_at_unix": time.time(),
              "frozen_predictions_sha256": sha256(output_dir / "frozen-predictions.json"),
              "scope": "Exact descriptive comparison on selected FOVs; no sampling interval or patient generalization",
              "physical_scale_verified": False, "reference": "Manual PanopTILs released region mask channel 0",
              "preprocessing_scope": "No stain normalization; engineering smoke, not reproduction of full MuTILs pipeline",
              "per_fov": rows, "aggregate_components": totals,
              "confusion_true_rows_pred_columns": confusion_total.tolist(),
              "nonexcluded_pixel_agreement": float(np.trace(confusion_total) / confusion_total.sum()) if confusion_total.sum() else None}
    write_json(output_dir / "reference-evaluation.json", report)
    print(json.dumps({"evaluation": str(output_dir / "reference-evaluation.json"), "roi_count": len(rows),
                      "aggregate_components": totals}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("infer")
    predict.add_argument("--data-dir", required=True)
    predict.add_argument("--code-dir", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument("--limit", type=int, default=1)
    predict.add_argument("--evaluate", action="store_true")
    reveal = commands.add_parser("evaluate")
    reveal.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "infer":
        if args.limit <= 0:
            parser.error("--limit must be positive")
        infer(args)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
