"""Frozen sources and native-pixel sampling frames for the new spatial study."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from .spatial import check_file, components_from_mask, file_sha256, image_windows, seal, verify
from .storage import now, read, write

ENDPOINT_DEFINITIONS = [
    {"id": "epithelial_fraction", "name": "Epithelial area fraction",
     "numerator_definition": "Native pixels labeled TUMOR (1) or NORMAL epithelium (4).",
     "numerator_codes": [1, 4]},
    {"id": "dense_lymphocytic_fraction", "name": "Dense lymphocytic tissue area fraction",
     "numerator_definition": "Native pixels labeled TILS / dense lymphocytic tissue (3); this is tissue area, not a lymphocyte count.",
     "numerator_codes": [3]},
]
for _definition in ENDPOINT_DEFINITIONS:
    _definition.update({
        "denominator_definition": "Native pixels labeled epithelium, stroma, or dense lymphocytic tissue (codes 1, 2, 3, 4).",
        "denominator_codes": [1, 2, 3, 4],
        "eligible_tissue": "The manual field of view in each selected image, restricted to codes 1–4.",
        "excluded_regions": "EXCLUDE (0), JUNK (5), BLOOD (6), OTHER (7), WHITE (8). Predictions and references each supply their own numerator and denominator.",
        "physical_scale": "Native-pixel area fractions; physical spacing is unverified for this new collection. No square-millimeter, density, or distance claim.",
        "aggregation": "Sum numerator pixels across all final fields, divided by sum eligible pixels; not an average of field fractions.",
        "boundary_rule": "Half-open manual FOV [left,right) × [top,bottom), partitioned into disjoint 64-pixel windows including clipped edges. Context is display only.",
        "zero_denominator": "Undefined if the corrected eligible total is nonpositive; preserve and flag invalid raw estimates.",
        "interpretation": "Finite collection only; selected fields do not represent whole slides, patients, or a clinical population.",
        "normalization_scale": 0.05,
    })


def prepare_inference(curation, output):
    """Freeze the already selected nominal predictor before opening new references."""
    curation, output = Path(curation).absolute(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    source = Path(__file__).parent.parent / "scripts" / "mutils_smoke.py"
    old = Path(os.environ.get("TISSUE_SPATIAL_MODEL_ROOT", "./artifacts/spatial-model")).resolve()
    plan_path = curation / "collection-plan.json"
    plan = seal({
        "schema": "spatial-research-inference-v1", "created_at": now(),
        "curation_dir": str(curation), "collection_plan_sha256": file_sha256(plan_path),
        "output_dir": str(output), "adapter": str(source), "adapter_sha256": file_sha256(source),
        "code_dir": str(old / "vendor" / "MuTILs_Panoptic"),
        "checkpoint": str(old / "checkpoints" / "mutils_06022021_fold2.pt"),
        "roles": {"pilot": 6, "final": 32},
        "frozen_predictor": "Released MuTILs fold 2, strict float32 weights; original nominal full-ROI adapter: 1024 RGB → 512 bilinear → 256 Lanczos, no stain normalization or augmentation, one full-field HPF.",
        "rationale": "Keep the existing frozen spatial predictor. No model selection or fitting on the new final fields; pilot labels inform allocation only.",
        "physical_scale_verified": False,
        "endpoints": ENDPOINT_DEFINITIONS,
        "separation": "Three pilot slides and sixteen final slides are disjoint from each other and all five previously inspected slides; checkpoint fold-2 train slides/hospitals excluded.",
    })
    write(output / "inference-plan.json", plan)
    return plan


def infer(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference must run in a Slurm GPU job")
    output = Path(output)
    plan = read(output / "inference-plan.json")
    verify(plan)
    check_file(plan["adapter"], plan["adapter_sha256"])
    check_file(Path(plan["curation_dir"]) / "collection-plan.json", plan["collection_plan_sha256"])
    for role, count in plan["roles"].items():
        destination = output / role
        if (destination / "frozen-predictions.json").exists():
            raise FileExistsError("Completed inference is immutable")
        subprocess.run([sys.executable, plan["adapter"], "infer", "--data-dir",
                        str(Path(plan["curation_dir"]) / "data" / role), "--code-dir", plan["code_dir"],
                        "--checkpoint", plan["checkpoint"], "--output-dir", str(destination),
                        "--limit", str(count)], check=True)


def prepare_frame(inference, destination, role):
    """Build the frame from geometry and predictions without decoding reference masks."""
    inference, destination = Path(inference), Path(destination)
    source_plan = read(inference / "frozen-plan.json")
    frozen = read(inference / "frozen-predictions.json")
    check_file(inference / "frozen-plan.json", frozen["plan_sha256"])
    outputs = {r["roi_id"]: r for r in frozen["predictions"]}
    rois, units, predictions, references = [], [], [], []
    data = Path(source_plan["source_data_dir"])
    for sample in source_plan["samples"]:
        identifier = sample["roi_id"]
        prediction = outputs[identifier]
        rgb = sample["files"]["rgbs"]
        ref = sample["files"]["masks"]
        image_path, ref_path = data / "rgbs" / rgb["name"], data / "masks" / ref["name"]
        check_file(image_path, rgb["sha256"])
        check_file(prediction["region_png"], prediction["region_png_sha256"])
        with Image.open(image_path) as image:
            size = image.size
        with Image.open(prediction["region_png"]) as image:
            mask = np.array(image)
        if size != (1024, 1024) or mask.shape != (1024, 1024):
            raise ValueError("Unexpected source/prediction geometry")
        fov = {key: int(sample["manual_fov"][key]) for key in ("left", "top", "right", "bottom")}
        roi = {"id": identifier, "slide_id": sample["slide"], "image_path": str(image_path),
               "image_sha256": rgb["sha256"], "size_px": list(size), "manual_fov": fov,
               "prediction_mask_path": prediction["region_png"],
               "prediction_mask_sha256": prediction["region_png_sha256"], "role": role}
        rois.append(roi)
        for unit in image_windows(identifier, fov, size, 64, 64):
            unit["role"] = role
            unit["slide_id"] = sample["slide"]
            b = unit["bounds"]
            components = components_from_mask(mask[b["top"]:b["bottom"], b["left"]:b["right"]])
            units.append(unit)
            predictions.append({"id": unit["id"], "components": components})
        references.append({"roi_id": identifier, "path": str(ref_path), "sha256": ref["sha256"],
                           "channel": 0, "source": "Manual pathologist-approved PanopTILs region annotations"})
    frame = seal({"schema": "research-image-frame-v1", "role": role, "rois": rois, "units": units,
                  "window_side_pixels": 64, "context_pixels": 64,
                  "target": "Finite selected manual FOV collection; native-pixel area fractions",
                  "total_pixels": sum(row["area_pixels"] for row in units)})
    pred = seal({"schema": "research-predictions-v1", "frame_sha256": frame["sha256"],
                 "source_prediction_sha256": file_sha256(inference / "frozen-predictions.json"),
                 "source_plan_sha256": frozen["plan_sha256"],
                 "components": ["epithelial", "stromal", "dense_lymphocytic_tissue", "eligible"], "units": predictions})
    refs = seal({"schema": "research-reference-sources-v1", "frame_sha256": frame["sha256"],
                 "rois": references, "reference_masks_decoded_by_frame_builder": False})
    config = seal({"schema": "research-source-study-v1", "id": f"panoptils-{role}-001",
                   "name": "Breast tissue area study" if role == "final" else "Separate allocation pilot",
                   "frame_sha256": frame["sha256"], "prediction_sha256": pred["sha256"],
                   "reference_sources_sha256": refs["sha256"], "endpoint_definitions": ENDPOINT_DEFINITIONS,
                   "role": role, "created_at": now(), "annotation_mode": "Simulated review of released manual reference masks",
                   "reference_policy": "Final references remain hidden from predictor and allocation. Separate evaluation may reveal truth after design freeze.",
                   "model": "Frozen MuTILs fold 2; nominal image preprocessing; physical scale unverified",
                   "scope": "Finite selected image fields; no patient generalization or clinical validation"})
    for name, value in [("frame", frame), ("predictions", pred), ("reference-sources", refs), ("study", config)]:
        write(destination / f"{name}.json", value)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "infer", "frame"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--curation")
    parser.add_argument("--inference")
    parser.add_argument("--role", choices=["pilot", "final"])
    args = parser.parse_args()
    if args.command == "freeze":
        result = prepare_inference(args.curation, args.output)
    elif args.command == "infer":
        result = infer(args.output)
    else:
        result = prepare_frame(args.inference, args.output, args.role)
    print(result)


if __name__ == "__main__":
    main()
