"""Freeze a label-unbalanced image collection and a disjoint allocation pilot.

Lightweight metadata and limited image I/O; no model loading. Labels never
enter sample selection. Only the reference artifact contains the class labels.
"""
import argparse
import hashlib
import random
from pathlib import Path

from PIL import Image

from .storage import digest, now, write

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
ENDPOINTS = ["TUM", "STR", "LYM"]
PROMPTS = {
    "ADI": "an H&E image of adipose tissue",
    "BACK": "an H&E image of background with no tissue",
    "DEB": "an H&E image of debris and necrosis",
    "LYM": "an H&E image of lymphocytes",
    "MUC": "an H&E image of mucus",
    "MUS": "an H&E image of smooth muscle",
    "NORM": "an H&E image of normal colon mucosa",
    "STR": "an H&E image of cancer-associated stroma",
    "TUM": "an H&E image of colorectal adenocarcinoma epithelium",
}


def prepare(source, output, n=600, pilot=90, seed=20261017):
    source, output = Path(source).resolve(), Path(output).resolve()
    if n < 1 or pilot < 0:
        raise ValueError("A positive study size and nonnegative pilot size are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new study directory to preserve frozen artifacts")
    paths = sorted(p for label in CLASSES for p in (source / label).glob("*.tif"))
    if len(paths) < n + pilot:
        raise ValueError("Insufficient source tiles")
    random.Random(seed).shuffle(paths)
    rows, references, exclusions, hashes = [], [], [], set()
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if len(rows) == n + pilot:
            break
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if image_hash in hashes:
            exclusions.append({"image_sha256": image_hash, "reason": "duplicate image bytes"})
            continue
        try:
            with Image.open(path) as image:
                image.load()
                size, mode = list(image.size), image.mode
                if size != [224, 224] or mode != "RGB":
                    raise ValueError(f"Unexpected native dimensions/mode {size}/{mode}")
        except Exception as error:
            exclusions.append({"path": str(path), "reason": str(error)})
            continue
        hashes.add(image_hash)
        role = "study" if len(rows) < n else "pilot"
        image_id = image_hash[:20]
        rows.append({"id": image_id, "path": str(path), "sha256": image_hash,
                     "size_px": size, "mode": mode, "role": role})
        references.append({"id": image_id, "label": path.parent.name, "role": role,
                           "source": "NCT-CRC-HE-100K supplied class folder"})
    if len(rows) != n + pilot:
        raise ValueError("Insufficient readable unique tiles after exclusions")
    manifest = {"schema_version": 1, "created_at": now(), "source_root": str(source),
                "source_file_count": len(paths), "seed": seed,
                "selection": "seeded uniform permutation of all TIFF paths; no class balancing",
                "n_study": n, "n_pilot": pilot, "exclusions": exclusions, "images": rows}
    manifest["sha256"] = digest(manifest)
    write(output / "manifest.json", manifest)
    write(output / "references.json", {"purpose": "simulated audit and isolated evaluation",
                                        "labels": references})
    write(output / "prompts.json", PROMPTS)
    write(output / "study.json", {
        "schema_version": 1, "title": "CRC tissue composition audit", "created_at": now(),
        "manifest_sha256": manifest["sha256"], "classes": CLASSES, "endpoints": ENDPOINTS,
        "target_population": f"The {n} unique, readable tiles in this frozen manifest",
        "measurement": {"numerator": "Number of tiles with the endpoint reference class",
                        "denominator": "All retained study tiles, including BACK and DEB",
                        "aggregation": "Equal weight per tile; ratio of counts with known denominator",
                        "eligible_tissue": "Entire native tile; no pixel-level mask",
                        "exclusions": "Unreadable, non-224x224 RGB, or duplicate-byte images before freeze",
                        "physical_scale": "Dataset documents 0.5 microns/pixel; no physical area or distance estimated"},
        "scope": "Finite tile collection; no patient, whole-slide, spatial or clinical-outcome inference",
        "reference_limit": "Agreement with supplied dataset class labels; simulated review is not expert validation",
        "split": "No model training; disjoint pilot for allocation only; final audit labels never train the predictor",
        "model_training_overlap": "Unknown overlap with foundation-model pretraining; no independent generalization claim",
        "default_audit_budget": min(120, n), "familywise_alpha": 0.05,
        "pilot_labels": pilot, "endpoint_precision_scales": {key: 1.0 for key in ENDPOINTS},
    })
    return {"study_directory": str(output), "study_tiles": n, "pilot_tiles": pilot,
            "manifest_sha256": manifest["sha256"], "exclusions": len(exclusions)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--pilot", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.n, args.pilot, args.seed))
