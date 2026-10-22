"""Frozen CONCH inference on a Slurm GPU; references never enter the encoder.

Use a new experiment directory. Retrieval and classifier representations are
stored separately even though both have 512 coordinates.
"""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .storage import digest, now, read, write

MODEL = "MahmoodLab/CONCH"
REVISION = "f9ca9f877171a28ade80228fb195ac5d79003357"
CODE_REVISION = "141cc09c7d4ff33d8eda562bd75169b457f71a62"
CHECKPOINT_SHA256 = "40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c"
REPRESENTATION = "L2-normalized projected CONCH image/text features"
CLASSIFIER_REPRESENTATION = "L2-normalized unprojected CONCH pooled image features"


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_artifact(value, name):
    expected = value.get("sha256")
    if not expected or digest({k: v for k, v in value.items() if k != "sha256"}) != expected:
        raise ValueError(f"{name} does not match its frozen hash")


def verify_code(code_dir):
    code_dir = Path(code_dir).resolve()
    def git(*args):
        # A read-only approved checkout may be mounted under another host UID.
        # Trust only this explicit administrator path, never all repositories.
        return subprocess.check_output(["git", "-c", "safe.directory=" + str(code_dir),
                                        "-C", str(code_dir), *args], text=True).strip()
    if git("rev-parse", "HEAD") != CODE_REVISION:
        raise ValueError("CONCH source revision differs from the pinned revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("CONCH tracked source has local modifications")
    return code_dir


def load_encoder(checkpoint, code_dir, checkpoint_sha256):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CONCH inference must run inside a Slurm GPU allocation")
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("Expected checkpoint SHA256 differs from the pinned model revision's LFS object")
    code_dir = verify_code(code_dir)
    checkpoint = Path(checkpoint).resolve()
    if file_hash(checkpoint) != checkpoint_sha256:
        raise ValueError("CONCH checkpoint differs from the expected SHA256")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for CONCH inference")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    # Import only the verified local encoder package; no Hub downloads occur.
    sys.path.insert(0, str(code_dir))
    import conch.open_clip_custom.factory as factory
    from conch.open_clip_custom import get_tokenizer, tokenize
    if not Path(factory.__file__).resolve().is_relative_to(code_dir):
        raise ValueError("Imported CONCH package is outside the verified source tree")
    checkpoint_info = {}

    def checked_load(model, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint must contain a nonempty state dictionary")
        if all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        factory.resize_pos_embed(state, model)
        incompatible = model.load_state_dict(state, strict=False)
        missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        # Released weights deliberately omit the caption decoder. All image and
        # text encoder parameters must load; don't silently accept random ones.
        bad_missing = [key for key in missing if not key.startswith("text_decoder.")]
        if bad_missing or unexpected:
            raise ValueError(f"Checkpoint mismatch: missing={bad_missing}; unexpected={unexpected}")
        checkpoint_info.update({"weights_only": True, "loaded_tensor_count": len(state),
                               "missing_decoder_keys": missing, "unexpected_keys": unexpected})

    original_loader = factory.load_checkpoint
    factory.load_checkpoint = checked_load
    try:
        model, preprocess = factory.create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=str(checkpoint), device="cuda")
    finally:
        factory.load_checkpoint = original_loader
    model.eval()
    # Caption generation is not part of the released encoder or this workflow.
    model.text_decoder = None
    model.requires_grad_(False)
    torch.cuda.empty_cache()
    metadata = {
        "model": MODEL, "model_revision": REVISION, "code_revision": CODE_REVISION,
        "checkpoint_sha256": checkpoint_sha256, "checkpoint_loading": checkpoint_info,
        "representation": REPRESENTATION, "classifier_representation": CLASSIFIER_REPRESENTATION,
        "classifier_feature_key": "classifier_embeddings", "feature_transform": CLASSIFIER_REPRESENTATION,
        "feature_dimension": 512, "retrieval_feature_dimension": 512,
        "preprocessing": {"image_size": 448, "resize": "bicubic shortest edge",
                          "crop": "center 448x448", "color_mode": "RGB",
                          "mean": list(model.visual.image_mean), "std": list(model.visual.image_std),
                          "transform": repr(preprocess)},
        "classification": "argmax cosine similarity against frozen prompts",
        "score_interpretation": "Cosine similarity and margin; not calibrated probabilities",
        "numeric_precision": "float32; no autocast",
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "tokenizers", "timm", "numpy", "Pillow", "ftfy")},
    }
    return model, preprocess, get_tokenizer(), tokenize, metadata


def infer(study_dir, encoder, limit=None, batch_size=16):
    import numpy as np
    import torch
    from PIL import Image
    if limit is not None and limit < 1:
        raise ValueError("Smoke limit must be positive")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    study_dir = Path(study_dir)
    manifest = read(study_dir / "manifest.json")
    verify_artifact(manifest, "Manifest")
    prompts = read(study_dir / "prompts.json")
    if not isinstance(prompts, dict) or len(prompts) < 2 or any(not isinstance(p, str) or not p for p in prompts.values()):
        raise ValueError("Prompts must map at least two class names to nonempty text")
    prompt_sha256 = digest(prompts)
    if (study_dir / "experiment.json").exists():
        experiment = read(study_dir / "experiment.json")
        verify_artifact(experiment, "Experiment")
        if experiment.get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("Experiment does not identify this manifest")
        if experiment.get("prompts_sha256", prompt_sha256) != prompt_sha256:
            raise ValueError("Prompts changed after experiment freeze")
    images = manifest["images"] if limit is None else manifest["images"][:limit]
    if not images or len({row["id"] for row in images}) != len(images):
        raise ValueError("Manifest requires nonempty images with unique identifiers")
    stem = "predictions" if limit is None else f"smoke-{limit}"
    output_path = study_dir / f"{stem}.json"
    embedding_path = study_dir / f"{stem}-embeddings.npz"
    if output_path.exists() or embedding_path.exists():
        raise FileExistsError("Predictions are frozen; use a new study directory or smoke limit")
    model, preprocess, tokenizer, tokenize, metadata = encoder
    labels = list(prompts)
    records, retrieval_embeddings, classifier_embeddings = [], [], []
    start = time.monotonic()
    projection_max_error = None
    with torch.inference_mode():
        text_features = model.encode_text(tokenize(tokenizer, list(prompts.values())).to("cuda"), normalize=True)
        if text_features.shape != (len(labels), 512) or not torch.isfinite(text_features).all():
            raise ValueError("Unexpected CONCH text feature shape or values")
        for offset in range(0, len(images), batch_size):
            batch_rows = images[offset:offset + batch_size]
            tensors = []
            for row in batch_rows:
                if file_hash(row["path"]) != row["sha256"]:
                    raise ValueError("Image changed since manifest freeze")
                with Image.open(row["path"]) as image:
                    if "size_px" in row and list(image.size) != row["size_px"]:
                        raise ValueError("Image dimensions differ from manifest")
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors).to("cuda")
            raw = model.encode_image(batch, proj_contrast=False, normalize=False)
            if raw.shape != (len(batch_rows), 512) or not torch.isfinite(raw).all() or (raw.norm(dim=-1) == 0).any():
                raise ValueError("Unexpected CONCH classifier feature shape or values")
            classifier_features = torch.nn.functional.normalize(raw, dim=-1)
            # The pinned official vision tower applies this matrix immediately
            # after forward_no_head's pooled feature. Reuse one trunk pass.
            features = torch.nn.functional.normalize(raw @ model.visual.proj_contrast, dim=-1)
            if offset == 0:
                official = model.encode_image(batch[:1], proj_contrast=True, normalize=True)
                projection_max_error = float((official - features[:1]).abs().max().cpu())
                if not torch.allclose(official, features[:1], atol=1e-5, rtol=1e-4):
                    raise ValueError("Cached projection differs from official CONCH image encoding")
            if not torch.isfinite(features).all():
                raise ValueError("Non-finite CONCH retrieval features")
            similarities = (features @ text_features.T).cpu().numpy()
            retrieval_embeddings.append(features.cpu().numpy())
            classifier_embeddings.append(classifier_features.cpu().numpy())
            for row, scores in zip(batch_rows, similarities):
                order = np.argsort(scores)[::-1]
                records.append({"id": row["id"], "role": row["role"],
                                "predicted_label": labels[int(order[0])],
                                "cosine_scores": {label: float(value) for label, value in zip(labels, scores)},
                                "margin": float(scores[order[0]] - scores[order[1]])})
            print(f"CONCH {study_dir.name}: {min(offset + batch_size, len(images))}/{len(images)}", flush=True)
    # Detect edits made during a longer run, before publishing immutable outputs.
    if read(study_dir / "manifest.json") != manifest or digest(read(study_dir / "prompts.json")) != prompt_sha256:
        raise ValueError("Manifest or prompts changed during inference")
    with embedding_path.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array([row["id"] for row in records]),
                            embeddings=np.concatenate(retrieval_embeddings).astype(np.float32),
                            classifier_embeddings=np.concatenate(classifier_embeddings).astype(np.float32),
                            text_embeddings=text_features.cpu().numpy().astype(np.float32), labels=np.array(labels))
    provenance = dict(metadata, manifest_sha256=manifest["sha256"], prompts_sha256=prompt_sha256,
                      embedding_file_sha256=file_hash(embedding_path), slurm_job_id=os.environ["SLURM_JOB_ID"],
                      hostname=socket.gethostname(), device=torch.cuda.get_device_name(), created_at=now(),
                      elapsed_seconds=time.monotonic() - start, projection_check_max_error=projection_max_error,
                      batch_size=batch_size, smoke_limit=limit)
    result = {"provenance": provenance, "images": records}
    result["sha256"] = digest(result)
    write(output_path, result)
    print({"tiles": len(records), "retrieval_dimension": 512, "classifier_dimension": 512,
           "elapsed_seconds": provenance["elapsed_seconds"], "output": str(output_path)}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dirs", nargs="+")
    parser.add_argument("--checkpoint", default=os.environ.get("CONCH_CHECKPOINT"))
    parser.add_argument("--checkpoint-sha256", default=os.environ.get("CONCH_CHECKPOINT_SHA256", CHECKPOINT_SHA256))
    parser.add_argument("--code-dir", default=os.environ.get("CONCH_CODE_DIR"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.checkpoint or not args.code_dir or not args.checkpoint_sha256:
        parser.error("Set --checkpoint, --checkpoint-sha256 and --code-dir (or corresponding CONCH_* variables)")
    encoder = load_encoder(args.checkpoint, args.code_dir, args.checkpoint_sha256)
    for directory in args.study_dirs:
        infer(directory, encoder, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
