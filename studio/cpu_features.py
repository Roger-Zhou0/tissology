"""Bounded, frozen CONCH image features inside a bfqi CPU Slurm allocation.

This worker does not classify images, load reference labels, train the encoder,
or publish collection scans. Its two feature spaces are saved separately.
"""
import argparse
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import re
import socket
import sys
import time

from .conch_infer import (CHECKPOINT_SHA256, CLASSIFIER_REPRESENTATION,
                         CODE_REVISION, MODEL, REPRESENTATION, REVISION,
                         file_hash, verify_artifact, verify_code)
from .storage import digest, now, read, write
from .execution import execution_provenance, require_cpu_execution

CPU_ACCOUNT = "bfqi-delta-cpu"
MAX_IMAGES = 2048
MAX_BATCH_SIZE = 8
MAX_THREADS = 8
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def require_cpu_allocation(threads=None):
    """Validate scheduled CPU work or an explicitly configured local host."""
    return require_cpu_execution(threads)


def load_cpu_encoder(checkpoint, code_dir, checkpoint_sha256=CHECKPOINT_SHA256, *, threads=None):
    """Return (frozen CPU model, official preprocessing, pinned provenance)."""
    selected_threads = require_cpu_allocation(threads)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("Expected checkpoint differs from the pinned CONCH checkpoint")
    code_dir = verify_code(code_dir)
    checkpoint = Path(checkpoint).resolve()
    if file_hash(checkpoint) != checkpoint_sha256:
        raise ValueError("CONCH checkpoint differs from the expected SHA256")
    import torch
    torch.set_num_threads(selected_threads)
    sys.path.insert(0, str(code_dir))
    import conch.open_clip_custom.factory as factory
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
        bad_missing = [key for key in missing if not key.startswith("text_decoder.")]
        if bad_missing or unexpected:
            raise ValueError(f"Checkpoint mismatch: missing={bad_missing}; unexpected={unexpected}")
        checkpoint_info.update(weights_only=True, loaded_tensor_count=len(state),
                               missing_decoder_keys=missing, unexpected_keys=unexpected)

    original_loader = factory.load_checkpoint
    factory.load_checkpoint = checked_load
    try:
        model, preprocess = factory.create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=str(checkpoint), device="cpu")
    finally:
        factory.load_checkpoint = original_loader
    model.eval().requires_grad_(False)
    model.text_decoder = None
    _verify_cpu_model(model)
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
        "numeric_precision": "float32; no autocast", "device": "cpu", "threads": selected_threads,
        "code_directory": str(code_dir), "factory_module": str(Path(factory.__file__).resolve()),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "tokenizers", "timm", "numpy", "Pillow", "ftfy")},
    }
    return model, preprocess, metadata


def _verify_cpu_model(model):
    if model.training:
        raise ValueError("CONCH must be in evaluation mode")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "cpu":
            raise ValueError("All CONCH tensors must remain on CPU")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("CONCH must remain frozen")


def _rows(rows, role, max_images):
    if type(max_images) is not int or not 1 <= max_images <= MAX_IMAGES:
        raise ValueError(f"max_images must be between 1 and {MAX_IMAGES}")
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= max_images:
        raise ValueError("A nonempty bounded image list is required")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("A nonempty image role is required")
    result = []
    for source in rows:
        identifier = source.get("id")
        image_hash = source.get("sha256")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Every image requires an identifier")
        if not isinstance(image_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            raise ValueError("Every image requires a frozen SHA256")
        image_role = source.get("role", role)
        if not isinstance(image_role, str) or not image_role.strip():
            raise ValueError("Every image requires a nonempty role")
        row = {"id": identifier, "path": str(Path(source["path"]).resolve()),
               "sha256": image_hash, "role": image_role}
        if "size_px" in source:
            size = source["size_px"]
            if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(x) is not int or x < 1 for x in size):
                raise ValueError("Image dimensions must be two positive integers")
            row["size_px"] = list(size)
        result.append(row)
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("Image identifiers must be unique")
    return result


def _image_bytes(row):
    path = Path(row["path"])
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 25 MiB input bound")
    with path.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 25 MiB input bound")
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Image changed since manifest freeze")
    return data


def encode_images(rows, encoder, output_dir, *, batch_size=1, max_images=256,
                  role="inference", manifest_path=None):
    """Create immutable features.npz and features.json in a new directory.

    Optional manifest_path must be a hash-verified study manifest containing
    every selected image. The image list is separately hashed in all cases.
    Failed jobs may leave an incomplete output directory; choose a new one.
    """
    threads = require_cpu_allocation()
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be between 1 and {MAX_BATCH_SIZE}")
    images = _rows(rows, role, max_images)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("CPU feature outputs are immutable; choose a new output directory")
    manifest = None
    manifest_file_hash = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path).resolve()
        manifest = read(manifest_path)
        verify_artifact(manifest, "Manifest")
        manifest_file_hash = file_hash(manifest_path)
        lookup = {row["id"]: row for row in manifest["images"]}
        if len(lookup) != len(manifest["images"]):
            raise ValueError("Manifest contains duplicate image identifiers")
        for row in images:
            if row["id"] not in lookup or _rows([lookup[row["id"]]], role, 1)[0] != row:
                raise ValueError("Selected image identity or role differs from the frozen manifest")
    # Reject stale inputs before starting the model computation.
    for row in images:
        _image_bytes(row)
    import numpy as np
    import torch
    from PIL import Image
    model, preprocess, metadata = encoder
    _verify_cpu_model(model)
    if metadata.get("model") != MODEL or metadata.get("model_revision") != REVISION or metadata.get("code_revision") != CODE_REVISION or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Encoder provenance differs from the pinned CONCH identity")
    if metadata.get("representation") != REPRESENTATION or metadata.get("classifier_representation") != CLASSIFIER_REPRESENTATION:
        raise ValueError("Encoder provenance does not identify both feature spaces")
    selected_threads = metadata.get("threads", threads)
    require_cpu_allocation(selected_threads)
    torch.set_num_threads(selected_threads)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    projected_parts, pooled_parts = [], []
    projection_max_error = None
    with torch.inference_mode():
        for offset in range(0, len(images), batch_size):
            batch_rows = images[offset:offset + batch_size]
            tensors = []
            for row in batch_rows:
                with Image.open(io.BytesIO(_image_bytes(row))) as image:
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Image exceeds the decoded pixel bound")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("CPU feature extraction accepts single-frame images only")
                    if "size_px" in row and list(image.size) != row["size_px"]:
                        raise ValueError("Image dimensions differ from the manifest")
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors).to(device="cpu", dtype=torch.float32)
            raw = model.encode_image(batch, proj_contrast=False, normalize=False)
            if raw.device.type != "cpu" or raw.shape != (len(batch_rows), 512) or not torch.isfinite(raw).all() or (raw.norm(dim=-1) == 0).any():
                raise ValueError("Unexpected CONCH pooled feature shape, device or values")
            pooled = torch.nn.functional.normalize(raw, dim=-1)
            projected_raw = raw @ model.visual.proj_contrast
            if projected_raw.shape != (len(batch_rows), 512) or not torch.isfinite(projected_raw).all() or (projected_raw.norm(dim=-1) == 0).any():
                raise ValueError("Unexpected CONCH projected feature shape or values")
            projected = torch.nn.functional.normalize(projected_raw, dim=-1)
            if offset == 0:
                official = model.encode_image(batch[:1], proj_contrast=True, normalize=True)
                projection_max_error = float((official - projected[:1]).abs().max())
                if not torch.allclose(official, projected[:1], atol=1e-5, rtol=1e-4):
                    raise ValueError("Cached projection differs from official CONCH image encoding")
            pooled_parts.append(pooled.numpy().astype(np.float32))
            projected_parts.append(projected.numpy().astype(np.float32))
            print(f"CPU CONCH: {min(offset + batch_size, len(images))}/{len(images)}", flush=True)
    _verify_cpu_model(model)
    for row in images:
        _image_bytes(row)
    if manifest_path is not None and file_hash(manifest_path) != manifest_file_hash:
        raise ValueError("Manifest changed during CPU feature extraction")
    feature_path = output_dir / "features.npz"
    with feature_path.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array([row["id"] for row in images]),
                            embeddings=np.concatenate(projected_parts),
                            classifier_embeddings=np.concatenate(pooled_parts))
    receipt = {
        "schema": "tissue-cpu-features-v1", "created_at": now(), "images": images,
        "image_list_sha256": digest(images), "feature_path": str(feature_path.resolve()),
        "feature_file_sha256": file_hash(feature_path),
        "provenance": dict(metadata, device="cpu", reference_labels_read=False,
                           encoder_training_performed=False, **execution_provenance(),
                           hostname=socket.gethostname(),
                           elapsed_seconds=time.monotonic() - started, batch_size=batch_size,
                           threads=selected_threads, projection_check_max_error=projection_max_error,
                           manifest_path=str(manifest_path) if manifest_path else None,
                           manifest_file_sha256=manifest_file_hash,
                           manifest_sha256=manifest["sha256"] if manifest else None,
                           implementation_sha256=file_hash(__file__)),
    }
    receipt["sha256"] = digest(receipt)
    write(output_dir / "features.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="A frozen study manifest containing image identities and roles")
    parser.add_argument("output", help="New output directory; existing directories are refused")
    parser.add_argument("--checkpoint", default=os.environ.get("CONCH_CHECKPOINT"))
    parser.add_argument("--code-dir", default=os.environ.get("CONCH_CODE_DIR"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if not args.checkpoint or not args.code_dir:
        parser.error("Set --checkpoint and --code-dir or their CONCH_* environment variables")
    if not 1 <= args.limit <= MAX_IMAGES:
        parser.error(f"--limit must be between 1 and {MAX_IMAGES}")
    require_cpu_allocation(args.threads)
    manifest = read(args.manifest)
    verify_artifact(manifest, "Manifest")
    selected = _rows(manifest["images"][:args.limit], "inference", args.limit)
    if Path(args.output).exists():
        raise FileExistsError("CPU feature outputs are immutable; choose a new output directory")
    encoder = load_cpu_encoder(args.checkpoint, args.code_dir, threads=args.threads)
    encode_images(selected, encoder, args.output, batch_size=args.batch_size,
                  max_images=args.limit, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
