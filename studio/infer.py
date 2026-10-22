"""Pinned PLIP inference, exclusively inside a Slurm GPU allocation.

Only image bytes and prompts enter the encoder; no reference artifact is read.
"""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import socket
import time

from .storage import digest, now, read, write

MODEL = "vinid/plip"
REVISION = "67ade53ddd32195868f422585f72698ef5d15094"


def infer(study_dir, limit=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference must run inside a Slurm GPU allocation")
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this inference job")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    study_dir = Path(study_dir)
    manifest = read(study_dir / "manifest.json")
    prompts = read(study_dir / "prompts.json")
    images = manifest["images"] if limit is None else manifest["images"][:limit]
    stem = "predictions" if limit is None else f"smoke-{limit}"
    if (study_dir / f"{stem}.json").exists():
        raise FileExistsError("Predictions are frozen; use a new study directory for another model")
    start = time.monotonic()
    # PLIP publishes tokenizer.json without the slow tokenizer's vocab/merges.
    # Select the fast tokenizer explicitly while preserving the saved PIL image pipeline.
    processor = CLIPProcessor(
        image_processor=CLIPImageProcessor.from_pretrained(MODEL, revision=REVISION,
                                                          local_files_only=True),
        tokenizer=CLIPTokenizerFast.from_pretrained(MODEL, revision=REVISION,
                                                  local_files_only=True))
    model = CLIPModel.from_pretrained(MODEL, revision=REVISION,
                                     local_files_only=True).to("cuda").eval()
    labels = list(prompts)
    with torch.inference_mode():
        tokens = processor(text=list(prompts.values()), return_tensors="pt", padding=True,
                           truncation=True)
        text_features = model.get_text_features(**{k: v.to("cuda") for k, v in tokens.items()})
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        records, embeddings = [], []
        for offset in range(0, len(images), 32):
            batch_rows = images[offset:offset + 32]
            batch_images = []
            for row in batch_rows:
                if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("Image changed since manifest freeze")
                with Image.open(row["path"]) as im:
                    batch_images.append(im.convert("RGB"))
            inputs = processor(images=batch_images, return_tensors="pt")
            features = model.get_image_features(pixel_values=inputs["pixel_values"].to("cuda"))
            features = torch.nn.functional.normalize(features, dim=-1)
            similarities = (features @ text_features.T).cpu().numpy()
            embs = features.cpu().numpy()
            for row, sims, embedding in zip(batch_rows, similarities, embs):
                order = np.argsort(sims)[::-1]
                records.append({"id": row["id"], "role": row["role"],
                                "predicted_label": labels[int(order[0])],
                                "cosine_scores": {label: float(v) for label, v in zip(labels, sims)},
                                "margin": float(sims[order[0]] - sims[order[1]])})
                embeddings.append(embedding)
            print(f"Embedded {min(offset + 32, len(images))}/{len(images)}", flush=True)
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(embedding_array).all():
        raise ValueError("Non-finite model output")
    embedding_file = study_dir / f"{stem}-embeddings.npz"
    with embedding_file.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array([r["id"] for r in records]),
                            embeddings=embedding_array,
                            text_embeddings=text_features.cpu().numpy(), labels=np.array(labels))
    provenance = {"model": MODEL, "model_revision": REVISION,
                  "representation": "L2-normalized projected CLIP image/text features",
                  "classification": "argmax cosine similarity against frozen prompts",
                  "score_interpretation": "Cosine similarity and margin; not calibrated probabilities",
                  "preprocessing": processor.image_processor.to_dict(),
                  "manifest_sha256": manifest["sha256"], "prompts_sha256": digest(prompts),
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                  "device": torch.cuda.get_device_name(), "created_at": now(),
                  "elapsed_seconds": time.monotonic() - start,
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("torch", "torchvision", "transformers", "numpy", "Pillow")},
                  "embedding_file_sha256": hashlib.sha256(embedding_file.read_bytes()).hexdigest()}
    result = {"provenance": provenance, "images": records}
    result["sha256"] = digest(result)
    write(study_dir / f"{stem}.json", result)
    print({"tiles": len(records), "embedding_dimension": embedding_array.shape[1],
           "elapsed_seconds": provenance["elapsed_seconds"], "output": stem}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    infer(args.study_dir, args.limit)
