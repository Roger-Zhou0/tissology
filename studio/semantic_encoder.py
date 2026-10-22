"""Pinned CONCH text tower on a Slurm CPU, paired with projected image features."""
import os
from pathlib import Path
import sys
import threading

import numpy as np

from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, file_hash, verify_code
from .storage import read
from .execution import require_cpu_execution


class TextEncoder:
    def __init__(self, config):
        threads = require_cpu_execution()
        if config["source_checkpoint_sha256"] != CHECKPOINT_SHA256 or config["code_revision"] != CODE_REVISION:
            raise ValueError("Text tower provenance differs from the pinned CONCH encoder")
        code = verify_code(config["code_directory"])
        weights = Path(config["weights_path"])
        if file_hash(weights) != config["weights_sha256"]:
            raise ValueError("Text tower weights changed")
        sys.path.insert(0, str(code))
        import torch
        import conch.open_clip_custom.coca_model as coca
        from conch.open_clip_custom import get_tokenizer, tokenize
        if not Path(coca.__file__).resolve().is_relative_to(code):
            raise ValueError("Text encoder imported outside the verified code tree")
        model_config = read(code / "conch/open_clip_custom/model_configs/conch_ViT-B-16.json")
        self.model = coca._build_text_tower(model_config["embed_dim"], model_config["text_cfg"])
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().requires_grad_(False)
        torch.set_num_threads(min(4, threads))
        self.tokenizer, self.tokenize = get_tokenizer(), tokenize
        self.lock = threading.Lock()
        self.cache = {}

    def encode(self, texts):
        require_cpu_execution()
        import torch
        cleaned = []
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > 600:
                raise ValueError("Search needs 1–600 characters; select a short description or claim")
            text = text.strip()
            if len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False)) > 127:
                raise ValueError("Search exceeds the model's 127 text tokens; select a shorter phrase")
            cleaned.append(text)
        with self.lock:
            missing = list(dict.fromkeys(t for t in cleaned if t not in self.cache))
            if missing:
                with torch.inference_mode():
                    # Official CoCa.encode_text removes the final placeholder
                    # token before TextTransformer appends its learned CLS token.
                    tokens = self.tokenize(self.tokenizer, missing)[:, :-1]
                    pooled, _ = self.model(tokens)
                    vectors = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
                if vectors.shape != (len(missing), 512) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid text embeddings")
                if len(self.cache) + len(missing) > 256:
                    self.cache.clear()
                self.cache.update(zip(missing, vectors))
            return np.stack([self.cache[t] for t in cleaned]).copy()
