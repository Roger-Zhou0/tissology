"""Optional concept interpretation around the frozen example-guided scorer."""
import json
import os
import socket
import urllib.error
from pathlib import Path

import numpy as np

from .semantic_language import LanguageInterpreter, concept_scores, digest, specification
from .semantic_v2_ranker import file_sha


class LanguageGroundedRanker:
    def __init__(self, store, base, config_path):
        self.base = base
        path = Path(config_path)
        self.config_path = path.resolve()
        self.config_file_sha256 = file_sha(path)
        self.config = json.loads(path.read_text())
        c = self.config
        if c["sha256"] != digest({k: v for k, v in c.items() if k != "sha256"}):
            raise ValueError("Language configuration hash changed")
        if (c["schema"] != "crc-language-runtime-v1" or c["catalog_sha256"] != store.catalog["sha256"]
                or c["specification_sha256"] != digest(specification())
                or c["source_sha256"] != file_sha(Path(__file__).with_name("semantic_language.py"))
                or c["adapter_sha256"] != file_sha(__file__)):
            raise ValueError("Language interpreter source or library differs from frozen candidate")
        probabilities_path = Path(c["library_probabilities_path"])
        if file_sha(probabilities_path) != c["library_probabilities_sha256"]:
            raise ValueError("Language classifier predictions changed")
        with np.load(probabilities_path, allow_pickle=False) as data:
            ids = data["ids"].tolist()
            self.class_codes = data["class_codes"].tolist()
            probabilities = data["probabilities"].copy()
        if len(ids) != len(set(ids)) or set(ids) != set(store.ids):
            raise ValueError("Language predictions are not aligned to this library")
        positions = {identifier: i for i, identifier in enumerate(ids)}
        self.probabilities = probabilities[[positions[x] for x in store.ids]]
        concept_scores(self.probabilities, self.class_codes,
                       {"targets": [self.class_codes[0]], "excluded_targets": [], "requires_examples": False})
        self.interpreter = LanguageInterpreter(os.environ.get("TISSUE_LANGUAGE_ENDPOINT", ""), c["model"])
        self.gallery_ids = base.gallery_ids
        self.version = c["version"]
        self.evaluation_asset = c.get("evaluation_asset")
        self.language_interpretation_available = True

    def rank(self, text, positive_ids, negative_ids, exclusions=""):
        if file_sha(self.config_path) != self.config_file_sha256:
            raise ValueError("Language configuration changed after initialization")
        result = self.base.rank(text, positive_ids, negative_ids, exclusions)
        result["provenance"]["language_config_sha256"] = self.config["sha256"]
        if positive_ids or negative_ids:
            result["provenance"]["language_interpretation"] = None
            return result
        try:
            interpretation = self.interpreter.interpret(text, exclusions)
            scores = concept_scores(self.probabilities, self.class_codes, interpretation)
        except (ValueError, KeyError, IndexError, TypeError, OSError, urllib.error.URLError, socket.timeout):
            interpretation = {"targets": [], "excluded_targets": [], "requires_examples": True,
                              "explanation": "The language interpreter could not produce a valid answer. Add matching and nonmatching examples."}
            scores = None
        result["provenance"]["language_interpretation"] = interpretation
        result["provenance"]["language_scope"] = specification()["scope"]
        if scores is None:
            result["notices"].append("Your description could not be grounded to a supported concept. These are exploratory text matches; examples can define your goal.")
            return result
        result["scores"] = scores
        result["method"] = "language_interpreted_crc_concept_ranking"
        result["score_interpretation"] = "Classifier probability mass for interpreted broad concepts, minus explicitly excluded concepts; not probability that your description is true"
        result["provenance"].update(abstain=False,
            classifier_head_sha256=self.config["classifier_head_sha256"],
            library_probabilities_sha256=self.config["library_probabilities_sha256"])
        result["notices"] = [notice for notice in result["notices"]
                             if not notice.startswith(("Text-only informal", "No learned relevance"))]
        result["notices"].append("The language model proposed broad tissue concepts. Check the visible interpretation; neither the language model nor class predictions verify every detail of your description.")
        return result
