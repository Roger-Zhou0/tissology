"""Recorded fitting/selection identities shared by both collection check workflows.

This reads metadata, never images, feature arrays, reference labels or models.
Identity matching is deliberately narrower than a claim of training independence.
"""
from pathlib import Path
import os
import re

from .storage import digest, read
from .conch_infer import file_hash


IDENTITY_SCOPE = (
    "Exclusions match recorded original-byte, canonical-image and available pixel identities. "
    "Legacy records can lack pixel fingerprints, so re-encoded or transformed copies may evade "
    "matching. Unknown foundation-model pretraining overlap is not assessed; this does not "
    "establish entirely training-independent or patient-independent validation."
)


def identities(row):
    return {row[key] for key in ("sha256", "source_sha256", "image_sha256", "pixel_sha256", "pixel_digest")
            if isinstance(row.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", row[key])}


def _checked(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Known development identity evidence is unavailable")
    value = read(path)
    if not isinstance(value, dict) or value.get("sha256") != digest({key: item for key, item in value.items() if key != "sha256"}):
        raise ValueError("Known development identity evidence changed")
    return value


def resolve_artifact(path, expected_file_hash=None):
    """Relocate a pinned external artifact using explicit administrator mapping.

    Frozen protocol paths remain unchanged. A matching map entry takes priority
    even if the original is readable, but can never supply or replace the
    protocol's expected hash. Receipts subsequently bind the resolved path.
    """
    original = Path(path).absolute()
    configured = os.environ.get("TISSUE_ARTIFACT_MAP")
    if not configured or expected_file_hash is None:
        return original
    mapping_path = Path(configured)
    if mapping_path.is_symlink() or not mapping_path.is_file():
        raise ValueError("Configured evidence relocation map is unavailable")
    mapping = read(mapping_path)
    if (not isinstance(mapping, dict) or mapping.get("schema") != "tissue-artifact-map-v1" or
            not isinstance(mapping.get("entries"), dict)):
        raise ValueError("Configured evidence relocation map has an invalid format")
    entry = mapping["entries"].get(str(original))
    if entry is None:
        # No suffix matching, path guessing or implicit search outside the
        # recorded location. Missing originals fail at the ordinary file check.
        return original
    if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or
            not isinstance(entry["path"], str) or not entry["path"] or
            not isinstance(expected_file_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_file_hash) or
            entry["sha256"] != expected_file_hash):
        raise ValueError("Relocated evidence must match the recorded expected SHA256")
    configured_bundle = os.environ.get("TISSUE_MODEL_BUNDLE")
    if not configured_bundle:
        raise ValueError("Evidence relocation requires TISSUE_MODEL_BUNDLE")
    bundle = Path(configured_bundle)
    if not bundle.is_absolute() or not bundle.is_dir():
        raise ValueError("Evidence relocation requires an absolute model bundle directory")
    bundle = bundle.resolve()
    target = Path(entry["path"])
    if ".." in target.parts:
        raise ValueError("Relocated evidence must stay inside the model bundle")
    target = target if target.is_absolute() else bundle / target
    if target.is_symlink() or not target.is_file():
        raise ValueError("Relocated development evidence is unavailable")
    target = target.resolve()
    if not target.is_relative_to(bundle):
        raise ValueError("Relocated evidence must stay inside the model bundle")
    if file_hash(target) != expected_file_hash:
        raise ValueError("Relocated development evidence differs from its recorded SHA256")
    return target


def known_development_receipt(original, replay_directory, collections_directory, *, exclude_plan=None):
    """Freeze the resolved recorded identities, including the saved scan's release."""
    sources, hashes = [], set()

    def add(path, expected_file_hash=None, identity_field="images"):
        recorded_path = Path(path).absolute()
        path = resolve_artifact(recorded_path, expected_file_hash)
        actual_file_hash = file_hash(path)
        if expected_file_hash and actual_file_hash != expected_file_hash:
            raise ValueError("Known development identity evidence changed")
        value = _checked(path)
        rows = value.get(identity_field, []) if identity_field else []
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Known development identities have an invalid format")
        source_hashes = set().union(*(identities(row) for row in rows)) if rows else set()
        sources.append({"path": str(path), "sha256": value["sha256"],
                        "file_sha256": actual_file_hash, "identity_field": identity_field,
                        "identity_count": len(source_hashes), "identity_sha256": digest(sorted(source_hashes)),
                        **({"recorded_path": str(recorded_path)} if path != recorded_path else {})})
        hashes.update(source_hashes)
        return value

    # Every original-experiment role is already known, including its held-outs.
    add(Path(replay_directory) / "manifest.json")
    model = original.get("model", {})
    if model.get("release_sha256"):
        release_root = Path(model.get("release_directory", ""))
        if not release_root.is_absolute():
            raise ValueError("Model release development evidence is unavailable")
        release = add(release_root / "release.json", identity_field=None)
        if release["sha256"] != model["release_sha256"]:
            raise ValueError("Model release identity changed")
        receipts = {row["path"]: row["sha256"] for row in release["files"]}
        for name in ("original-manifest.json", "new-source-manifest.json"):
            relative = "evidence/training/" + name
            if relative not in receipts:
                raise ValueError("Model release training identities are unavailable")
            add(release_root / relative, receipts[relative])
        relative = "evidence/training/protocol.json"
        if relative not in receipts:
            raise ValueError("Model release selection evidence changed")
        protocol = add(release_root / relative, receipts[relative], identity_field=None)
        for directory in ("crc-expanded-development-003", "starc-train-feedback-001"):
            matches = [(path, sha) for path, sha in protocol.get("inputs", {}).items()
                       if Path(path).parent.name == directory and Path(path).name == "manifest.json"]
            if len(matches) != 1:
                raise ValueError("Model release selection identities are unavailable")
            add(*matches[0])
    omitted = Path(exclude_plan).absolute() if exclude_plan is not None else None
    for path in sorted(Path(collections_directory).glob("*/teaching/candidates/*/plan.json")):
        if path.absolute() != omitted:
            add(path, identity_field="development_labels")
    receipt = {"schema": "collection-known-development-v1",
               "original_prediction_sha256": original["sha256"],
               "sources": sources, "identity_hashes": sorted(hashes),
               "identity_count": len(hashes), "identity_scope": IDENTITY_SCOPE}
    return {**receipt, "sha256": digest(receipt)}


def verify_development_receipt(receipt, original_prediction_sha256, *, verify_sources=False):
    """Check a frozen receipt; reread external evidence only before new sampling."""
    if (receipt.get("schema") != "collection-known-development-v1" or
            receipt.get("sha256") != digest({key: value for key, value in receipt.items() if key != "sha256"}) or
            receipt.get("original_prediction_sha256") != original_prediction_sha256 or
            not isinstance(receipt.get("identity_hashes"), list) or
            any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in receipt["identity_hashes"]) or
            receipt["identity_hashes"] != sorted(set(receipt["identity_hashes"])) or
            receipt.get("identity_count") != len(receipt["identity_hashes"])):
        raise ValueError("Known development exclusion receipt changed")
    if verify_sources:
        hashes = set()
        for source in receipt["sources"]:
            path = Path(source["path"])
            if file_hash(path) != source["file_sha256"]:
                raise ValueError("Known development identity evidence changed after approval")
            value = _checked(path)
            field = source.get("identity_field")
            if field not in (None, "images", "development_labels") or value["sha256"] != source["sha256"]:
                raise ValueError("Known development exclusion source changed")
            rows = value.get(field, []) if field else []
            found = set().union(*(identities(row) for row in rows)) if rows else set()
            if source["identity_count"] != len(found) or source["identity_sha256"] != digest(sorted(found)):
                raise ValueError("Known development source identities changed")
            hashes.update(found)
        if sorted(hashes) != receipt["identity_hashes"]:
            raise ValueError("Known development exclusion identities changed")
    return set(receipt["identity_hashes"])
