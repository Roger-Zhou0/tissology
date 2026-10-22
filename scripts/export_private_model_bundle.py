#!/usr/bin/env python3
"""Export approved local artifacts for the owner's private CPU deployment.

No downloads, model inference, training, publication or frozen-record rewriting.
The output contains gated weights and development evidence: never commit it or
serve it as a public download. This script does not grant or proxy model access.
Run on an authorized CPU compute allocation, or an explicitly configured local
CPU host; the default execution policy requires bfqi-delta-cpu Slurm.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio.execution import execution_provenance, require_cpu_execution
from studio.storage import digest, now, read, write

REPLAY_FILES = ("classifier.json", "classifier-coefficients.npz", "experiment.json",
                "manifest.json", "development-labels.json", "predictions.json",
                "predictions-embeddings.npz")
_SECRET = re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                     rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|https?://[^\s/@:]+:[^\s/@]+@)")
_TEXT_SUFFIXES = {".json", ".py", ".txt", ".md", ".toml", ".yml", ".yaml", ".ipynb", ".ini", ".cfg"}
_GIT_CONFIG = "[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n\tlogallrefupdates = false\n"


def _git(arguments):
    # Absolute local paths plus file-only transport make network access
    # impossible, even if a user's Git configuration has URL rewrites/helpers.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="file", GIT_LFS_SKIP_SMUDGE="1")
    result = subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
                             "-c", "init.templateDir=", *arguments], env=environment,
                            capture_output=True, text=True, check=False, timeout=180)
    if result.returncode:
        # Git diagnostics may echo configured credentials. Keep them out of
        # receipts and console output; the fixed operation identifies failure.
        raise ValueError("The local-only CONCH Git verification/copy failed")
    return result.stdout.strip()


def _regular(path, *, allow_pinned_symlink=False):
    path = Path(path)
    if (path.is_symlink() and not allow_pinned_symlink) or not path.is_file():
        raise ValueError("A required private artifact is not a regular file")
    # Hugging Face snapshot files commonly link to local blob files. This is
    # allowed only for a separately hash-pinned checkpoint/text weight copy.
    return path.resolve()


def _check_text(path):
    if path.suffix.lower() in _TEXT_SUFFIXES or path.name in {"config", "LICENSE", ".gitignore"}:
        if _SECRET.search(path.read_bytes()):
            raise ValueError("A source artifact contains a possible credential; export refused")


def _check_filename(path):
    name = path.name.lower()
    if (name == ".env" or name.startswith(".env.") or name in {"credentials", ".netrc", ".git-credentials", "id_rsa", "id_ed25519"} or
            path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}):
        raise ValueError("Unexpected credential-like file in approved source inventory")


def check_sources(scan_config, replay_dir, text_config=None):
    """Verify only allowlisted assets, without copying ambient caches/config."""
    require_cpu_execution()
    from studio.collection_scan import CLASSIFIER_SHA256, validate_config
    from studio.conch_infer import CHECKPOINT_SHA256, CODE_REVISION, MODEL, file_hash, verify_code
    from studio.fusion_development import load_encoder
    from studio.model_releases import load_release
    from studio.study import verify

    configuration_path = _regular(scan_config)
    _check_text(configuration_path)
    configuration = read(configuration_path)
    validate_config(configuration)
    if configuration.get("schema_version") != 2 or configuration.get("backend") != "cpu":
        raise ValueError("Export requires a frozen schema-2 CPU scan configuration")
    release_dir = Path(configuration["release_directory"]).resolve()
    release, _, _, _ = load_release(release_dir, configuration["release_sha256"])
    checkpoint = _regular(configuration["checkpoint"], allow_pinned_symlink=True)
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("The checkpoint differs from the approved pinned CONCH weights")
    code = verify_code(configuration["code_directory"])
    tracked = _git(["-C", str(code), "ls-tree", "-r", "--name-only", CODE_REVISION]).splitlines()
    for relative in tracked:
        candidate = code / relative
        _check_filename(candidate)
        _check_text(_regular(candidate))
    replay = Path(replay_dir).resolve()
    # This verifies coefficient/feature hashes and role/encoder lineage without
    # loading image bytes or a foundation model. Labels stay in this private copy.
    blocks, replay_binding = load_encoder(replay, MODEL)
    replay_classifier = read(replay / "classifier.json")
    if (replay_classifier["sha256"] != CLASSIFIER_SHA256 or replay_classifier.get("refit_labels") != 5400 or
            sum(len(blocks[role]["ids"]) for role in ("train", "validation")) != 5400):
        raise ValueError("Replay must be the original 5,400-image CONCH development experiment")
    replay_files = []
    for name in REPLAY_FILES:
        _check_text(_regular(replay / name))
        replay_files.append((name, file_hash(replay / name)))
    release_files = [("release.json", file_hash(release_dir / "release.json"))] + [
        (row["path"], row["sha256"]) for row in release["files"]]
    for relative, expected in release_files:
        candidate = _regular(release_dir / relative)
        _check_filename(candidate)
        _check_text(candidate)
        if file_hash(candidate) != expected:
            raise ValueError("Release inventory changed during export validation")
    protocol = read(release_dir / "evidence/training/protocol.json")
    external = []
    for directory in ("crc-expanded-development-003", "starc-train-feedback-001"):
        matches = [(source, sha) for source, sha in protocol["inputs"].items()
                   if Path(source).parent.name == directory and Path(source).name == "manifest.json"]
        if len(matches) != 1:
            raise ValueError("Release protocol does not identify its external selection evidence")
        source, expected = matches[0]
        if not Path(source).is_absolute():
            raise ValueError("The protocol's source path must be absolute")
        from studio.collection_lineage import resolve_artifact
        resolved = _regular(resolve_artifact(source, expected))
        _check_text(resolved)
        verify(read(resolved))
        if file_hash(resolved) != expected:
            raise ValueError("External selection evidence differs from the frozen protocol")
        external.append({"source_path": source, "path": "external-evidence/" + directory + "/manifest.json",
                         "sha256": expected, "resolved_source": resolved})
    text = None
    if text_config:
        text_path = _regular(text_config)
        _check_text(text_path)
        text = read(text_path)
        verify(text)
        if text.get("source_checkpoint_sha256") != CHECKPOINT_SHA256 or text.get("code_revision") != CODE_REVISION:
            raise ValueError("Text tower provenance differs from the pinned CONCH encoder")
        weights = _regular(text["weights_path"], allow_pinned_symlink=True)
        if file_hash(weights) != text["weights_sha256"]:
            raise ValueError("Extracted text tower weights changed")
        text = {"source": weights, "sha256": text["weights_sha256"]}
    return {"configuration": configuration, "release": release, "release_dir": release_dir,
            "release_files": release_files, "checkpoint": checkpoint, "code": code, "replay": replay,
            "replay_binding": replay_binding, "replay_files": replay_files, "external": external, "text": text}


def _copy(source, destination, expected):
    from studio.conch_infer import file_hash
    source = _regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Export will not overwrite an existing artifact")
    if file_hash(source) != expected:
        raise ValueError("A private source changed after validation")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if file_hash(destination) != expected or file_hash(source) != expected:
        raise ValueError("Private artifact copy failed SHA256 readback")


def _clone_code(source, destination):
    from studio.conch_infer import CODE_REVISION, verify_code
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A shallow file-transport clone copies only the pinned reachable tree,
    # rather than unrelated/unreachable objects from the owner's local .git.
    _git(["clone", "--no-hardlinks", "--no-local", "--depth=1", "--no-checkout", "--", str(source), str(destination)])
    _git(["-C", str(destination), "checkout", "--detach", CODE_REVISION])
    _git(["-C", str(destination), "remote", "remove", "origin"])
    for name in ("logs", "hooks"):
        shutil.rmtree(destination / ".git" / name, ignore_errors=True)
    for name in ("FETCH_HEAD", "ORIG_HEAD"):
        (destination / ".git" / name).unlink(missing_ok=True)
    (destination / ".git/config").write_text(_GIT_CONFIG)
    if _git(["-C", str(destination), "remote"]) or (destination / ".git/config").read_text() != _GIT_CONFIG:
        raise ValueError("Exported CONCH checkout retained unexpected Git configuration")
    verify_code(destination)


def export(scan_config, replay_dir, output, text_config=None):
    require_cpu_execution()
    requested = Path(output).expanduser()
    if requested.is_symlink() or (requested.exists() and (not requested.is_dir() or any(requested.iterdir()))):
        raise ValueError("Output must be a new or empty directory")
    destination = requested.resolve()
    if destination.is_relative_to(Path(__file__).resolve().parents[1] / "studio/static"):
        raise ValueError("Private model artifacts cannot be exported into web static files")
    source = check_sources(scan_config, replay_dir, text_config)
    if any(destination.is_relative_to(source[key]) for key in ("release_dir", "code", "replay")):
        raise ValueError("Export output cannot be nested inside a source artifact directory")
    previous_umask = os.umask(0o077)
    try:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.chmod(0o700)
        (destination / ".gitignore").write_text("*\n")
        (destination / "PRIVATE_README.txt").write_text(
            "PRIVATE MODEL BUNDLE — DO NOT COMMIT OR PUBLISH\n\n"
            "Contains approved gated CONCH weights and development evidence for the owner's private deployment.\n"
            "Upstream model and dataset access terms still apply. Possessing this bundle does not grant others\n"
            "access or redistribution rights. It is not an access proxy. Never mount this directory as web static\n"
            "files, attach it to a public repository/release, or expose a public file download.\n\n"
            "Original release, protocol, replay and source-evidence bytes are unchanged. Bundle installation\n"
            "creates separate host configuration; it does not rewrite historical research records.\n")
        from studio.conch_infer import CHECKPOINT_SHA256, CODE_REVISION, file_hash
        for relative, expected in source["release_files"]:
            _copy(source["release_dir"] / relative, destination / "release" / relative, expected)
        _copy(source["checkpoint"], destination / "weights/conch.bin", CHECKPOINT_SHA256)
        _clone_code(source["code"], destination / "code/CONCH")
        for name, expected in source["replay_files"]:
            _copy(source["replay"] / name, destination / "replay" / name, expected)
        external = []
        for row in source["external"]:
            _copy(row["resolved_source"], destination / row["path"], row["sha256"])
            external.append({key: row[key] for key in ("source_path", "path", "sha256")})
        bundle = {"schema": "tissue-private-model-bundle-v1", "created_at": now(),
                  "release_directory": "release", "release_sha256": source["release"]["sha256"],
                  "checkpoint": "weights/conch.bin", "checkpoint_sha256": CHECKPOINT_SHA256,
                  "code_directory": "code/CONCH", "code_revision": CODE_REVISION,
                  "replay_directory": "replay", "external_evidence": external,
                  "source_scan_configuration_sha256": source["configuration"]["sha256"],
                  "private_deployment_only": True, "downloads_performed": False,
                  "frozen_records_rewritten": False, **execution_provenance()}
        if source["text"]:
            _copy(source["text"]["source"], destination / "weights/text-tower.pt", source["text"]["sha256"])
            bundle.update(text_weights="weights/text-tower.pt", text_weights_sha256=source["text"]["sha256"])
        # Reverify the independently copied scientific artifacts, not just their
        # parent inventory. No model execution or original image access occurs.
        from studio.model_releases import load_release
        from studio.fusion_development import load_encoder
        from studio.conch_infer import MODEL
        load_release(destination / "release", source["release"]["sha256"])
        load_encoder(destination / "replay", MODEL)
        inventory = []
        for path in sorted(destination.rglob("*")):
            if path.is_symlink():
                raise ValueError("Unexpected symlink in private artifact export")
            if path.is_file():
                _check_filename(path)
                _check_text(path)
                inventory.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size,
                                  "sha256": file_hash(path)})
        bundle["files"] = inventory
        bundle["total_bytes"] = sum(row["bytes"] for row in inventory)
        bundle["sha256"] = digest(bundle)
        write(destination / "bundle.json", bundle)
        if read(destination / "bundle.json") != bundle:
            raise ValueError("Private bundle metadata readback differs")
        return bundle
    finally:
        os.umask(previous_umask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-config", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--text-config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = export(args.scan_config, args.replay_dir, args.output, args.text_config)
    print(f"Private bundle verified: {len(bundle['files'])} files, {bundle['total_bytes']} bytes; no network download or publication.")


if __name__ == "__main__":
    main()
