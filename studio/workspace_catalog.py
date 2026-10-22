"""Read-only discovery of saved work, independent of image/embedding loading.

This is a metadata catalog, not a replacement for the full validation performed
when a record is opened. It never creates an index, reconciles a job, or rewrites
scientific records. Explicit limits bound directory traversal and JSON reads;
partial results and damaged records are surfaced rather than silently omitted.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlencode

from fastapi import Query

from .storage import digest

ID = re.compile(r"^[a-f0-9]{32}$")
KINDS = {"all", "collection", "study", "concept", "description"}
EXAMPLE_COLLECTIONS = {"307b6b38b73f455390059a997ec2ceb2", "38482ea0200b42ef9fffa85e1b939db1"}
MAX_ENTRIES = 20000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_READ_BYTES = 128 * 1024 * 1024


class CatalogLimit(ValueError):
    pass


def checked(value):
    if not isinstance(value, dict) or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Saved metadata failed its integrity check")
    return value


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("Saved record has an invalid identifier")
    return value


def date(value):
    if not isinstance(value, str):
        raise ValueError("Saved record has no valid date")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Saved record date has no timezone")
    return parsed.timestamp()


def link(page, **params):
    return page + "?" + urlencode({k: v for k, v in params.items() if v is not None})


class Reader:
    def __init__(self, max_entries=MAX_ENTRIES, max_bytes=MAX_READ_BYTES):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries = self.bytes = 0
        self.cache = {}

    @staticmethod
    def safe(path, root):
        path, root = Path(path), Path(root)
        if not path.is_relative_to(root):
            raise ValueError("Saved metadata is outside its configured workspace")
        current = path
        while current != root:
            if current.is_symlink():
                raise ValueError("Saved metadata contains a symbolic link")
            current = current.parent

    def paths(self, directory, root, *, directories=False):
        self.safe(directory, root)
        if not directory.exists():
            return []
        found = []
        with os.scandir(directory) as entries:
            for entry in entries:
                self.entries += 1
                if self.entries > self.max_entries:
                    raise CatalogLimit("The saved-work listing reached its directory limit; results are incomplete")
                if entry.name.startswith("."):
                    continue
                if directories or entry.name.endswith(".json"):
                    found.append(Path(entry.path))
        return sorted(found, key=lambda path: path.name)

    def read(self, path, root, *, sealed=True):
        self.safe(path, root)
        if path in self.cache:
            value = self.cache[path]
            return checked(value) if sealed else value
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
                raise ValueError("Saved metadata is not a bounded regular file")
            if self.bytes + before.st_size > self.max_bytes:
                raise CatalogLimit("The saved-work listing reached its metadata limit; results are incomplete")
            raw = handle.read(MAX_FILE_BYTES + 1)
            self.bytes += len(raw)
            after = os.fstat(handle.fileno())
            if len(raw) > MAX_FILE_BYTES or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Saved metadata changed while it was read")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Saved metadata is not an object")
        self.cache[path] = value
        return checked(value) if sealed else value


class WorkspaceCatalog:
    def __init__(self, collections=None, semantic=None, research=None, *, collection_encoder=None,
                 max_entries=MAX_ENTRIES, max_bytes=MAX_READ_BYTES):
        self.roots = {key: Path(value).resolve() if value else None for key, value in
                      (("collections", collections), ("semantic", semantic), ("research", research),
                       ("collection_encoder", collection_encoder if collection_encoder is not None else semantic))}
        self.max_entries, self.max_bytes = max_entries, max_bytes

    def listing(self, *, q="", kind="all", offset=0, limit=12, include_examples=False):
        if kind not in KINDS or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100 or len(q) > 500:
            raise ValueError("Invalid saved-work filters or page")
        reader = Reader(self.max_entries, self.max_bytes)
        rows, warnings = [], []
        example_study = None

        def warning(message):
            if message not in warnings and len(warnings) < 20:
                warnings.append(message)

        def unavailable(record_kind, record_id, source, collection_id=None):
            return {"kind": record_kind, "id": record_id, "name": "Unavailable saved record · " + record_id[:8],
                    "source": source, "collection_id": collection_id, "collection_name": source,
                    "created_at": "", "status": "Needs attention", "detail": "Saved metadata could not be verified. Refresh or restore its source record.",
                    "available": False, "href": None, "report_href": None, "example": False}

        def append_record(path, record_kind, source, function, collection_id=None):
            try:
                rows.append(function())
            except CatalogLimit:
                raise
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                rows.append(unavailable(record_kind, path.stem if path.suffix else path.name, source, collection_id))
                warning("Some saved records could not be verified. They remain listed as needing attention.")

        def record_base(record, record_kind, source, collection_id=None):
            identifier(record["id"])
            date(record["created_at"])
            name = record.get("name", record.get("text"))
            if not isinstance(name, str) or not name.strip() or len(name) > 6000 or not isinstance(source, str):
                raise ValueError("Saved record name is invalid")
            if record.get("reviewer") is not None and not isinstance(record["reviewer"], str):
                raise ValueError("Saved reviewer metadata is invalid")
            return {"kind": record_kind, "id": record["id"], "name": record.get("name", record.get("text", "Saved record")),
                    "source": source, "collection_name": source, "collection_id": collection_id,
                    "created_at": record["created_at"], "reviewer": record.get("reviewer"),
                    "available": True, "report_href": None, "example": False}

        def semantic_records(directory, root, source, context, collection_id=None):
            # Only explicitly named searches are work items. Their source query
            # records are validated below, without flooding the list with every keystroke.
            for folder, record_kind in (("concepts", "concept"), ("descriptions", "description")):
                for path in reader.paths(directory / folder, root):
                    def item(path=path, folder=folder, record_kind=record_kind):
                        record = load_record(directory, root, folder, path.stem, context, collection_id)
                        result = record_base(record, record_kind, source, collection_id)
                        version = record.get("prediction_version")
                        result["prediction_version"] = version
                        params = {"collection": collection_id, "prediction_version": ((version or {}).get("id", "original") if record_kind == "description" else (version or {}).get("id")) if collection_id else None}
                        params["concept" if record_kind == "concept" else "description"] = record["id"]
                        result.update(href=link("semantic", **params), status="Saved search" if record_kind == "concept" else "Saved finding",
                                      detail=record.get("reviewer", ""))
                        if record_kind == "description":
                            result["report_href"] = link("report", kind="description", id=record["id"], collection=collection_id)
                        return result
                    append_record(path, record_kind, source, item, collection_id)

        def load_record(directory, root, folder, record_id, context, collection_id=None, *, dependencies=True):
            identifier(record_id)
            record = reader.read(directory / folder / (record_id + ".json"), root)
            if record.get("id") != record_id:
                raise ValueError("Saved record identity differs")
            catalog, images = context()
            if record.get("catalog_sha256") != catalog["sha256"]:
                raise ValueError("Saved record catalog binding differs")
            if collection_id and (record.get("collection_id") != collection_id or record.get("provenance") != catalog):
                raise ValueError("Saved record collection binding differs")
            if dependencies and folder == "concepts":
                query = load_record(directory, root, "searches", record["search_id"], context, collection_id, dependencies=False)
                if record.get("search_sha256") != query["sha256"]:
                    raise ValueError("Saved search query binding differs")
            if folder == "searches":
                for image in record.get("results", []):
                    if images.get(image["id"]) != image["image_sha256"]:
                        raise ValueError("Saved search image binding differs")
            if folder == "descriptions":
                for claim in record["claims"]:
                    if (claim["text"] not in record["text"] or
                            claim["assessment"] not in {"supported_by_reviewer", "contradicted_by_reviewer", "unresolved"} or
                            [e["image_id"] for e in claim["evidence"] if not e.get("region_id")] != claim["image_ids"] or
                            [e["region_id"] for e in claim["evidence"] if e.get("region_id")] != claim.get("region_ids", [])):
                        raise ValueError("Saved finding claim binding differs")
                    for evidence in claim["evidence"]:
                        if images.get(evidence["image_id"]) != evidence["image_sha256"]:
                            raise ValueError("Saved finding image binding differs")
                        if evidence.get("region_id"):
                            # Match CollectionRegions.evidence() without loading images,
                            # feature matrices, or an encoder. Full pixel validation
                            # still happens when the finding or its evidence is opened.
                            from .collection_regions import COORDINATES, _bounds
                            region_id = evidence["region_id"]
                            if not collection_id or not isinstance(region_id, str) or not re.fullmatch(r"[a-f0-9]{64}", region_id):
                                raise ValueError("Saved finding region identity differs")
                            region = reader.read(directory / "regions" / region_id / "region.json", root)
                            size = region["source_size_px"]
                            if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
                                raise ValueError("Saved finding region dimensions differ")
                            binding = {"collection_id": collection_id, "catalog_sha256": catalog["sha256"],
                                       "image_id": evidence["image_id"], "image_sha256": images[evidence["image_id"]],
                                       "bounds_px": _bounds(region["bounds_px"], size), "source_size_px": size,
                                       "prediction_version": record.get("prediction_version"),
                                       "coordinate_system": COORDINATES,
                                       "crop_format": "RGB PNG; exact integer pixel rectangle; no resizing"}
                            expected = {"region_id": region_id, "region_sha256": region["sha256"],
                                        **{key: region[key] for key in (
                                            "image_id", "image_sha256", "image_url", "source_image_url",
                                            "bounds_px", "source_size_px", "crop_sha256", "prediction_version",
                                            "coordinate_system", "annotation_extent")}}
                            base = f"api/collections/{collection_id}/semantic"
                            if (region.get("schema") != "collection-region-v1" or region.get("id") != region_id or
                                    digest(binding) != region_id or any(region.get(key) != value for key, value in binding.items()) or
                                    region.get("image_url") != f"{base}/regions/{region_id}/image" or
                                    region.get("source_image_url") != f"{base}/images/{evidence['image_id']}" or
                                    region.get("annotation_extent") != "selected rectangular region" or
                                    evidence != expected):
                                raise ValueError("Saved finding region binding differs")
                if dependencies and record.get("parent_id"):
                    parent = load_record(directory, root, folder, record["parent_id"], context, collection_id, dependencies=False)
                    if parent["sha256"] != record.get("parent_sha256"):
                        raise ValueError("Saved finding parent binding differs")
                if collection_id:
                    version = record.get("prediction_version")
                    if version:
                        if version.get("id") == "original":
                            prediction = reader.read(directory.parent / "predictions.json", root)
                        else:
                            version_id = identifier(version.get("id"))
                            prediction = reader.read(directory.parent / "teaching" / "candidates" / version_id / "predictions.json", root)
                        if version.get("prediction_sha256") != prediction["sha256"]:
                            raise ValueError("Saved prediction version binding differs")
                        if prediction.get("collection_id") != collection_id:
                            raise ValueError("Saved prediction version belongs to another collection")
            return record

        collections = self.roots["collections"]
        if collections:
            try:
                for directory in reader.paths(collections, collections, directories=True):
                    if not ID.fullmatch(directory.name):
                        continue
                    collection_id = directory.name
                    source = "Image collection · " + collection_id[:8]
                    try:
                        metadata = reader.read(directory / "collection.json", collections)
                        if metadata.get("id") != collection_id:
                            raise ValueError("Collection identity differs")
                        source = metadata["name"]
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        if isinstance(error, CatalogLimit):
                            raise
                        rows.append(unavailable("collection", collection_id, source, collection_id))
                        warning("Some collection metadata could not be verified; its saved work may be unavailable.")
                        # Still discover individual record IDs when their container is damaged.
                        metadata = None

                    def collection_item():
                        scan = reader.read(directory / "scan.json", collections, sealed=False)
                        result = record_base(metadata, "collection", "Imported images", collection_id)
                        result["collection_name"] = metadata["name"]
                        status = scan.get("status", "unknown")
                        result.update(status={"complete": "Scan complete", "ready": "Ready to scan", "importing": "Import in progress", "queued": "Scan queued", "scanning": "Scanning", "failed": "Needs attention"}.get(status, "Scan status unavailable"),
                                      detail="", href=link("collections", collection=collection_id), example=collection_id in EXAMPLE_COLLECTIONS)
                        if status == "complete":
                            total = scan.get("total")
                            if isinstance(total, int) and total >= 0:
                                result["detail"] = f"{total:,} images"
                        return result
                    if metadata is not None:
                        append_record(directory, "collection", source, collection_item, collection_id)

                    context_cache = []
                    def context(directory=directory, metadata=metadata, context_cache=context_cache, collection_id=collection_id):
                        if context_cache:
                            return context_cache[0]
                        if metadata is None:
                            raise ValueError("Collection metadata is unavailable")
                        manifest = reader.read(directory / "manifest.json", collections)
                        predictions = reader.read(directory / "predictions.json", collections)
                        scan = reader.read(directory / "scan.json", collections, sealed=False)
                        encoder_root = self.roots["collection_encoder"]
                        if encoder_root is None:
                            raise ValueError("Search encoder metadata is unavailable")
                        encoder = reader.read(encoder_root / "encoder.json", encoder_root)
                        if (manifest.get("collection_id") != collection_id or manifest.get("role") != "unlabeled_inspection"
                                or predictions.get("collection_id") != collection_id or predictions.get("manifest_sha256") != manifest["sha256"]
                                or scan.get("status") != "complete" or scan.get("prediction_sha256") != predictions["sha256"]):
                            raise ValueError("Collection prediction binding differs")
                        # Match the saved catalog exactly, without opening feature chunks.
                        from .collection_semantic import SCOPE
                        from .conch_infer import CHECKPOINT_SHA256, MODEL, REPRESENTATION
                        catalog = {"schema": "collection-semantic-catalog-v1", "collection": {"id": collection_id, "name": metadata["name"]},
                                   "collection_sha256": metadata["sha256"], "manifest_sha256": manifest["sha256"],
                                   "prediction_sha256": predictions["sha256"], "encoder_sha256": encoder["sha256"],
                                   "feature_chunks": predictions["model"]["feature_chunks"], "model": MODEL,
                                   "source_checkpoint_sha256": CHECKPOINT_SHA256, "representation": REPRESENTATION,
                                   "scope": SCOPE, "role": "unlabeled_inspection"}
                        catalog["sha256"] = digest(catalog)
                        images = {row["id"]: row["image_sha256"] for row in manifest["images"]}
                        if len(images) != len(manifest["images"]):
                            raise ValueError("Collection image identities differ")
                        context_cache.append((catalog, images))
                        return context_cache[0]
                    semantic_records(directory / "semantic", collections, source, context, collection_id)
            except (OSError, ValueError, KeyError, TypeError) as error:
                warning(str(error) if isinstance(error, CatalogLimit) else "Some collection records could not be listed. Refresh to reconnect.")

        semantic = self.roots["semantic"]
        if semantic:
            try:
                library_cache = []
                def library_context():
                    if library_cache:
                        return library_cache[0]
                    catalog = reader.read(semantic / "catalog.json", semantic)
                    encoder = reader.read(semantic / "encoder.json", semantic)
                    if catalog.get("encoder_sha256") != encoder["sha256"]:
                        raise ValueError("Library encoder binding differs")
                    images = {row["id"]: row["image_sha256"] for row in catalog["examples"]}
                    if len(images) != len(catalog["examples"]):
                        raise ValueError("Library image identities differ")
                    library_cache.append((catalog, images))
                    return library_cache[0]
                semantic_records(semantic, semantic, "Prepared colorectal library", library_context)
            except (OSError, ValueError, KeyError, TypeError) as error:
                warning(str(error) if isinstance(error, CatalogLimit) else "Some library records could not be listed. Refresh to reconnect.")

        research = self.roots["research"]
        if research:
            try:
                collection = reader.read(research / "collection.json", research)
                comparison = research / "comparison-public.json"
                if comparison.exists():
                    try:
                        example_study = reader.read(comparison, research, sealed=False).get("demo", {}).get("study_id")
                    except (OSError, ValueError, TypeError):
                        warning("The prepared example-study link could not be verified.")
                for directory in reader.paths(research / "studies", research, directories=True):
                    if not ID.fullmatch(directory.name):
                        continue
                    def study_item(directory=directory):
                        study = reader.read(directory / "study.json", research)
                        if study.get("id") != directory.name or study.get("collection_sha256") != collection["sha256"] or study.get("prediction_sha256") != collection["prediction_sha256"]:
                            raise ValueError("Study source binding differs")
                        result = record_base(study, "study", collection["name"], collection["id"])
                        result.update(status="Saved tissue study", detail="", href=link("research", study=study["id"], step="overview"), example=study["id"] == example_study)
                        audits = []
                        for audit_directory in reader.paths(directory / "audits", research, directories=True):
                            if not ID.fullmatch(audit_directory.name):
                                continue
                            plan = reader.read(audit_directory / "plan.json", research)
                            if plan.get("id") != audit_directory.name or plan.get("study_sha256") != study["sha256"] or plan.get("prediction_sha256") != study["prediction_sha256"]:
                                raise ValueError("Saved audit study binding differs")
                            date(plan["created_at"])
                            annotations = reader.read(audit_directory / "annotations.json", research)
                            selected = {unit["id"] for unit in plan["units"]}
                            reviewed = [unit["id"] for unit in annotations["rows"]]
                            if annotations.get("plan_sha256") != plan["sha256"] or len(reviewed) != len(set(reviewed)) or not set(reviewed) <= selected or len(selected) != plan["budget"]:
                                raise ValueError("Saved audit annotation binding differs")
                            complete = len(reviewed) == plan["budget"]
                            has_result = False
                            if (audit_directory / "result.json").exists():
                                saved_result = reader.read(audit_directory / "result.json", research)
                                if saved_result.get("plan_sha256") != plan["sha256"] or saved_result.get("annotation_sha256") != annotations["sha256"]:
                                    raise ValueError("Saved audit result binding differs")
                                has_result = complete
                            audits.append({"id": plan["id"], "created_at": plan["created_at"], "reviewed": len(reviewed), "budget": plan["budget"], "complete": has_result})
                        if audits:
                            latest = max(audits, key=lambda audit: (date(audit["created_at"]), audit["id"]))
                            result.update(status="Audit complete · simulated review" if latest["complete"] else "Audit in progress · simulated review",
                                          detail=f"{latest['reviewed']} / {latest['budget']} windows reviewed",
                                          href=link("research", study=study["id"], audit=latest["id"], step="results" if latest["complete"] else "review"))
                            if latest["complete"]:
                                result["report_href"] = link("report", kind="study", id=study["id"], audit=latest["id"])
                        return result
                    append_record(directory, "study", collection.get("name", "Tissue study"), study_item)
            except (OSError, ValueError, KeyError, TypeError) as error:
                warning(str(error) if isinstance(error, CatalogLimit) else "Some tissue studies could not be listed. Refresh to reconnect.")

        query = q.strip().casefold()
        matching = [row for row in rows if (include_examples or not row["example"]) and (kind == "all" or row["kind"] == kind)
                    and (not query or query in " ".join(str(row.get(key) or "") for key in ("name", "source", "reviewer", "collection_name", "id")).casefold())]
        matching.sort(key=lambda row: (date(row["created_at"]) if row["created_at"] else float("-inf"), row["kind"], row["id"]), reverse=True)
        return {"items": matching[offset:offset + limit], "total": len(matching), "offset": offset, "limit": limit,
                "next_offset": offset + limit if offset + limit < len(matching) else None,
                "previous_offset": max(0, offset - limit) if offset else None,
                "complete": not warnings, "warnings": warnings,
                "unavailable_count": sum(not row["available"] for row in rows), "warning_count": len(warnings),
                "verification_scope": "Saved metadata and source bindings. Images, model weights and feature arrays are checked when opening the record; none are loaded by this catalog."}


def register_workspace(app, *, collections=None, semantic=None, research=None, collection_encoder=None):
    catalog = WorkspaceCatalog(collections, semantic, research, collection_encoder=collection_encoder)

    @app.get("/api/workspace")
    def saved_work(q: str = Query(default="", max_length=500),
                   kind: str = Query(default="all", pattern="^(all|collection|study|concept|description)$"),
                   offset: int = Query(default=0, ge=0, le=1000000), limit: int = Query(default=12, ge=1, le=100),
                   include_examples: bool = False):
        return catalog.listing(q=q, kind=kind, offset=offset, limit=limit, include_examples=include_examples)

    return catalog
