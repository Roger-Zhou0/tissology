"""Durable indexed collection metadata; immutable JSON remains the audit artifact.

DELETE journaling avoids WAL's shared-memory requirements on a shared filesystem.
Writes use SQLite transactions and collection writer locks. Index publication is
atomic; legacy frozen collections are indexed without rewriting their artifacts.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .storage import digest, read


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def checked(path):
    artifact = read(path)
    if artifact.get("sha256") != digest({k: v for k, v in artifact.items() if k != "sha256"}):
        raise ValueError(f"Frozen collection artifact failed integrity check: {path.name}")
    return artifact


def identity(path):
    if not path.exists():
        return None
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


class CollectionIndex:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "index.sqlite3"
        self._identities = None
        self.ensure()

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(f"file:{self.path}?mode=rw", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def schema(connection):
        connection.executescript('''
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE images(ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                image_sha256 TEXT UNIQUE NOT NULL, filename TEXT NOT NULL,
                payload TEXT NOT NULL);
            CREATE INDEX image_filename ON images(filename);
            CREATE TABLE issues(ordinal INTEGER PRIMARY KEY, kind TEXT NOT NULL,
                filename TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX issue_kind ON issues(kind, ordinal);
            CREATE INDEX issue_filename ON issues(filename);
            CREATE TABLE predictions(id TEXT PRIMARY KEY REFERENCES images(id),
                label TEXT NOT NULL, uncertain INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX prediction_label ON predictions(label,id);
            CREATE INDEX prediction_uncertain ON predictions(uncertain,id);
            CREATE INDEX prediction_label_uncertain ON predictions(label,uncertain,id);
            CREATE TABLE uploads(upload_key TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,
                filename TEXT NOT NULL, source_bytes INTEGER NOT NULL,
                payload TEXT NOT NULL);
        ''')

    @staticmethod
    def get_state(connection, key, default=None):
        row = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    @staticmethod
    def put_state(connection, key, value):
        connection.execute("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, encode(value)))

    def owned_artifact(self, name):
        artifact = checked(self.directory / name)
        key = "id" if name == "collection.json" else "collection_id"
        if artifact.get(key) != self.directory.name:
            raise ValueError(f"Frozen collection artifact belongs to a different collection: {name}")
        return artifact

    def initialize(self):
        """Build once from existing artifacts, or the brand-new empty intake."""
        metadata = self.owned_artifact("collection.json")
        manifest_path = self.directory / "manifest.json"
        manifest = self.owned_artifact("manifest.json") if manifest_path.exists() else None
        intake = read(self.directory / "intake.json")
        if metadata.get("metadata_storage") == "indexed-v2" and not manifest and intake.get("index_initialized"):
            raise ValueError("Collection metadata index is missing; restore its backup before resuming this import")
        images = manifest["images"] if manifest else intake["images"]
        errors = manifest.get("exclusions", []) if manifest else intake.get("errors", [])
        duplicates = manifest.get("duplicates", []) if manifest else intake.get("duplicates", [])
        stats = {"image_count": len(images), "rejected_count": len(errors), "duplicate_count": len(duplicates),
                 "attempted_files": intake.get("attempted_files", len(images)+len(errors)+len(duplicates)),
                 "imported_bytes": sum(row.get("source_bytes", 0) for row in images),
                 "stored_bytes": intake.get("stored_bytes", 0)}
        temporary = self.directory / (".index-build-" + uuid.uuid4().hex)
        connection = sqlite3.connect(temporary)
        try:
            self.schema(connection)
            connection.executemany("INSERT INTO images(id,image_sha256,filename,payload) VALUES(?,?,?,?)",
                                   [(row["id"], row["image_sha256"], row["filename"], encode(row)) for row in images])
            for kind, entries in (("errors", errors), ("duplicates", duplicates)):
                connection.executemany("INSERT INTO issues(kind,filename,payload) VALUES(?,?,?)",
                                       [(kind, row.get("filename", ""), encode(row)) for row in entries])
            self.put_state(connection, "stats", stats)
            self.put_state(connection, "bindings", {"collection.json": metadata["sha256"],
                **({"manifest.json": manifest["sha256"]} if manifest else {})})
            self.put_state(connection, "identities", {})
            connection.commit()
            connection.close()
            os.replace(temporary, self.path)
        finally:
            connection.close()
            if temporary.exists():
                temporary.unlink()

    def ensure(self):
        paths = ["collection.json", "manifest.json", "predictions.json"]
        identities = {name: identity(self.directory / name) for name in paths}
        if self.path.exists() and identities == self._identities:
            return self
        with (self.directory / ".index.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            try:
                if not self.path.exists():
                    self.initialize()
                with self.connection() as connection:
                    saved = self.get_state(connection, "identities", {})
                    bindings = self.get_state(connection, "bindings", {})
                    for name in paths:
                        if identities[name] == saved.get(name):
                            continue
                        if identities[name] is None:
                            if name in bindings:
                                raise ValueError(f"Frozen collection artifact is missing: {name}")
                            continue
                        artifact = self.owned_artifact(name)
                        if name in bindings and bindings[name] != artifact["sha256"]:
                            raise ValueError(f"Frozen collection artifact changed: {name}")
                        if name == "predictions.json" and name not in bindings:
                            if artifact.get("manifest_sha256") != bindings.get("manifest.json"):
                                raise ValueError("Prediction artifact does not match the frozen collection")
                            predictions = artifact["predictions"]
                            ids = [row["id"] for row in predictions]
                            actual = {row[0] for row in connection.execute("SELECT id FROM images")}
                            if len(ids) != len(set(ids)) or set(ids) != actual:
                                raise ValueError("Prediction artifact does not cover the indexed images exactly once")
                            connection.executemany("INSERT INTO predictions(id,label,uncertain,payload) VALUES(?,?,?,?)",
                                [(row["id"], row["predicted_label"], int(row["uncertain"]), encode(row)) for row in predictions])
                            self.put_state(connection, "prediction", {key: artifact.get(key) for key in
                                ("sha256", "model", "score_type", "summary", "created_at")})
                        bindings[name] = artifact["sha256"]
                    self.put_state(connection, "bindings", bindings)
                    self.put_state(connection, "identities", identities)
                self._identities = identities
            finally:
                fcntl.flock(guard, fcntl.LOCK_UN)
        return self

    def statistics(self):
        with self.connection() as connection:
            return self.get_state(connection, "stats")

    def metadata(self):
        with self.connection() as connection:
            return {"stats": self.get_state(connection, "stats"),
                    "prediction": self.get_state(connection, "prediction", {}),
                    "bindings": self.get_state(connection, "bindings", {})}

    def image(self, identifier):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM images WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise FileNotFoundError("Unknown collection image")
            return json.loads(row[0])

    def prediction(self, identifier):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM predictions WHERE id=?", (identifier,)).fetchone()
            return json.loads(row[0]) if row else None

    def seen_filename(self, filename):
        with self.connection() as connection:
            return (connection.execute("SELECT 1 FROM images WHERE filename=? LIMIT 1", (filename,)).fetchone() is not None or
                    connection.execute("SELECT 1 FROM issues WHERE filename=? LIMIT 1", (filename,)).fetchone() is not None)

    def issues(self, kind, limit=None):
        with self.connection() as connection:
            query = "SELECT payload FROM issues WHERE kind=? ORDER BY ordinal"
            params = [kind]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            return [json.loads(row[0]) for row in connection.execute(query, params)]

    def intake(self):
        with self.connection() as connection:
            images = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM images ORDER BY ordinal")]
            stats = self.get_state(connection, "stats")
        return {"images": images, "errors": self.issues("errors"), "duplicates": self.issues("duplicates"),
                "imported_bytes": stats["imported_bytes"], "stored_bytes": stats["stored_bytes"],
                "attempted_files": stats["attempted_files"], "index_initialized": True}

    def page(self, offset, limit, label=None, uncertain=None, q=None):
        clauses, parameters = [], []
        if label is not None:
            clauses.append("p.label=?")
            parameters.append(label)
        if uncertain is not None:
            clauses.append("p.uncertain=?")
            parameters.append(int(uncertain))
        if q:
            clauses.append("instr(lower(i.filename), lower(?)) > 0")
            parameters.append(q)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        base = " FROM images i LEFT JOIN predictions p ON p.id=i.id" + where
        with self.connection() as connection:
            total = connection.execute("SELECT COUNT(*)" + base, parameters).fetchone()[0]
            rows = connection.execute("SELECT i.payload,p.payload" + base + " ORDER BY i.ordinal LIMIT ? OFFSET ?",
                                      parameters + [limit, offset]).fetchall()
            return [(json.loads(row[0]), json.loads(row[1]) if row[1] else None) for row in rows], total

    def receipt(self, key):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM uploads WHERE upload_key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def receipts(self, offset=0, limit=500):
        with self.connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
            rows = connection.execute("SELECT payload FROM uploads ORDER BY rowid LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return {"uploads": [json.loads(row[0]) for row in rows], "total": total, "offset": offset, "limit": limit}
