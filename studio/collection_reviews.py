"""Researcher observations kept apart from frozen collection predictions.

The history is append-only, with per-image hash chains and optimistic revision
checks. Reviewer identity is self-declared; these are inspection labels, not a
probability sample, training labels, or verified ground truth.
"""
from contextlib import contextmanager
import json
import os
import sqlite3
import tempfile

from .storage import digest, now

REVIEW_SCOPE = ("Researcher observations on selected images. These records do not change "
                "the model or its predictions and do not estimate accuracy for the collection. "
                "Reviewer names are self-declared, not authenticated identities.")
DISPOSITIONS = {"confirmed", "corrected", "unsure"}
FILTERS = {"all", "unreviewed", "reviewed", "confirmed", "corrected", "unsure"}


class ReviewConflict(ValueError):
    pass


class CollectionReviews:
    def __init__(self, directory):
        self.path = directory / "reviews.sqlite3"

    def initialize(self):
        """Called under the collection lock. Publish only a complete schema."""
        if self.path.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=".reviews-", dir=self.path.parent)
        os.close(fd)
        try:
            connection = sqlite3.connect(temporary)
            try:
                connection.executescript('''
                    PRAGMA journal_mode=DELETE;
                    PRAGMA synchronous=FULL;
                    CREATE TABLE history(ordinal INTEGER PRIMARY KEY, image_id TEXT NOT NULL,
                        revision INTEGER NOT NULL, request_id TEXT UNIQUE NOT NULL,
                        request_sha256 TEXT NOT NULL, payload TEXT NOT NULL,
                        UNIQUE(image_id,revision));
                    CREATE TABLE latest(image_id TEXT PRIMARY KEY, disposition TEXT NOT NULL,
                        revision INTEGER NOT NULL, payload TEXT NOT NULL);
                    CREATE INDEX latest_disposition ON latest(disposition,image_id);
                ''')
                connection.commit()
            finally:
                connection.close()
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(f"file:{self.path}?mode=rw", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def checked(payload):
        row = json.loads(payload)
        if row.get("sha256") != digest({key: value for key, value in row.items() if key != "sha256"}):
            raise ValueError("A saved review failed its integrity check")
        return row

    def summary(self):
        result = {"reviewed": 0, "confirmed": 0, "corrected": 0, "unsure": 0, "revisions": 0}
        if not self.path.exists():
            return result
        with self.connection() as connection:
            for row in connection.execute("SELECT disposition,COUNT(*) AS n FROM latest GROUP BY disposition"):
                result[row["disposition"]] = row["n"]
                result["reviewed"] += row["n"]
            result["revisions"] = connection.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        return result

    def image(self, image_id):
        if not self.path.exists():
            return None, []
        with self.connection() as connection:
            history = [self.checked(row[0]) for row in connection.execute(
                "SELECT payload FROM history WHERE image_id=? ORDER BY revision", (image_id,))]
        previous = None
        for revision, row in enumerate(history, 1):
            if row["revision"] != revision or row["previous_sha256"] != previous:
                raise ValueError("Saved review history is incomplete or inconsistent")
            previous = row["sha256"]
        return (history[-1] if history else None), history

    def save(self, request, binding):
        """Called under the collection lock; retries are idempotent by request ID."""
        self.initialize()
        request_hash = digest({"image_id": binding["image_id"], **request})
        with self.connection() as connection:
            previous_request = connection.execute("SELECT request_sha256,payload FROM history WHERE request_id=?",
                                                  (request["client_request_id"],)).fetchone()
            if previous_request:
                if previous_request[0] != request_hash:
                    raise ReviewConflict("This save identity belongs to a different review. Reload the image before saving.")
                return self.checked(previous_request[1])
            latest, _ = self.image(binding["image_id"])
            revision = latest["revision"] if latest else 0
            if request["expected_revision"] != revision:
                raise ReviewConflict("Another review was saved for this image. Reload the image to see it before saving your revision.")
            event = {"schema": "collection-image-review-v1", **binding,
                     "revision": revision + 1, "previous_sha256": latest["sha256"] if latest else None,
                     "created_at": now(), "reviewer": request["reviewer"],
                     "reviewer_identity": "self_declared", "disposition": request["disposition"],
                     "label": request["label"], "note": request["note"],
                     "selection": "researcher_selected", "role": "inspection",
                     "client_request_id": request["client_request_id"]}
            event["sha256"] = digest(event)
            payload = json.dumps(event, separators=(",", ":"), allow_nan=False)
            connection.execute("INSERT INTO history(image_id,revision,request_id,request_sha256,payload) VALUES(?,?,?,?,?)",
                (event["image_id"], event["revision"], request["client_request_id"], request_hash, payload))
            connection.execute("INSERT INTO latest(image_id,disposition,revision,payload) VALUES(?,?,?,?) "
                "ON CONFLICT(image_id) DO UPDATE SET disposition=excluded.disposition,revision=excluded.revision,payload=excluded.payload",
                (event["image_id"], event["disposition"], event["revision"], payload))
        return event

    def page(self, index, offset, limit, label=None, uncertain=None, q=None, review="all"):
        if review not in FILTERS:
            raise ValueError("Unknown image review filter")
        if not self.path.exists():
            if review not in {"all", "unreviewed"}:
                return [], 0
            rows, total = index.page(offset, limit, label, uncertain, q)
            return [(row, prediction, None) for row, prediction in rows], total
        clauses, parameters = [], []
        if label is not None:
            clauses.append("p.label=?")
            parameters.append(label)
        if uncertain is not None:
            clauses.append("p.uncertain=?")
            parameters.append(int(uncertain))
        if q:
            clauses.append("instr(lower(i.filename),lower(?))>0")
            parameters.append(q)
        if review == "unreviewed":
            clauses.append("r.image_id IS NULL")
        elif review == "reviewed":
            clauses.append("r.image_id IS NOT NULL")
        elif review in DISPOSITIONS:
            clauses.append("r.disposition=?")
            parameters.append(review)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        base = " FROM images i LEFT JOIN predictions p ON p.id=i.id LEFT JOIN reviews.latest r ON r.image_id=i.id" + where
        with index.connection() as connection:
            connection.execute("ATTACH DATABASE ? AS reviews", (str(self.path),))
            # One read transaction makes count and rows describe the same snapshot.
            connection.execute("BEGIN")
            total = connection.execute("SELECT COUNT(*)" + base, parameters).fetchone()[0]
            rows = connection.execute("SELECT i.payload,p.payload,r.payload" + base + " ORDER BY i.ordinal LIMIT ? OFFSET ?",
                                      parameters + [limit, offset]).fetchall()
        return [(json.loads(row[0]), json.loads(row[1]) if row[1] else None,
                 self.checked(row[2]) if row[2] else None) for row in rows], total

    def export(self):
        if not self.path.exists():
            return {"latest": [], "history": []}
        with self.connection() as connection:
            connection.execute("BEGIN")
            latest = [self.checked(row[0]) for row in connection.execute("SELECT payload FROM latest ORDER BY image_id")]
            history = [self.checked(row[0]) for row in connection.execute("SELECT payload FROM history ORDER BY ordinal")]
        return {"latest": latest, "history": history}
