"""A retrieval index in SQLite: one solved M3Kang problem per row."""
import json
import re
import sqlite3
from array import array

from .embed import embed

SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    key TEXT PRIMARY KEY,   -- id|language
    problem TEXT NOT NULL,  -- the full problem as JSON
    vec BLOB NOT NULL       -- float32 vector
)"""


def embed_text(problem):
    """Question stem without the exam number or OCR option boilerplate."""
    return re.sub(r"^\s*\d+\.\s*", "", problem["text"].strip())


def build(db_path, problems, batch=16):
    db = sqlite3.connect(db_path)
    db.execute(SCHEMA)
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        vectors = embed([embed_text(p) for p in chunk])
        db.executemany("INSERT OR REPLACE INTO problems VALUES (?, ?, ?)", [
            (f"{p['id']}|{p['language']}", json.dumps(p, ensure_ascii=False), array("f", v).tobytes())
            for p, v in zip(chunk, vectors)])
        db.commit()
        yield min(i + batch, len(problems))
    db.close()


class Index:
    """Loads every vector into memory once, so each search is a plain loop."""

    def __init__(self, db_path):
        db = sqlite3.connect(db_path)
        self.rows = []
        for key, problem, blob in db.execute("SELECT key, problem, vec FROM problems"):
            vec = array("f")
            vec.frombytes(blob)
            self.rows.append((key, json.loads(problem), vec))
        db.close()
        self.languages = {p.get("language") for _, p, _ in self.rows}

    def search(self, problem, k=1, exclude_id=None, min_score=0.0, only_with_options=True):
        """Return up to k (score, problem) pairs, most similar first.

        exclude_id skips the same problem in other languages, so validation
        never retrieves its own translation.
        """
        if not self.rows or k <= 0:
            return []
        query = embed([embed_text(problem)], query=True)[0]
        lang = problem.get("language") if problem.get("language") in self.languages else None
        scored = []
        for _key, stored, vec in self.rows:
            if exclude_id and stored["id"] == exclude_id:
                continue
            if only_with_options and not stored.get("options"):
                continue
            if lang and stored.get("language") != lang:
                continue
            score = sum(a * b for a, b in zip(query, vec))
            if score < min_score:
                continue
            scored.append((score, stored))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        seen_ids = set()
        for score, stored in scored:
            sid = stored["id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            results.append((score, stored))
            if len(results) == k:
                break
        return results
