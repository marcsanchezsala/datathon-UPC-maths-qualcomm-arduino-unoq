"""A keyword index with the same interface as rag.Index, for the laptop.

It needs no model server, so you can develop and measure the tutor without
the board. Scores are cosine similarities of word counts: they are lower than
bge-small's, so use a lower threshold (about 0.3) when you use this index.
"""
import math
import re
from collections import Counter

from .llm import format_problem

_WORD = re.compile(r"[a-zà-ÿ]+|\d+", re.I)


def _vec(problem):
    words = [w.lower() for w in _WORD.findall(format_problem(problem))]
    return Counter(w for w in words if len(w) > 2 or w.isdigit())


class LexicalIndex:
    def __init__(self, problems):
        self.rows = []
        for p in problems:
            v = _vec(p)
            self.rows.append((p, v, math.sqrt(sum(c * c for c in v.values())) or 1.0))

    def search(self, problem, k=1, exclude_id=None):
        q = _vec(problem)
        qn = math.sqrt(sum(c * c for c in q.values())) or 1.0
        scored = []
        for p, v, n in self.rows:
            if exclude_id and p["id"] == exclude_id:
                continue
            dot = sum(c * v.get(w, 0) for w, c in q.items())
            scored.append((dot / (qn * n), p))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]
