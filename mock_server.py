#!/usr/bin/env python3
"""Mock server for testing the M3Kang explorer on a laptop, without the board.

It mimics explorer/server.py: the same endpoints and the same JSON shapes, but
with three made-up problems and canned hints. Nothing is downloaded and no
model runs.

    python3 mock_server.py            # from the repository root
    open http://127.0.0.1:8000

Unlike a flat mock, it answers per problem and per hint level, honours the
teaching-mode `reveal` flag, and fills every field the page reads.
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Fake data ──────────────────────────────────────────────────────────

PROBLEMS = {
    "lvl-0_2015_14": {
        "level": 0, "year": 2015, "label": "C",
        "text": {
            "eng": "Joan has built a cube using 27 small black and white cubes. The cubes at the corners are black, "
                   "and two small cubes of the same color never share a face. How many small white cubes has Joan used?",
            "cat": "Joan ha construït un cub amb 27 cubs petits blancs i negres. Els cubs dels vèrtexs són negres i "
                   "dos cubs petits del mateix color mai no comparteixen cap cara. Quants cubs petits blancs ha fet servir?",
            "esp": "Joan ha construido un cubo con 27 cubos pequeños blancos y negros. Los cubos de las esquinas son negros "
                   "y dos cubos pequeños del mismo color nunca comparten una cara. ¿Cuántos cubos blancos ha usado?",
        },
        "options": {"A": "10", "B": "12", "C": "13", "D": "14", "E": "15"},
        "hints": [
            "A similar, easier problem: a 2x2x2 cube coloured like a chessboard has 4 black and 4 white cubes.",
            "This is a counting problem. Split the cube into its 3 layers and count each layer separately.",
            "Layers 1 and 3 look like a 3x3 chessboard with black corners. What does the middle layer look like?",
        ],
        "solution": "Layers 1 and 3 have black corners: 5 black and 4 white each. The middle layer is the opposite: "
                    "4 black and 5 white. White cubes = 4 + 5 + 4 = 13.",
    },
    "lvl-1_2016_20": {
        "level": 1, "year": 2016, "label": "B",
        "text": {
            "eng": "A snail climbs 3 m up a wall each day and slides 2 m down each night. The wall is 10 m high. "
                   "On which day does the snail reach the top?",
            "cat": "Un cargol puja 3 m per una paret cada dia i rellisca 2 m cada nit. La paret fa 10 m d'alçada. "
                   "Quin dia arriba al capdamunt?",
            "esp": "Un caracol sube 3 m por una pared cada día y resbala 2 m cada noche. La pared mide 10 m. "
                   "¿Qué día llega a la cima?",
        },
        "options": {"A": "7", "B": "8", "C": "9", "D": "10", "E": "12"},
        "hints": [
            "A similar, easier problem: a 5 m wall with the same snail. When does it arrive?",
            "This is a pattern problem. Find the net gain per full day, then look at the last day on its own.",
            "After the first day and night the snail is 1 m up. Once it reaches the top during the day, it doesn't slide back.",
        ],
        "solution": "Each full day gains 1 m, so after 7 days and nights it is 7 m up. On day 8 it climbs 3 m and reaches 10 m "
                    "before the night. The answer is day 8.",
    },
    "lvl-2_2017_19": {
        "level": 2, "year": 2017, "label": "A",
        "text": {
            "eng": "What is the sum of the digits of the number 2^10 × 5^8?",
            "cat": "Quina és la suma de les xifres del nombre 2^10 × 5^8?",
            "esp": "¿Cuál es la suma de las cifras del número 2^10 × 5^8?",
        },
        "options": {"A": "4", "B": "2", "C": "8", "D": "10", "E": "13"},
        "hints": [
            "A similar, easier problem: what is 2^3 × 5^3? Look at the result.",
            "This is a powers problem. Pair each 2 with a 5 to make a 10, and see what is left over.",
            "2^10 × 5^8 = 2^2 × (2 × 5)^8. What is 2^2 × 10^8, written out?",
        ],
        "solution": "2^10 × 5^8 = 2^2 × 10^8 = 4 × 100000000 = 400000000. Its digits add up to 4.",
    },
}

LANGS = ["eng", "cat", "esp"]


def key_of(pid, lang):
    return f"{pid}_{lang}"


def parse_key(key):
    pid, _, lang = key.rpartition("_")
    return (pid, lang) if pid in PROBLEMS and lang in LANGS else (None, None)


def public(pid, lang):
    p = PROBLEMS[pid]
    return {
        "key": key_of(pid, lang), "id": pid, "language": lang, "level": p["level"], "year": p["year"],
        "text": p["text"][lang], "options": p["options"], "image": False,
    }


def leaks(pid, text):
    """Same idea as the real server: a hint must not contain the correct value."""
    p = PROBLEMS[pid]
    value = p["options"][p["label"]]
    return len(value) >= 2 and re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", text) is not None


# ── HTTP ───────────────────────────────────────────────────────────────

class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")

    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}

        if url.path == "/":
            for candidate in (os.path.join("explorer", "index.html"), os.path.join(HERE, "index.html"),
                              os.path.join(HERE, "explorer", "index.html")):
                if os.path.exists(candidate):
                    with open(candidate, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            return self.send_json({"error": "index.html not found"}, 404)

        if url.path == "/api/info":
            return self.send_json({
                "input": "mock", "count": len(PROBLEMS) * len(LANGS), "languages": LANGS,
                "levels": sorted({p["level"] for p in PROBLEMS.values()}),
                "images": 0, "index": 0, "llm": False, "embed": False, "vision": False, "time_limit": 60.0,
            })

        if url.path == "/api/problems":
            return self.send_json([
                {"key": key_of(pid, lang), "id": pid, "language": lang, "level": p["level"], "options": True}
                for pid, p in PROBLEMS.items() for lang in LANGS
            ])

        if url.path == "/api/problem":
            pid, lang = parse_key(params.get("key", ""))
            return self.send_json(public(pid, lang)) if pid else self.send_json({"error": "unknown problem"}, 404)

        if url.path == "/api/similar":
            return self.send_json([])

        return self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.send_json({"error": "invalid JSON"}, 400)

        pid, lang = parse_key(body.get("key", ""))
        if not pid:
            return self.send_json({"error": "unknown problem"}, 404)
        p = PROBLEMS[pid]
        path = self.path

        if path == "/api/check":
            choice = body.get("choice")
            if choice not in list("ABCDE"):
                return self.send_json({"error": "bad request"}, 400)
            correct = choice == p["label"]
            reveal = bool(body.get("reveal", True))
            return self.send_json({"correct": correct, "label": p["label"] if (correct or reveal) else None})

        if path == "/api/reveal":
            return self.send_json({"label": p["label"]})

        if path == "/api/hint":
            try:
                level = max(1, min(int(body.get("type", 1)), 3))
            except (TypeError, ValueError):
                level = 1
            text = p["hints"][level - 1]
            if leaks(pid, text):
                return self.send_json({"type": "hint", "level": level, "withheld": True,
                                       "text": "This hint would give the answer away, so it was withheld."})
            return self.send_json({"type": "hint", "level": level, "text": text})

        if path == "/api/solution":
            return self.send_json({"type": "solution", "level": 4, "text": p["solution"]})

        if path == "/api/reasoning":
            text = (body.get("reasoning") or "").strip()
            if not text:
                return self.send_json({"type": "rule_based", "text": "Write your steps first."})
            if p["options"][p["label"]] in text:
                return self.send_json({"type": "correction",
                                       "text": "Your final number matches. Can you explain why each step is true?",
                                       "detected_errors": []})
            return self.send_json({"type": "correction",
                                   "text": "Check each step again: where could a count be off by one?",
                                   "detected_errors": ["off_by_one"]})

        if path == "/api/translate":
            target = body.get("language", "")
            if target in LANGS:
                return self.send_json({"found": True, "id": key_of(pid, target), "language": target,
                                       "level": p["level"], "text": p["text"][target]})
            return self.send_json({"found": False})

        if path == "/api/ask":
            rag = int(body.get("rag", 0))
            prompt = "--- user ---\n" + p["text"]["eng"] + "\n" + "\n".join(f"{k}) {v}" for k, v in p["options"].items())
            return self.send_json({
                "answer": "D", "correct": "D" == p["label"], "label": p["label"], "seconds": 5.5,
                "status": "ok", "prompt_tokens": len(prompt.split()) * 2, "prompt": prompt, "examples": [],
                "rag": rag,
            })

        if path == "/api/adaptive":
            return self.send_json({"id": None})

        if path == "/api/counterexample":
            return self.send_json({"found": False, "detected_errors": []})

        if path == "/api/hint-reset":
            return self.send_json({"ok": True})

        return self.send_json({"error": "bad request"}, 400)


if __name__ == "__main__":
    print("\nM3Kang mock server (fake data and fake models)")
    print("Open http://127.0.0.1:8000 in your browser (Ctrl+C to stop)\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8000), MockHandler).serve_forever()