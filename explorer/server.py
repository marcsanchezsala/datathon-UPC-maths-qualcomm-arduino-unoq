#!/usr/bin/env python3
"""M3Kang Teaching Copilot Server - Edge Math Companion.

Serves the teaching interface for the Math on the Edge datathon.
Standard library only. Runs on the Arduino UNO Q.

    python3 explorer/server.py --input data/val.jsonl --index data/index.db --images data/images
"""
import argparse
import base64
import json
import os
import sys
import traceback
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import get  # noqa: E402
from copilot.config import CONFIG, ROOT  # noqa: E402
from copilot.llm import build_messages  # noqa: E402
from copilot.pipeline import solve  # noqa: E402
from copilot.rag import Index  # noqa: E402
from copilot.tutor import (  # noqa: E402
    tutor_generate,
    tutor_student_correction,
    tutor_adaptive_next,
    find_rag_examples,
    find_counterexample,
    detect_typical_errors,
    translate_problem,
    _build_problem_text,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LLM = f"http://127.0.0.1:{CONFIG['LLM_PORT']}"
EMBED = f"http://127.0.0.1:{CONFIG['EMBED_PORT']}"
VISION = f"http://127.0.0.1:{CONFIG['VISION_PORT']}"
VISION_TIME_LIMIT = 300
MAX_HINT_LEVEL = 3  # rungs 1-3 are hints; rung 4 (the solution) only via /api/solution


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/val.jsonl", help="problems to explore")
    p.add_argument("--images", default="data/images", help="folder from prepare_m3kang.py --save-images")
    p.add_argument("--index", default="data/index.db", help="retrieval index from build_index.py")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open on network")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def key_of(problem):
    return f"{problem['id']}_{problem['language']}"


def is_up(base):
    try:
        return get(base + "/health", timeout=2).get("status") == "ok"
    except (OSError, ValueError):
        return False


def sees_images(base):
    try:
        return bool(get(base + "/props", timeout=2).get("modalities", {}).get("vision"))
    except (OSError, ValueError):
        return False


def public(problem, images):
    """A problem without its label, so the page can't give the answer away."""
    shown = {k: v for k, v in problem.items() if k != "label"}
    shown["key"] = key_of(problem)
    shown["image"] = os.path.exists(os.path.join(images, key_of(problem) + ".png"))
    return shown


def printable(messages):
    """The prompt as plain text."""
    lines = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = "\n".join(
                "[image]" if part["type"] == "image_url" else part["text"] for part in content
            )
        lines.append(f"--- {m['role']} ---\n{content}")
    return "\n".join(lines)


class App:
    def __init__(self, args):
        os.chdir(ROOT)
        self.images = args.images
        with open(args.input, encoding="utf-8") as f:
            self.problems = [json.loads(line) for line in f if line.strip()]
        self.by_key = {key_of(p): p for p in self.problems}
        self.index = Index(args.index) if os.path.exists(args.index) else None
        self.input = args.input
        # Teaching state
        self.hint_state = {}  # key -> {level, history, reasoning, ...}
        self.student_profile = {}  # key -> {correct, wrong, level}
        # Images the page may ask for
        self.image_keys = set(self.by_key)
        if self.index:
            self.image_keys |= {row[0].replace("|", "_") for row in self.index.rows}

    def info(self):
        return {
            "input": self.input,
            "count": len(self.problems),
            "languages": sorted({p["language"] for p in self.problems}),
            "levels": sorted({p["level"] for p in self.problems}),
            "images": sum(
                os.path.exists(os.path.join(self.images, k + ".png")) for k in self.by_key
            ),
            "index": len(self.index.rows) if self.index else 0,
            "llm": is_up(LLM),
            "embed": is_up(EMBED),
            "vision": is_up(VISION) and sees_images(VISION),
            "time_limit": float(CONFIG["TIME_LIMIT_SECONDS"]),
        }

    def listing(self):
        return [
            {
                "key": key_of(p),
                "id": p["id"],
                "language": p["language"],
                "level": p["level"],
                "options": bool(p.get("options")),
            }
            for p in self.problems
        ]

    def image_b64(self, key):
        with open(os.path.join(self.images, key + ".png"), "rb") as f:
            return base64.b64encode(f.read()).decode()

    def check(self, key, choice):
        label = self.by_key[key]["label"]
        correct = choice == label
        state = self._get_or_create_state(key)
        state["last_correct"] = correct
        if correct:
            state["score_correct"] = state.get("score_correct", 0) + 1
        else:
            state["score_wrong"] = state.get("score_wrong", 0) + 1
        return {"correct": correct, "label": label}

    def _get_or_create_state(self, key):
        if key not in self.hint_state:
            self.hint_state[key] = {
                "level": 1,
                "history": [],
                "reasoning": "",
                "correct": False,
                "score_correct": 0,
                "score_wrong": 0,
            }
        return self.hint_state[key]

    # ── Teaching: hint ladder (3 hints, then the solution on request) ────

    def get_hint(self, key, hint_type=None):
        """Generate a hint. The page sends the rung it wants (1-3).

        The ladder is capped at MAX_HINT_LEVEL: the solution (rung 4) is never
        returned here, only by get_solution, when the student asks for it.
        """
        problem = self.by_key[key]
        state = self._get_or_create_state(key)
        try:
            level = int(hint_type) if hint_type is not None else state["level"]
        except (TypeError, ValueError):
            level = state["level"]
        level = max(1, min(level, MAX_HINT_LEVEL))
        result = tutor_generate(problem, index=self.index, hint_level=level, timeout=120)
        state["level"] = min(level + 1, MAX_HINT_LEVEL)
        state["history"].append(result)
        return result

    def get_solution(self, key):
        """Force the solution (level 4)."""
        problem = self.by_key[key]
        result = tutor_generate(problem, index=self.index, hint_level=4, timeout=120)
        state = self._get_or_create_state(key)
        state["history"].append(result)
        state["level"] = 4
        return result

    def student_reasoning(self, key, reasoning):
        """Student writes their thinking → correction + counterexamples."""
        problem = self.by_key[key]
        state = self._get_or_create_state(key)
        state["reasoning"] = reasoning

        result = tutor_student_correction(problem, reasoning, index=self.index, timeout=120)
        state["history"].append(result)
        return result

    def adaptive_next(self, key, is_correct):
        """Find the next problem: harder if correct, easier if wrong."""
        problem = self.by_key[key]
        next_prob = tutor_adaptive_next(problem, is_correct, self.index)
        if next_prob:
            return {
                "id": next_prob["id"],
                "level": next_prob["level"],
                "suggestion": next_prob["text"][:100] + "...",
            }
        return {"id": None}

    def get_counterexample(self, key, reasoning):
        """Find a counterexample from similar problems."""
        problem = self.by_key[key]
        counterexamples = find_counterexample(problem, reasoning, self.index, k=3)
        if counterexamples:
            _score, p, wrong_letter, wrong_value = counterexamples[0]
            return {
                "found": True,
                "counterexample": {
                    "id": p["id"],
                    "score": round(_score, 3),
                    "text": p["text"][:150],
                    "wrong_option": wrong_letter,
                    "wrong_value": wrong_value,
                    "correct_letter": p["label"],
                },
                "detected_errors": detect_typical_errors(reasoning, ""),
            }
        return {"found": False, "detected_errors": detect_typical_errors(reasoning, "")}

    def translate(self, key, target_lang):
        """Translate a problem with the multilingual bridge.

        M3Kang has the same problem id in every language. The validation file
        holds all the languages of a validation problem, so look there first.
        The retrieval index only has training problems, so it can't find them.
        """
        problem = self.by_key[key]
        other = self.by_key.get(f"{problem['id']}_{target_lang}")
        if other:
            return {
                "found": True,
                "id": other["id"],
                "text": other["text"],
                "language": other["language"],
                "level": other["level"],
            }
        try:
            translated = translate_problem(problem, target_lang, self.index)
        except Exception:  # noqa: BLE001 - fall back to "not found" instead of a 500
            traceback.print_exc()
            translated = None
        if translated:
            return {
                "found": True,
                "id": translated["id"],
                "text": translated["text"],
                "language": translated["language"],
                "level": translated["level"],
            }
        return {"found": False}

    def clear_hint(self, key):
        """Reset the hint state for a problem."""
        if key in self.hint_state:
            del self.hint_state[key]

    def ask(self, key, rag, vision):
        """Standard model ask (same as before)."""
        problem = self.by_key[key]
        image = self.image_b64(key) if vision else None
        result = solve(
            problem,
            index=self.index if rag else None,
            k=rag,
            exclude_id=problem["id"],
            image_b64=image,
            llm_url=(VISION if vision else LLM) + "/v1/chat/completions",
            time_limit=VISION_TIME_LIMIT if vision else float(CONFIG["TIME_LIMIT_SECONDS"]),
        )
        examples = [p for _score, p in result["examples"]]
        return {
            "answer": result["answer"],
            "status": result["status"],
            "seconds": result["seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "correct": result["answer"] == problem["label"],
            "label": problem["label"],
            "prompt": printable(build_messages(problem, examples, image)),
            "examples": [
                {"score": round(s, 3), **public(p, self.images), "label": p["label"]}
                for s, p in result["examples"]
            ],
        }

    def similar(self, key, k):
        """Get similar problems (same as before)."""
        if not self.index:
            return []
        problem = self.by_key[key]
        hits = self.index.search(problem, k=k, exclude_id=problem["id"])
        return [
            {"score": round(s, 3), **public(p, self.images), "label": p["label"]}
            for s, p in hits
        ]


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("  " + fmt % args + "\n")

        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            params = {k: v[0] for k, v in parse_qs(query).items()}
            try:
                if path == "/":
                    with open(os.path.join(HERE, "index.html"), "rb") as f:
                        return self.send(200, f.read(), "text/html; charset=utf-8")
                if path == "/api/info":
                    return self.send(200, app.info())
                if path == "/api/problems":
                    return self.send(200, app.listing())
                if path == "/api/problem" and params.get("key") in app.by_key:
                    return self.send(200, public(app.by_key[params["key"]], app.images))
                if path == "/api/similar" and params.get("key") in app.by_key:
                    return self.send(200, app.similar(params["key"], int(params.get("k", 3))))
                if path.startswith("/image/"):
                    key = path[len("/image/"):]
                    file = os.path.join(app.images, key + ".png")
                    if key in app.image_keys and os.path.exists(file):
                        with open(file, "rb") as f:
                            return self.send(200, f.read(), "image/png")
            except (OSError, urllib.error.URLError) as error:
                return self.send(502, {"error": f"model server: {error}"})
            except Exception as error:  # noqa: BLE001 - show the problem instead of hanging
                traceback.print_exc()
                return self.send(500, {"error": f"{type(error).__name__}: {error}"})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
                )
            except (ValueError, UnicodeDecodeError):
                return self.send(400, {"error": "invalid JSON"})
            key = body.get("key")
            if key not in app.by_key:
                return self.send(404, {"error": "unknown problem"})
            try:
                # Existing endpoints
                if self.path == "/api/check" and body.get("choice") in list("ABCDE"):
                    return self.send(200, app.check(key, body["choice"]))
                if self.path == "/api/ask":
                    return self.send(
                        200,
                        app.ask(
                            key,
                            int(body.get("rag", 0)),
                            bool(body.get("vision")),
                        ),
                    )

                # ── Teaching endpoints ──────────────────────────────
                if self.path == "/api/hint":
                    hint_type = body.get("type", None)  # rung 1-3 chosen by the page
                    return self.send(200, app.get_hint(key, hint_type=hint_type))

                if self.path == "/api/solution":
                    return self.send(200, app.get_solution(key))

                if self.path == "/api/reasoning":
                    reasoning = body.get("reasoning", "")
                    return self.send(200, app.student_reasoning(key, reasoning))

                if self.path == "/api/adaptive":
                    is_correct = body.get("correct", False)
                    return self.send(200, app.adaptive_next(key, is_correct))

                if self.path == "/api/counterexample":
                    reasoning = body.get("reasoning", "")
                    return self.send(200, app.get_counterexample(key, reasoning))

                if self.path == "/api/translate":
                    lang = body.get("language", "cat")
                    return self.send(200, app.translate(key, lang))

                if self.path == "/api/hint-reset":
                    app.clear_hint(key)
                    return self.send(200, {"ok": True})

            except (OSError, urllib.error.URLError) as error:
                return self.send(502, {"error": f"model server: {error}"})
            except Exception as error:  # noqa: BLE001 - show the problem instead of hanging
                traceback.print_exc()
                return self.send(500, {"error": f"{type(error).__name__}: {error}"})
            return self.send(400, {"error": "bad request"})

    return Handler


def main():
    args = parse_args()
    app = App(args)
    info = app.info()
    print(
        f"{info['count']} problems from {args.input}, {info['images']} with images, "
        f"{info['index']} in the retrieval index"
    )
    print(
        f"Servers: language model {'up' if info['llm'] else 'DOWN'}, "
        f"embeddings {'up' if info['embed'] else 'DOWN'}, "
        f"vision {'up' if info['vision'] else 'off'}"
    )
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()