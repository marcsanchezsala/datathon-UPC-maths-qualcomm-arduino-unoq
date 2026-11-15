"""Math Matrix Copilot — Python (runs on Linux, QRB2210)

Shows the copilot's state on the 8x13 LED matrix through the Bridge:
answer letters, a tick or a cross, the hint level, the streak, and a
"thinking" animation while the model works.

One display thread owns the matrix. Everything else only sets the scene, so
two threads never draw at the same time, and nothing flashes faster than
2 times per second (the limit is 3).

The tutor (explorer/server.py) can drive the matrix through these endpoints:
    GET /feedback/{letter}/{correct}   correct = true/false/1/0 -> letter, then tick or cross
    GET /hint/{level}                  0..4 -> bar with one segment per rung
    GET /streak/{n}                    0..13 -> bar of n columns
    GET /thinking/{on}                 1 = start the animation, 0 = stop it
    GET /label/{text}                  up to 3 characters in a 3x5 font, e.g. OK, NO, T1
    GET /clear, /demo/all, /status
"""

import json
import threading
import time
import urllib.request

import numpy as np

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

LLM = "http://127.0.0.1:8081"

H, W = 8, 13
IDLE_AFTER = 30.0         # seconds without activity before the idle heartbeat replaces the frame
MIN_BLINK_PERIOD = 0.5    # seconds per on/off cycle: at most 2 per second

# ── Fonts ───────────────────────────────────────────────────────────────────
FONT5x7 = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "?": [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "OK": [".....", "....#", "...#.", "#..#.", ".##..", ".#...", "....."],  # tick
}
FONT3x5 = {
    "A": [".#.", "#.#", "###", "#.#", "#.#"], "E": ["###", "#..", "##.", "#..", "###"],
    "F": ["###", "#..", "##.", "#..", "#.."], "G": [".##", "#..", "#.#", "#.#", ".##"],
    "H": ["#.#", "#.#", "###", "#.#", "#.#"], "I": ["###", ".#.", ".#.", ".#.", "###"],
    "K": ["#.#", "##.", "#..", "##.", "#.#"], "L": ["#..", "#..", "#..", "#..", "###"],
    "N": ["#.#", "###", "###", "###", "#.#"], "O": [".#.", "#.#", "#.#", "#.#", ".#."],
    "R": ["##.", "#.#", "##.", "#.#", "#.#"], "S": [".##", "#..", ".#.", "..#", "##."],
    "T": ["###", ".#.", ".#.", ".#.", ".#."], "0": ["###", "#.#", "#.#", "#.#", "###"],
    "1": [".#.", "##.", ".#.", ".#.", "###"], "2": ["##.", "..#", ".#.", "#..", "###"],
    "3": ["##.", "..#", ".#.", "..#", "##."], "4": ["#.#", "#.#", "###", "..#", "..#"],
    "5": ["###", "#..", "##.", "..#", "##."], " ": ["...", "...", "...", "...", "..."],
}


def blank():
    return np.zeros((H, W), dtype=np.uint8)


def put(grid, pattern, x, y, b=7):
    for r, line in enumerate(pattern):
        for c, px in enumerate(line):
            if px == "#" and 0 <= y + r < H and 0 <= x + c < W:
                grid[y + r, x + c] = b


def glyph(ch, b=7):
    g = blank()
    put(g, FONT5x7.get(ch, FONT5x7["?"]), 4, 0, b)
    return g


def label(text, b=7):
    """Up to 3 characters in the 3x5 font, centred (3 cols each, 1 col between)."""
    text = text.upper()[:3]
    g = blank()
    width = len(text) * 4 - 1
    x = (W - width) // 2
    for ch in text:
        put(g, FONT3x5.get(ch, FONT3x5[" "]), x, 1, b)
        x += 4
    return g


def bar(level, maximum, b=7):
    """A bar in the middle rows: `level` segments lit out of `maximum`, 2 columns each when it fits."""
    g = blank()
    level = max(0, min(int(level), maximum))
    seg = 2 if maximum * 3 - 1 <= W else 1
    gap = 1 if seg == 2 else 0
    total = maximum * seg + (maximum - 1) * gap
    x = (W - total) // 2
    for i in range(maximum):
        val = b if i < level else 1          # dim segments show the maximum
        g[3:5, x:x + seg] = val
        x += seg + gap
    return g


# ── The display thread ─────────────────────────────────────────────────────
class Display:
    """Owns the matrix. Others call show(), set() or thinking()."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scene = [(glyph("?", 4), 0)]   # list of (frame, seconds); 0 = hold
        self.loop = False
        self.version = 0
        self.last_activity = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _send(self, grid):
        try:
            # 104 characters '0'..'7', one per LED, row by row: a plain string is the
            # most portable type for the Bridge on every App Lab version
            Bridge.call("draw", "".join(str(int(v)) for v in grid.flatten()))
        except Exception as exc:  # the sketch may still be starting
            print("Bridge error:", exc, flush=True)

    def set(self, frames, loop=False):
        with self.lock:
            self.scene, self.loop = frames, loop
            self.version += 1
            self.last_activity = time.time()

    def show(self, grid):
        self.set([(grid, 0)])

    def thinking(self, on):
        if on:  # a slow pulse of brightness, never off: no flashing
            self.set([(glyph("?", b), 0.35) for b in (3, 5, 7, 5)], loop=True)
        else:
            self.show(glyph("?", 4))

    def _changed(self, version):
        return self.version != version

    def _wait(self, seconds, version):
        """Sleep up to `seconds`; True if the scene changed meanwhile."""
        end = time.time() + seconds
        while time.time() < end:
            if self._changed(version):
                return True
            time.sleep(0.05)
        return self._changed(version)

    def _run(self):
        while True:
            with self.lock:
                frames, loop, version = list(self.scene), self.loop, self.version
            interrupted = False
            for grid, secs in frames:
                self._send(grid)
                if secs and self._wait(secs, version):
                    interrupted = True
                    break
            if interrupted or loop:
                continue
            # Hold the last frame until something changes; heartbeat when idle
            while not self._changed(version):
                if time.time() - self.last_activity > IDLE_AFTER:
                    self._idle(version)
                else:
                    time.sleep(0.1)

    def _idle(self, version):
        """Slow heartbeat while nobody uses it: one small dot, 1 cycle per 1.6 s."""
        dot = blank()
        dot[6:8, 6] = 3
        while not self._changed(version):
            self._send(dot)
            if self._wait(0.8, version):
                return
            self._send(blank())
            if self._wait(0.8, version):
                return


display = Display()
state = {"busy": False, "message": "Welcome! Choose an answer.", "confidence": 0,
         "mode": "RST", "correct": None, "hint": 0, "streak": 0}


def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "si", "sí", "ok", "correct")


# ── Endpoints ──────────────────────────────────────────────────────────────
def solve_answer(letter: str):
    """Placeholder kept from the first version: shows the chosen letter after
    checking that the model server answers. The tutor uses /feedback instead."""
    letter = letter.upper()
    if len(letter) != 1 or letter not in "ABCDE":
        return {"error": "Choose A to E"}
    if state["busy"]:
        return {"error": "Still working on previous problem"}
    state["busy"] = True
    display.thinking(True)

    def worker():
        try:
            with urllib.request.urlopen(LLM + "/health", timeout=5) as resp:
                status = json.loads(resp.read().decode()).get("status", "ok")
            state.update(mode="T1", confidence=5, message=f"Model: {status}")
            display.set([(glyph(letter), 1.0), (bar(5, 5), 0)])
        except Exception as exc:
            state.update(mode="ERR", message=f"Error: {exc}")
            display.show(label("ERR"))
        finally:
            state["busy"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


def set_answer_with_feedback(letter: str, correct: str):
    letter = letter.upper()
    if len(letter) != 1 or letter not in "ABCDE":
        return {"error": "Choose A to E"}
    ok = truthy(correct)
    state.update(correct=ok, mode="OK" if ok else "NO",
                 streak=state["streak"] + 1 if ok else 0,
                 message=f"Correct! Answer: {letter}" if ok else "Incorrect. Try again.")
    mark = glyph("OK") if ok else glyph("X")
    half = MIN_BLINK_PERIOD / 2
    # letter 1 s, then the mark blinking twice slowly (2 cycles per second at most), then held
    display.set([(glyph(letter), 1.0), (mark, half), (blank(), half),
                 (mark, half), (blank(), half), (mark, 0)])
    return {"ok": True, "correct": ok}


def set_confidence(level: str):
    level = max(0, min(int(level), 5))
    state.update(confidence=level, message=f"Confidence: {level}/5")
    display.show(bar(level, 5))
    return {"ok": True}


def set_hint(level: str):
    level = max(0, min(int(level), 4))
    state.update(hint=level, message=f"Hint {level}/4")
    display.show(bar(level, 4))
    return {"ok": True}


def set_streak(n: str):
    n = max(0, min(int(n), W))
    state.update(streak=n, message=f"Streak: {n}")
    display.show(bar(n, W))
    return {"ok": True}


def set_thinking(on: str):
    display.thinking(truthy(on))
    return {"ok": True}


def set_label(text: str):
    display.show(label(text))
    return {"ok": True}


def clear():
    display.show(blank())
    return {"ok": True}


def demo_all():
    """Every capability in sequence. Runs in the display thread: the API returns at once."""
    display.set([(glyph(c), 0.7) for c in "ABCDE"] +
                [(glyph("OK"), 1.0), (glyph("X"), 1.0)] +
                [(bar(i, 4), 0.5) for i in range(5)] +
                [(label(t), 1.0) for t in ("T1", "RAG", "OK", "NO")] +
                [(glyph("?", b), 0.35) for b in (3, 5, 7, 5, 3)] +
                [(blank(), 0)])
    return {"ok": True}


ui = WebUI()
ui.expose_api("GET", "/answer/{letter}", solve_answer)
ui.expose_api("GET", "/feedback/{letter}/{correct}", set_answer_with_feedback)
ui.expose_api("GET", "/confidence/{level}", set_confidence)
ui.expose_api("GET", "/hint/{level}", set_hint)
ui.expose_api("GET", "/streak/{n}", set_streak)
ui.expose_api("GET", "/thinking/{on}", set_thinking)
ui.expose_api("GET", "/label/{text}", set_label)
ui.expose_api("GET", "/status", lambda: state)
ui.expose_api("GET", "/clear", clear)
ui.expose_api("GET", "/demo/all", demo_all)

App.run()