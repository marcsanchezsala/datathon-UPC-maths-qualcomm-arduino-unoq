"""Math Matrix Copilot — Python (runs on Linux QRB2210)

Interactive math copilot app that shows answers, confidence levels,
thinking animations, and feedback on the 8×13 LED matrix via Bridge.

Usage on the board:
    arduino-app-cli app start user:math-matrix
"""

import json
import threading
import time
import urllib.request

import numpy as np

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Frame

# ── Model server URLs ────────────────────────────────────────────────────────
LLM = "http://127.0.0.1:8081"
EMBED = "http://172.16.189.56:8082"

# ── LED Matrix Constants ─────────────────────────────────────────────────────
MATRIX_W = 13
MATRIX_H = 8
BRIDGE_DELAY = 0.05   # seconds between frame sends during animation

# ── 5×7 Font (centered in 8×13 matrix: x_offset=4) ──────────────────────────
FONT = {
    "A": [
        ".###.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ],
    "B": [
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#...#",
        "#...#",
        "####.",
    ],
    "C": [
        ".###.",
        "#...#",
        "#....",
        "#....",
        "#....",
        "#...#",
        ".###.",
    ],
    "D": [
        "####.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "####.",
    ],
    "E": [
        "#####",
        "#....",
        "#....",
        "####.",
        "#....",
        "#....",
        "#####",
    ],
    "?": [
        ".###.",
        "#...#",
        "....#",
        "...#.",
        "..#..",
        ".....",
        "..#..",
    ],
    "X": [
        "....#",
        "..#..",
        ".#.#.",
        ".#.#.",
        ".#.#.",
        "..#..",
        "#....",
    ],
    # ── Extra glyphs for teaching ─────────────────────────────────────────
    "✓": [
        "..#..",
        ".#.#.",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#..#.",
        "#..#.",
    ],
    # "Y" — used for "Yes / Correct" (alternative to ✓)
    "Y": [
        "#...#",
        "#...#",
        ".#.#.",
        ".#.#.",
        "..#..",
        "..#..",
        "..#..",
    ],
}

# ── Drawing Primitives ──────────────────────────────────────────────────────


def _build_blank() -> np.ndarray:
    """Return an 8×13 uint8 zero-filled array."""
    return np.zeros((MATRIX_H, MATRIX_W), dtype=np.uint8)


def _draw_char(grid: np.ndarray, ch: str, x_start: int, y_start: int, brightness: int = 7):
    """Draw a 5×7 character font at position (x_start, y_start).

    Args:
        grid: frame buffer (modified in-place)
        ch: font key (A-E, ?, X, ✓, Y)
        x_start: column to start at (0–8, since 5 cols wide)
        y_start: row to start at (0–1, since 7 rows tall in 8-row matrix)
        brightness: 0-7
    """
    pattern = FONT.get(ch)
    if pattern is None:
        return
    for row_idx, row_str in enumerate(pattern):
        row = y_start + row_idx
        if row >= MATRIX_H:
            continue
        for col_idx, pixel in enumerate(row_str):
            col = x_start + col_idx
            if col >= MATRIX_W:
                continue
            if pixel == "#":
                grid[row, col] = brightness


def draw_letter(ch: str, x: int = 4, y: int = 0, brightness: int = 7):
    """Draw a single letter centered or at specified position."""
    grid = _build_blank()
    _draw_char(grid, ch, x, y, brightness)
    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_checkmark(brightness: int = 7):
    """Draw a 5×7 checkmark (✓)."""
    grid = _build_blank()
    # Center the checkmark (5 cols → x=4, 7 rows → y=0)
    _draw_char(grid, "✓", 4, 0, brightness)
    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_cross(brightness: int = 7):
    """Draw an X / cross. Used for incorrect answers."""
    grid = _build_blank()
    _draw_char(grid, "X", 4, 0, brightness)
    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_confidence_bar(level: int, max_level: int = 5, brightness: int = 7):
    """Draw a horizontal confidence bar using filled and empty circles.

    5 segments, each 1 column wide, centered vertically.
    Segments are at rows 3-4 (two rows high for visibility).
    """
    level = max(0, min(level, max_level))
    grid = _build_blank()

    # 5 bar segments, each 1 column, 2 rows tall, centered
    bar_start_col = 4  # (13 - 5) // 2
    bar_row_start = 3  # (8 - 2) // 2

    for i in range(max_level):
        col = bar_start_col + i
        filled = i < level
        fill_val = brightness if filled else 0
        grid[bar_row_start, col] = fill_val
        grid[bar_row_start + 1, col] = fill_val

    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_question_anim_step(step: int = 0):
    """Animated question mark: 3 frames of pulsing brightness."""
    grid = _build_blank()
    brightness = 3 + (step % 3) * 2  # alternates 3, 5, 7
    _draw_char(grid, "?", 4, 0, brightness)
    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_thinking_animation(stop_event: threading.Event):
    """Run an animated "thinking" sequence: ? pulsing + side bars.

    Args:
        stop_event: threading.Event to break the animation loop.
    """
    for i in range(120):  # ~10 seconds at 6 fps
        if stop_event.is_set():
            break
        # Draw a ? with oscillating brightness
        draw_question_anim_step(i)
        for _ in range(3):
            time.sleep(BRIDGE_DELAY)
            if stop_event.is_set():
                break


def draw_flash_animation(flashes: int = 3, duration_ms: int = 200):
    """Flash the matrix on/off a number of times.

    Used to signal that a result is ready or a new step started.
    """
    for _ in range(flashes):
        grid = _build_blank()
        # Fill the whole matrix briefly
        grid[:, :] = 7
        Bridge.call("draw", Frame(grid).to_board_bytes())
        time.sleep(duration_ms / 2000.0)
        grid[:, :] = 0
        Bridge.call("draw", Frame(grid).to_board_bytes())
        time.sleep(duration_ms / 2000.0)


def draw_status_text(text: str, brightness: int = 7):
    """Draw up to 8 characters horizontally across the matrix.

    Characters are drawn at row 2 (1 row above middle) with spacing.
    Falls back to single-character display if too many chars.
    """
    grid = _build_blank()
    short_text = text[:8].upper()

    if len(short_text) <= 4:
        # Wider spacing for short texts
        spacing = (MATRIX_W - len(short_text) * 3) // (len(short_text) + 1)
        for i, ch in enumerate(short_text):
            x = spacing + i * 3
            if ch == " ":
                continue
            _draw_char(grid, ch, x, 0, brightness)
    else:
        # Just squeeze them in
        for i, ch in enumerate(short_text):
            if i >= MATRIX_W // 2:
                break
            if ch == " ":
                continue
            _draw_char(grid, ch, i, 0, brightness)

    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_mode_indicator(mode: str):
    """Show a tiny mode label on the left side.

    Modes: "RAG", "T1", "T2", "RST" (reset)
    """
    grid = _build_blank()
    mode_label = mode[:3]
    for i, ch in enumerate(mode_label):
        if i >= 3:
            break
        _draw_char(grid, ch, 0, 0, 5)
    # Add a separator dot
    grid[0, 4] = 3
    Bridge.call("draw", Frame(grid).to_board_bytes())


def draw_heartbeat(stop_event: threading.Event):
    """Continuous heartbeat animation: small pulse at center-bottom.

    Used as an idle indicator that the app is alive.
    """
    pulse_rows = [6, 7]
    pulse_cols = [6, 7, 8]
    while not stop_event.is_set():
        grid = _build_blank()
        for r in pulse_rows:
            for c in pulse_cols:
                grid[r, c] = 4
        Bridge.call("draw", Frame(grid).to_board_bytes())
        time.sleep(0.4)
        if stop_event.is_set():
            break
        grid = _build_blank()
        Bridge.call("draw", Frame(grid).to_board_bytes())
        time.sleep(0.4)


def clear_matrix():
    """Turn off all LEDs."""
    grid = _build_blank()
    Bridge.call("draw", Frame(grid).to_board_bytes())


# ── State ────────────────────────────────────────────────────────────────────

state = {
    "busy": False,
    "message": "Welcome! Choose an answer.",
    "confidence": 0,
    "mode": "RST",
    "correct": None,
}

# ── Copilot Worker (runs in a background thread) ─────────────────────────────


def solve_answer(letter: str):
    """Answer pipeline: call model → show result on matrix."""
    letter = letter.upper()
    if letter not in "ABCDE":
        return {"error": "Choose A to E"}

    if state["busy"]:
        return {"error": "Still working on previous problem"}

    state["busy"] = True
    stop_thinking = threading.Event()

    def worker():
        global state
        try:
            # ── Phase 1: Thinking animation ──────────────────────────
            draw_mode_indicator("T1")
            draw_thinking_animation(stop_thinking)

            # ── Phase 2: Call the model ─────────────────────────────
            # Try the LLM health endpoint first (replace with real solve())
            with urllib.request.urlopen(LLM + "/health", timeout=5) as resp:
                model_status = json.loads(resp.read().decode())

            state["mode"] = "T1"
            state["confidence"] = 5
            state["message"] = f"Model: {model_status.get('status', 'ok')}"

            # ── Phase 3: Flash + show answer ────────────────────────
            draw_flash_animation(flashes=2, duration_ms=100)

            # Show letter + confidence bar
            draw_letter(letter)
            time.sleep(0.5)
            draw_confidence_bar(level=5)

        except Exception as exc:
            state["mode"] = "ERR"
            state["message"] = f"Error: {exc}"
            draw_cross(brightness=7)

        finally:
            state["busy"] = False
            stop_thinking.set()

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


def set_answer_with_feedback(letter: str, is_correct: bool):
    """Manually set the LED output with check/cross feedback."""
    letter = letter.upper()
    if letter not in "ABCDE":
        return {"error": "Choose A to E"}

    state["busy"] = True
    stop_thinking = threading.Event()
    state["correct"] = is_correct

    def worker():
        global state
        try:
            draw_mode_indicator("FIN")
            draw_thinking_animation(stop_thinking)

            if is_correct:
                draw_flash_animation(flashes=3, duration_ms=120)
                draw_letter(letter)
                time.sleep(0.3)
                draw_checkmark(brightness=7)
                state["mode"] = "OK"
                state["message"] = f"Correct! Answer: {letter}"
            else:
                draw_flash_animation(flashes=2, duration_ms=150)
                draw_letter(letter)
                time.sleep(0.3)
                draw_cross(brightness=7)
                state["mode"] = "NO"
                state["message"] = f"Incorrect. Try again."

        except Exception as exc:
            state["message"] = f"Error: {exc}"
            clear_matrix()

        finally:
            state["busy"] = False
            stop_thinking.set()

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


def set_confidence(level: int):
    """Set confidence bar level directly (0-5)."""
    level = max(0, min(int(level), 5))
    state["confidence"] = level
    draw_confidence_bar(level)
    state["message"] = f"Confidence: {level}/5"
    return {"ok": True}


# ── Web UI Endpoints ─────────────────────────────────────────────────────────

ui = WebUI()

# Solve answer (calls model)
ui.expose_api("GET", "/answer/{letter}", solve_answer)

# Set answer with manual feedback (check/cross)
ui.expose_api("GET", "/feedback/{letter}/{correct}", set_answer_with_feedback)

# Confidence bar control
ui.expose_api("GET", "/confidence/{level}", set_confidence)

# Status polling
ui.expose_api("GET", "/status", lambda: state)

# Clear matrix
ui.expose_api("GET", "/clear", lambda: (clear_matrix(), {"ok": True})[1])

# Demo: show all glyphs
ui.expose_api("GET", "/demo/all", lambda: (draw_demo_all(), {"ok": True})[1])


def draw_demo_all():
    """Show a full demo of all LED matrix capabilities."""
    draw_letter("M")
    time.sleep(0.5)
    draw_letter("A")
    time.sleep(0.5)
    draw_letter("T")
    time.sleep(0.5)
    draw_letter("H")
    time.sleep(1)
    draw_checkmark()
    time.sleep(0.5)
    draw_cross()
    time.sleep(0.5)
    draw_confidence_bar(3)
    time.sleep(1)
    draw_confidence_bar(5)
    time.sleep(1)
    draw_confidence_bar(0)
    time.sleep(0.5)
    draw_status_text("HELLO!")
    time.sleep(1)
    draw_mode_indicator("RAG")
    time.sleep(1)
    draw_mode_indicator("T2")
    time.sleep(1)
    clear_matrix()


# ── Idle Heartbeat ───────────────────────────────────────────────────────────

heartbeat_stop = threading.Event()


def heartbeat_loop():
    """Run the heartbeat animation until stopped."""
    draw_question_anim_step(0)
    draw_heartbeat(heartbeat_stop)


# ── Main ─────────────────────────────────────────────────────────────────────

show = draw_letter  # alias for initial display
show("?")

# Start heartbeat in background (shows app is alive)
heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
heartbeat_thread.start()

App.run()