"""Tests for Teaching Copilot: Theory Nuggets, Socratic Dialogue, LED Matrix Triggers."""
import json
import pytest
from copilot.tutor import (
    tutor_theory_nugget,
    detect_problem_domain,
    tutor_socratic_step,
    tutor_analyze_trap,
)


def test_theory_nugget_domains_and_bilingual():
    # Geometry Catalan
    p_geom_cat = {
        "id": "lvl-0_test_geom_cat",
        "text": "Quin és el perímetre d'un rectangle si té costats de 4 cm i 6 cm?",
        "language": "cat",
        "options": {"A": "10", "B": "20", "C": "24", "D": "16", "E": "12"},
        "label": "B",
    }
    nugget_geom = tutor_theory_nugget(p_geom_cat)
    assert nugget_geom["domain"] == "geometry"
    assert "Perímetre" in nugget_geom["title"]
    assert len(nugget_geom["tip"]) > 10
    assert len(nugget_geom["rule"]) > 10
    assert "B" not in nugget_geom["rule"]  # Never leak answer

    # Geometry English
    p_geom_eng = {
        "id": "lvl-0_test_geom_eng",
        "text": "Find the perimeter of a rectangle with sides 4 and 6.",
        "language": "eng",
        "options": {"A": "10", "B": "20", "C": "24", "D": "16", "E": "12"},
        "label": "B",
    }
    nugget_geom_en = tutor_theory_nugget(p_geom_eng)
    assert nugget_geom_en["domain"] == "geometry"
    assert "Perimeter" in nugget_geom_en["title"]

    # Algebra English
    p_alg = {
        "id": "lvl-1_test_alg",
        "text": "If x + 7 = 19, what is the value of x?",
        "language": "eng",
        "options": {"A": "11", "B": "12", "C": "13", "D": "14", "E": "15"},
        "label": "B",
    }
    nugget_alg = tutor_theory_nugget(p_alg)
    assert nugget_alg["domain"] == "algebra"
    assert "Algebraic" in nugget_alg["title"]

    # Arithmetic Catalan
    p_arith = {
        "id": "lvl-0_test_arith",
        "text": "Quants nombres parells de dues xifres són múltiples de 5?",
        "language": "cat",
        "options": {"A": "9", "B": "10", "C": "18", "D": "20", "E": "25"},
        "label": "A",
    }
    nugget_arith = tutor_theory_nugget(p_arith)
    assert nugget_arith["domain"] == "arithmetic"
    assert "Divisibilitat" in nugget_arith["title"] or "Paritat" in nugget_arith["title"]

    # Combinatorics Catalan (Pigeonhole)
    p_comb = {
        "id": "lvl-2_test_comb",
        "text": "A la gàbia hi ha 5 coloms i tenim 4 forats (principi del colomar). Quantes maneres?",
        "language": "cat",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        "label": "B",
    }
    nugget_comb = tutor_theory_nugget(p_comb)
    assert nugget_comb["domain"] == "combinatorics"
    assert "Colomar" in nugget_comb["title"]

    # Logic Catalan
    p_logic = {
        "id": "lvl-1_test_logic",
        "text": "En Joan sempre diu la veritat i en Marc és mentider.",
        "language": "cat",
        "options": {"A": "Joan", "B": "Marc", "C": "Cap", "D": "Tots dos", "E": "No es pot saber"},
        "label": "A",
    }
    nugget_logic = tutor_theory_nugget(p_logic)
    assert nugget_logic["domain"] == "logic"
    assert "Lògic" in nugget_logic["title"]


def test_socratic_step_fallbacks_all_domains():
    problem = {
        "id": "lvl-0_test_soc",
        "text": "Un rectangle té costats de 4 cm i 6 cm. Quin és el perímetre?",
        "language": "cat",
        "options": {"A": "10", "B": "20", "C": "24", "D": "16", "E": "12"},
        "label": "B",
    }
    # Test all 3 steps without model (fallback path is tested under timeout or mock)
    for step in [1, 2, 3]:
        res = tutor_socratic_step(problem, step=step, timeout=0.01)
        assert res["step"] == step
        assert len(res["text"]) > 5
        assert res["domain"] == "geometry"


def test_server_app_integration(tmp_path):
    from explorer.server import App
    import argparse

    # Create dummy input data
    val_file = tmp_path / "val.jsonl"
    dummy_probs = [
        {
            "id": "lvl-0_dummy1",
            "language": "cat",
            "level": 0,
            "text": "Quin és el perímetre d'un rectangle de 3 per 5?",
            "options": {"A": "8", "B": "15", "C": "16", "D": "10", "E": "20"},
            "label": "C",
        },
        {
            "id": "lvl-1_dummy2",
            "language": "eng",
            "level": 1,
            "text": "Find x if 2*x + 4 = 10.",
            "options": {"A": "2", "B": "3", "C": "4", "D": "5", "E": "6"},
            "label": "B",
        },
    ]
    with open(val_file, "w", encoding="utf-8") as f:
        for p in dummy_probs:
            f.write(json.dumps(p) + "\n")

    args = argparse.Namespace(
        input=str(val_file),
        images=str(tmp_path),
        index=str(tmp_path / "index.db"),
    )
    app = App(args)

    # 1. Test theory nugget via App
    nugget = app.theory_nugget("lvl-0_dummy1_cat")
    assert nugget["domain"] == "geometry"
    assert "Perímetre" in nugget["title"]

    # 2. Test Socratic dialogue via App
    dialogue_res = app.socratic_dialogue("lvl-0_dummy1_cat", message="Crec que hem de sumar els costats")
    assert dialogue_res["step"] == 1
    assert len(dialogue_res["text"]) > 5

    # 3. Test LED trigger via App (silent safe)
    led_res = app.trigger_led("thinking")
    assert led_res["status"] == "ok"
    assert led_res["action"] == "thinking"

    led_correct = app.trigger_led("correct")
    assert led_correct["status"] == "ok"
    assert led_correct["action"] == "correct"

    led_distractor = app.trigger_led("distractor")
    assert led_distractor["status"] == "ok"
    assert led_distractor["action"] == "distractor"

    led_clear = app.trigger_led("clear")
    assert led_clear["status"] == "ok"
    assert led_clear["action"] == "clear"
