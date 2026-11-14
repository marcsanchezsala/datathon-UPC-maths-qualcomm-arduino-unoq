"""M3Kang Teaching Copilot - Edge Math Companion.

Complete teaching system for the Math on the Edge datathon.
Separate from copilot/pipeline.py so the reasoning team can optimise
predict.py without touching teaching code.

Features:
  1. Hint ladder (4 levels, fast): RAG example → method hint → model hint → solution
  2. Student reasoning correction: student writes their thinking, we find a similar problem
     where the same mistake leads to a wrong answer (counterexample).
  3. Typical error detection: regex rules for off-by-one, missing operation, wrong sign...
  4. Counterexample search: find similar train problems where the student's mistake → wrong option.
  5. Multilingual bridge: translate problems using their ID.
  6. Adaptive difficulty: next problem easier if wrong, harder if right.
"""
import re
import math
from .client import post
from .config import CONFIG
from .rag import Index

TUTOR_URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# ── Error detection patterns ──────────────────────────────────────────────

# These patterns detect common student mistakes and suggest the correction.
ERROR_RULES = [
    {
        "name": "off_by_one",
        "pattern": r"^(?:diff|differs?|more|less|short|exceed|missed?)\s+(-?\d+)$",
        "instruction": "Check: is the answer off by 1? Try +1 or -1 from the correct answer.",
    },
    {
        "name": "missing_addition",
        "pattern": r"^(?:sum|total|together|add(ed|ition|ition))\s*[=:]?\s*(-?\d+)$",
        "instruction": "Check: did you add all the parts? Try adding the missing number.",
    },
    {
        "name": "missing_multiplication",
        "pattern": r"^(?:product|times|multipl(ie|y))\s*[=:]?\s*(-?\d+)$",
        "instruction": "Check: did you multiply by the right factor? Try multiplying.",
    },
    {
        "name": "wrong_sign",
        "pattern": r"^(-)\s*([\d.]+)$",
        "instruction": "Check: did you forget the sign? Try the positive version.",
    },
]

# Regex to extract numbers from student reasoning text
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _build_problem_text(problem):
    """Format a problem into plain text, same style as the explorer."""
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for l in "ABCDE":
        if l in options:
            lines.append(f"{l}) {options[l]}")
    return "\n".join(lines)


def detect_typical_errors(student_text, correct_answer, correct_value=None):
    """Detect common mistakes in student's reasoning using regex rules.

    Returns a list of detected error suggestions.
    """
    detected = []
    if not student_text:
        return detected

    student_num = NUMBER_RE.search(student_text)
    student_value = float(student_num.group()) if student_num else None

    for rule in ERROR_RULES:
        if rule["pattern"] in student_text.lower():
            detected.append(rule["instruction"])

    if correct_value and student_value and student_value != correct_value:
        if abs(student_value - correct_value) == 1:
            detected.append("Your answer might be off by 1. Try +1 or -1.")
        elif abs(student_value - correct_value) < 10:
            detected.append("Your answer is close but not quite right. Check your calculations.")

    return detected


def find_rag_examples(problem, index, k=2):
    """Find similar problems at a lower level for RAG-based hints."""
    if not index:
        return []
    hits = index.search(problem, k=k * 3, exclude_id=problem["id"])
    examples = []
    for score, p in hits:
        if p.get("level") is not None and p["level"] < problem["level"]:
            examples.append((score, p))
            if len(examples) >= k:
                break
        elif p.get("level") is not None and p["level"] == problem["level"] and len(examples) == 0:
            examples.append((score, p))
            if len(examples) >= k:
                break
    return examples


def find_counterexample(problem, student_reasoning, index, k=3):
    """Find a similar train problem where the student's mistake leads to a wrong answer.

    Strategy: find similar problems, check if the student's answer pattern would lead
    to one of the wrong options (A-E except the correct one).
    """
    if not index:
        return []

    hits = index.search(problem, k=k * 2, exclude_id=problem["id"])
    counterexamples = []

    for score, p in hits:
        if score < 0.6:
            continue
        if p.get("options") and student_reasoning:
            # Check if the student's reasoning would match one wrong option in the similar problem
            student_num = NUMBER_RE.search(student_reasoning)
            if student_num:
                student_val = float(student_num.group())
                for letter, value in p["options"].items():
                    if letter != p["label"]:
                        try:
                            opt_val = float(value)
                            if abs(opt_val - student_val) < 0.5:
                                counterexamples.append((score, p, letter, value))
                                break
                        except (ValueError, TypeError):
                            pass
        if len(counterexamples) >= k:
            break

    return counterexamples


def translate_problem(problem, target_lang, index=None):
    """Translate a problem's text using the multilingual bridge.

    M3Kang has the same problem (same ID) in many languages.
    We search for the same problem in the target language.
    """
    if not index:
        return None

    # Search for the same problem ID in the target language
    problem_id = problem["id"].rsplit("_", 1)[0]  # e.g. "lvl-0_2015_14"
    hits = index.search(problem, k=10, exclude_id=problem["id"])
    for _score, p in hits:
        if p["language"] == target_lang and problem_id in p["id"]:
            return p
    return None


def tutor_messages(problem, hint_level=3, student_reasoning=None):
    """Return the messages for a tutor request."""
    problem_text = _build_problem_text(problem)

    if hint_level == 4:
        system_content = (
            "The student needs the full solution now. Explain reasoning step by step, "
            "show the calculation or logic, and state the correct letter clearly."
        )
        user_content = f"The student is ready for the solution.\n{problem_text}"
    elif hint_level == 2:
        system_content = (
            "You are a concise math tutor. Provide a ONE-SENTENCE method hint. "
            "Do NOT calculate. Identify the problem type and what approach to use."
        )
        user_content = f"The student needs a method hint.\n{problem_text}"
    elif hint_level == 3:
        system_content = (
            "You are a patient math tutor. Give a ONE-LINE hint. "
            "Point out ONE key thing the student is missing. Max 25 words."
        )
        user_content = f"The student needs a short hint.\n{problem_text}"
    elif hint_level == 5:  # Student reasoning correction
        system_content = (
            "The student wrote their reasoning below. Give ONE short sentence "
            "pointing out what might be wrong, without giving the answer. "
            "Reference the student's specific thinking."
        )
        user_content = (
            f"Problem: {problem_text}\n"
            f"Student's reasoning: {student_reasoning}\n"
            f"Give a short correction hint."
        )
    else:  # Level 1: RAG
        system_content = "No message needed, RAG example is shown directly."
        user_content = problem_text

    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


def tutor_generate(problem, index=None, hint_level=3, student_reasoning=None, timeout=120):
    """Generate a teaching response for the given hint level.

    hint_level:
      1 = RAG example (similar problem, instant)
      2 = Method hint (LLM, fast, short)
      3 = Model hint (LLM, short)
      4 = Full solution (LLM, longer)
      5 = Student reasoning correction (LLM, short)
    """
    # Level 1: RAG pure, no LLM call (instant, <0.5s)
    if hint_level == 1 and index:
        examples = find_rag_examples(problem, index, k=2)
        if examples:
            rag_text = "Here is a similar problem that was already solved. Think about how you would solve it:\n"
            for _score, p in examples:
                rag_text += _build_problem_text(p) + "\n"
            return {
                "text": rag_text,
                "type": "hint",
                "level": 1,
                "is_rag": True,
                "prompt_tokens": 0,
                "timings": {},
            }

    messages = tutor_messages(problem, hint_level, student_reasoning)

    # Optimization: short responses for hints to avoid long waits
    if hint_level <= 3:
        payload = {
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "cache_prompt": True,
        }
    elif hint_level == 5:  # Reasoning correction
        payload = {
            "messages": messages,
            "max_tokens": 60,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "cache_prompt": True,
        }
    else:  # Level 4: solution
        payload = {
            "messages": messages,
            "max_tokens": 250,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "cache_prompt": True,
        }

    try:
        response = post(TUTOR_URL, payload, timeout=timeout)
        text = response["choices"][0]["message"]["content"].strip()
        return {
            "text": text,
            "type": "solution" if hint_level == 4 else "hint",
            "level": hint_level,
            "prompt_tokens": response.get("timings", {}).get("prompt_n", 0),
            "timings": response.get("timings", {}),
        }
    except Exception as e:
        return {"text": f"Error generating response: {e}", "type": "error", "level": hint_level}


def tutor_student_correction(problem, student_reasoning, index=None, timeout=120):
    """Special endpoint for student reasoning correction.

    Finds a counterexample from training data where the student's mistake
    would lead to a wrong answer in the similar problem.
    """
    # 1. Detect typical errors (instant, no LLM)
    detected_errors = detect_typical_errors(student_reasoning, "")

    # 2. Find counterexamples from similar problems
    counterexamples = find_counterexample(problem, student_reasoning, index)

    # 3. If we have counterexamples, use one
    if counterexamples:
        _score, p, wrong_letter, wrong_value = counterexamples[0]
        counterexample_text = (
            f"I found a similar problem where the same approach leads to a different answer.\n"
            f"Similar problem: {p['text']}\n"
            f"If you solve it like yours, you get {wrong_letter}) {wrong_value}, "
            f"but the correct answer is {p['label']}) {p.get('options', {}).get(p['label'], '?')}.\n"
            f"Think: what's different about your problem? Try again."
        )
        return {
            "text": counterexample_text,
            "type": "counterexample",
            "level": 6,
            "detected_errors": detected_errors,
            "counterexample": {
                "id": p["id"],
                "score": round(_score, 3),
                "wrong_option": wrong_letter,
                "wrong_value": wrong_value,
                "correct_letter": p["label"],
            },
        }

    # 4. No counterexample: use the model to give a targeted correction
    messages = tutor_messages(problem, 5, student_reasoning)
    payload = {
        "messages": messages,
        "max_tokens": 80,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }

    try:
        response = post(TUTOR_URL, payload, timeout=timeout)
        correction = response["choices"][0]["message"]["content"].strip()
        return {
            "text": correction,
            "type": "correction",
            "level": 5,
            "detected_errors": detected_errors,
        }
    except Exception:
        if detected_errors:
            return {
                "text": detected_errors[0] if detected_errors else "Please try again.",
                "type": "rule_based",
                "level": 5,
                "detected_errors": detected_errors,
            }
        return {"text": "Please try again.", "type": "error", "level": 5}


def tutor_adaptive_next(current_problem, is_correct, index=None):
    """Find the next problem for adaptive difficulty.

    If correct: find a neighbor at level + 1 (harder).
    If wrong: find a neighbor at level - 1 (easier) or same level.
    """
    if not index or current_problem.get("level") is None:
        return None

    target_level = current_problem["level"] + (1 if is_correct else -1)
    if target_level < 0:
        target_level = 0

    # Find a similar problem at the target level
    hits = index.search(current_problem, k=10, exclude_id=current_problem["id"])
    for _score, p in hits:
        if p["level"] == target_level and p.get("label"):
            return {"id": p["id"], "level": p["level"], "text": p["text"]}

    # Fallback: any problem at a nearby level
    for delta in [0, 1, -1]:
        alt_level = current_problem["level"] + delta
        for _score, p in hits:
            if p["level"] == alt_level and p.get("label"):
                return {"id": p["id"], "level": p["level"], "text": p["text"]}

    return None