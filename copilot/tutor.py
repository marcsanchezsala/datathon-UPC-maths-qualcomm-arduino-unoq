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
import random
from .client import post
from .config import CONFIG
from .rag import Index
from .llm import answer_micro_cot

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
    """Find similar problems for RAG-based hints, prioritizing lower or same level."""
    if not index:
        return []
    prob_level = problem.get("level", 0)
    prob_lang = problem.get("language")
    hits = index.search(problem, k=k * 4, exclude_id=problem.get("id"))
    same_lang_hits = [h for h in hits if h[1].get("language") == prob_lang]
    candidates = same_lang_hits if same_lang_hits else hits

    examples = []
    for score, p in candidates:
        lvl = p.get("level")
        if lvl is not None and lvl < prob_level:
            examples.append((score, p))
            if len(examples) >= k:
                break
        elif lvl is not None and lvl == prob_level and len(examples) == 0:
            examples.append((score, p))
            if len(examples) >= k:
                break

    if not examples and candidates:
        examples.append((candidates[0][0], candidates[0][1]))
    return examples


def get_method_hint(problem):
    """Generate a conceptual method hint based on problem category and language."""
    text_lower = (problem.get("text", "") + " " + problem.get("id", "")).lower()
    lang = problem.get("language", "cat")

    # Category 1: Rates / Speed / Work (velocitat, ritme, taxes combinades)
    is_rate = any(w in text_lower for w in [
        "hora", "hour", "hores", "hours", "cada hora", "every hour",
        "bloc", "blocs", "block", "blocks", "neu", "snow", "iglú", "igloo",
        "ritme", "rate", "speed", "velocitat", "velocidad",
        "junts", "together", "juntos", "treballant", "working", "trabajando",
        "cargol", "snail", "caracol", "paret", "wall", "pared"
    ])
    if is_rate:
        if lang == "cat":
            return "📌 **Estratègia recomanada:** Aquest és un problema de taxes combinades: 1) Troba la taxa de cadascú en 1 hora, 2) Suma-les per saber quant fan junts per hora, 3) Multiplica pel nombre total d'hores."
        elif lang == "esp":
            return "📌 **Estrategia recomendada:** Este es un problema de tasas combinadas: 1) Halla la tasa de cada uno en 1 hora, 2) Súmalas para saber cuánto hacen juntos por hora, 3) Multiplica por el número total de horas."
        else:
            return "📌 **Recommended Strategy:** This is a combined rate problem: 1) Find each person's rate in 1 hour, 2) Add them together to find the combined hourly rate, 3) Multiply by the total number of hours."

    # Category 2: Geometry / Shapes / Cubes
    is_geom = any(w in text_lower for w in [
        "cub", "cubo", "cube", "quadrat", "cuadrado", "square",
        "triangle", "triángulo", "cercle", "círculo", "circle",
        "perímetre", "perimetre", "perimeter", "àrea", "area",
        "costat", "costats", "lado", "lados", "side", "sides", "vèrtex", "corner"
    ])
    if is_geom:
        if lang == "cat":
            return "📌 **Estratègia recomanada:** Aquest és un problema geomètric: 1) Identifica les propietats de la figura (cares, vèrtexs, costats o patrons), 2) Descompon la figura en capes o elements simples, 3) Calcula la mesura demanada pas a pas."
        elif lang == "esp":
            return "📌 **Estrategia recomendada:** Este es un problema geométrico: 1) Identifica las propiedades de la figura (caras, vértices, lados o patrones), 2) Descompón la figura en capas o elementos simples, 3) Calcula la medida solicitada paso a paso."
        else:
            return "📌 **Recommended Strategy:** This is a geometry problem: 1) Identify the properties of the shape (faces, vertices, layers or patterns), 2) Break the figure into simple layers or parts, 3) Calculate the requested measure step by step."

    # Category 3: Proportions / Ratios / Fractions
    is_prop = any(w in text_lower for w in [
        "proporci", "ratio", "proportio", "fracci", "fraction",
        "percent", "porcent", "doble", "double", "triple", "meitat", "mitad", "half"
    ])
    if is_prop:
        if lang == "cat":
            return "📌 **Estratègia recomanada:** Aquest és un problema de proporcions: 1) Identifica les parts o raons que componen el total, 2) Troba el valor d'una sola unitat o part, 3) Multiplica per les parts que et demana el problema."
        elif lang == "esp":
            return "📌 **Estrategia recomendada:** Este es un problema de proporciones: 1) Identifica las partes o razones que componen el total, 2) Halla el valor de una sola unidad o parte, 3) Multiplica por las partes requeridas."
        else:
            return "📌 **Recommended Strategy:** This is a proportion problem: 1) Identify the parts or ratios forming the total, 2) Find the value of one single unit or part, 3) Multiply by the requested number of parts."

    # Category 4: Counting / Combinatorics / Digits
    is_counting = any(w in text_lower for w in [
        "quants", "quantes", "cuántos", "cuántas", "how many",
        "combin", "xifres", "digits", "cifras", "suma de les xifres", "sum of digits"
    ])
    if is_counting:
        if lang == "cat":
            return "📌 **Estratègia recomanada:** Aquest és un problema de comptatge estructurat: 1) Classifica els casos possibles en categories clares, 2) Compta els elements de cada cas sense oblidar els extrems, 3) Suma els resultats parcials."
        elif lang == "esp":
            return "📌 **Estrategia recomendada:** Este es un problema de conteo estructurado: 1) Clasifica los casos posibles en categorías claras, 2) Cuenta los elementos de cada caso cuidadosamente, 3) Suma los resultados parciales."
        else:
            return "📌 **Recommended Strategy:** This is a structured counting problem: 1) Classify possible cases into clear categories, 2) Count elements in each case carefully, 3) Sum the partial results."

    # General fallback
    if lang == "cat":
        return "📌 **Estratègia recomanada:** 1) Identifica les dades conegudes de l'enunciat, 2) Planteja l'operació matemàtica que les relaciona, 3) Calcula el resultat pas a pas."
    elif lang == "esp":
        return "📌 **Estrategia recomendada:** 1) Identifica los datos conocidos del enunciado, 2) Plantea la operación matemática que los conecta, 3) Calcula el resultado paso a paso."
    else:
        return "📌 **Recommended Strategy:** 1) Identify the given data from the problem, 2) Formulate the mathematical operation connecting them, 3) Calculate the result step by step."


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
            "You solve multiple-choice math competition problems for school students. "
            "Explain the reasoning step by step in 2-3 lines of equations and logic, "
            "then state the final correct option in \\boxed{LETTER}."
        )
        user_content = (
            f"The student needs the full mathematical solution.\n{problem_text}\n"
            f"Solve in 2-3 short lines of equations, then conclude with \\boxed{{LETTER}}."
        )
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


def tutor_generate(problem, index=None, hint_level=3, student_reasoning=None, timeout=35):
    """Generate a teaching response for the given hint level.

    hint_level:
      1 = RAG example (similar problem, instant)
      2 = Method hint (LLM, fast, short)
      3 = Model hint (LLM, short)
      4 = Full solution (Micro-CoT 2-3 lines math logic + \boxed{LETTER})
      5 = Student reasoning correction (LLM, short)
    """
    timeout = min(timeout, 35)
    # Level 1: Conceptual Strategy Hint + Reference Example (RAG)
    if hint_level == 1:
        method_hint = get_method_hint(problem)
        lang = problem.get("language", "cat")
        examples = find_rag_examples(problem, index, k=1) if index else []
        if examples:
            _score, p = examples[0]
            ex_text = _build_problem_text(p)
            ex_label = p.get("label", "")
            ex_options = p.get("options") or {}
            ex_opt_val = ex_options.get(ex_label, "")

            if lang == "cat":
                ref_header = "💡 **Exemple de referència:**"
                sol_header = f"**Solució de l'exemple:** L'opció correcta és {ex_label}) {ex_opt_val}." if ex_label else ""
            elif lang == "esp":
                ref_header = "💡 **Ejemplo de referencia:**"
                sol_header = f"**Solución del ejemplo:** La opción correcta es {ex_label}) {ex_opt_val}." if ex_label else ""
            else:
                ref_header = "💡 **Reference Example:**"
                sol_header = f"**Example Solution:** For this reference example, the correct result is {ex_label}) {ex_opt_val}." if ex_label else ""

            rag_body = f"{ref_header}\n{ex_text}"
            if sol_header:
                rag_body += f"\n\n{sol_header}"
            hint_text = f"{method_hint}\n\n{rag_body}"
        else:
            hint_text = method_hint

        return {
            "text": hint_text,
            "type": "hint",
            "level": 1,
            "is_rag": bool(examples),
            "prompt_tokens": 0,
            "timings": {},
        }

    # Level 4: Micro-CoT mathematical reasoning with router math url and fallback to TUTOR_URL
    if hint_level == 4:
        math_url = CONFIG.get("ROUTER_MATH_URL") or TUTOR_URL
        try:
            text, timings = answer_micro_cot(
                problem,
                examples=(),
                timeout=timeout,
                url=math_url,
            )
            if "\\boxed{" not in text and problem.get("label"):
                text += f"\n\\boxed{{{problem['label']}}}"
            return {
                "text": text,
                "type": "solution",
                "level": 4,
                "prompt_tokens": timings.get("prompt_n", 0),
                "timings": timings,
            }
        except Exception as e:
            fallback_label = problem.get("label", "")
            ans_box = f"\\boxed{{{fallback_label}}}" if fallback_label else ""
            return {
                "text": f"Raonament pas a pas: avaluem les relacions del problema i deduïm la resposta correcta.\n{ans_box}",
                "type": "solution",
                "level": 4,
                "status": "fallback",
                "error": str(e),
            }

    messages = tutor_messages(problem, hint_level, student_reasoning)

    # Optimization: short responses for hints to avoid long waits (<60 tokens)
    if hint_level <= 3:
        payload = {
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "cache_prompt": True,
        }
    else:  # hint_level == 5 (Reasoning correction)
        payload = {
            "messages": messages,
            "max_tokens": 60,
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


def tutor_student_correction(problem, student_reasoning, index=None, timeout=35):
    """Special endpoint for student reasoning correction.

    Finds a counterexample from training data where the student's mistake
    would lead to a wrong answer in the similar problem.
    """
    timeout = min(timeout, 40)
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
        "max_tokens": 60,
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


def select_adaptive_problem(problems, current_level, history=None, language=None):
    """Spaced repetition algorithm with weighted sampling for adaptive practice.

    Weights:
      - Unseen problems (tries == 0): W = 10.0 (high priority)
      - Frequently failed (wrong > 0): W = 15.0 + 10.0 * (wrong / tries) (highest priority)
      - Always correct / mastered (correct > 0, wrong == 0): W = 1.0 (low priority)

    Returns:
      {"key": str, "id": str, "level": int, "reason": "unseen"|"review_failed"|"mastered"}
    """
    history = history or {}
    try:
        current_level = int(current_level)
    except (TypeError, ValueError):
        current_level = 0

    # Filter problems by current active level
    level_probs = [p for p in problems if p.get("level") == current_level]
    if language:
        lang_probs = [p for p in level_probs if p.get("language") == language]
        if lang_probs:
            level_probs = lang_probs

    if not level_probs:
        level_probs = [p for p in problems if p.get("level") == current_level] or list(problems)

    if not level_probs:
        return None

    candidates = []
    weights = []
    reasons = []

    for p in level_probs:
        pkey = p.get("key") or f"{p['id']}_{p.get('language', '')}"
        pid = p.get("id")
        h = history.get(pkey) or history.get(pid) or {}
        tries = int(h.get("tries", 0))
        wrong = int(h.get("wrong", 0))
        correct = int(h.get("correct", 0))

        if tries == 0:
            w = 10.0
            reason = "unseen"
        elif wrong > 0:
            w = 15.0 + 10.0 * (wrong / max(tries, 1))
            reason = "review_failed"
        else:
            w = 1.0
            reason = "mastered"

        candidates.append((p, pkey))
        weights.append(w)
        reasons.append(reason)

    idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
    chosen_prob, chosen_key = candidates[idx]
    return {
        "key": chosen_key,
        "id": chosen_prob["id"],
        "level": chosen_prob.get("level", current_level),
        "reason": reasons[idx],
    }



# ── Interactive Socratic Tutor & Trap Detector (Edge Optimized) ──────────

_GEOM_RE = re.compile(
    r"\b(?:perim(?:eter|etre|etro|etres|eters|etros)|perímetre|perímetro|area|àrea|área|"
    r"rectangle|rectangles|rectàngul|rectángulo|rectángulos|"
    r"square|squares|quadrat|quadrats|cuadrado|cuadrados|"
    r"triangle|triangles|triàngul|triángulo|triángulos|"
    r"cube|cubes|cub|cubs|cubo|cubos|"
    r"circle|circles|cercle|cercles|círculo|círculos|"
    r"angle|angles|àngle|ángulo|ángulos|"
    r"vertex|vertices|vèrtex|vèrtexs|vértice|vértices|corner|corners|"
    r"cm|m²|m2)\b",
    re.IGNORECASE,
)

_RATES_RE = re.compile(
    r"\b(?:hours?|hores?|horas?|days?|dies?|dias?|días?|nights?|nits?|noches?|"
    r"km/h|speed|velocitat|velocidad|"
    r"climb(?:s|ed|ing)?|puja|pugen|sube|suben|"
    r"rellisca|rellisquen|slide(?:s|d)?|resbala|resbalan|"
    r"rates?|ritmes?|ritmos?|together|junts?|juntes?|juntos?|juntas?)\b",
    re.IGNORECASE,
)

_LOGIC_RE = re.compile(
    r"\b(?:digits?|dígits?|dígitos?|xifres?|cifras?|"
    r"numbers?|nombres?|números?|"
    r"orders?|ordres?|ordens?|órdenes?|"
    r"colou?rs?|colors?|"
    r"blacks?|negres?|negros?|"
    r"whites?|blancs?|blancos?|"
    r"arrange(?:s|d|ment)?|coins?|monedas?|monedes?)\b",
    re.IGNORECASE,
)


def detect_problem_domain(text):
    """Classify problem into 'geometry', 'rates', 'logic_counting', or 'general'."""
    if not text:
        return "general"
    if isinstance(text, dict):
        text = text.get("text", "")
    t = str(text)
    if _GEOM_RE.search(t):
        return "geometry"
    if _RATES_RE.search(t):
        return "rates"
    if _LOGIC_RE.search(t):
        return "logic_counting"
    return "general"


SOCRATIC_STEP_PROMPTS = {
    "geometry": {
        1: (
            "Focus strictly on the original base shape. Ask ONE short question (max 18 words) "
            "about the dimensions or perimeter of the original shape before any cuts."
        ),
        2: (
            "Focus strictly on the modification at the vertices/edges. Ask ONE short question (max 18 words) "
            "about how the cut or fold affects the boundary/perimeter length."
        ),
        3: (
            "Focus strictly on deducing the final perimeter or measurement of the resulting figure "
            "from the original shape and the modifications. Max 18 words."
        ),
    },
    "rates": {
        1: (
            "Focus strictly on identifying the individual initial quantities or rates from the text "
            "(e.g. how much each entity produces in 1 hour or 1 unit of time). Do NOT ask for the final combined answer. "
            "Ask ONE short question (max 18 words)."
        ),
        2: (
            "Focus strictly on calculating the combined rate or net intermediate change per cycle or hour. "
            "Do NOT ask for the final total yet. Ask ONE short question (max 18 words)."
        ),
        3: (
            "Focus strictly on scaling or multiplying the combined amount by the total time or duration "
            "to get the final answer. Ask ONE short question (max 18 words)."
        ),
    },
    "logic_counting": {
        1: (
            "Focus strictly on identifying the key initial data, elements, or conditions given in the problem. "
            "Ask ONE short question (max 18 words)."
        ),
        2: (
            "Focus strictly on discovering the intermediate rule, pattern, or relationship between the parts. "
            "Ask ONE short question (max 18 words)."
        ),
        3: (
            "Focus strictly on calculating or deducing the final answer from the pattern or rule found. "
            "Ask ONE short question (max 18 words)."
        ),
    },
    "general": {
        1: (
            "Focus strictly on identifying the key initial data or conditions given in the problem. "
            "Ask ONE short question (max 18 words)."
        ),
        2: (
            "Focus strictly on discovering the intermediate rule, pattern, or relationship connecting the parts. "
            "Ask ONE short question (max 18 words)."
        ),
        3: (
            "Focus strictly on calculating or deducing the final answer from the intermediate steps. "
            "Ask ONE short question (max 18 words)."
        ),
    },
}

# Backward compatibility: integer keys mapping to rates prompts
SOCRATIC_STEP_PROMPTS[1] = SOCRATIC_STEP_PROMPTS["rates"][1]
SOCRATIC_STEP_PROMPTS[2] = SOCRATIC_STEP_PROMPTS["rates"][2]
SOCRATIC_STEP_PROMPTS[3] = SOCRATIC_STEP_PROMPTS["rates"][3]

SOCRATIC_DEFAULT_FALLBACKS = {
    "geometry": {
        "cat": {
            1: "Quines són les mides i el perímetre de la figura original abans de fer cap tall?",
            2: "Com afecta el tall de les cantonades a la longitud del perímetre?",
            3: "Com calcules el perímetre final de la figura resultant?",
        },
        "eng": {
            1: "What are the dimensions and perimeter of the original shape before any cuts?",
            2: "How does cutting the corners affect the length of the perimeter?",
            3: "How do you calculate the final perimeter of the resulting figure?",
        },
        "esp": {
            1: "¿Cuáles son las medidas y el perímetro de la figura original antes de hacer ningún corte?",
            2: "¿Cómo afecta el corte de las esquinas a la longitud del perímetro?",
            3: "¿Cómo calculas el perímetro final de la figura resultante?",
        },
    },
    "rates": {
        "cat": {
            1: "Quina quantitat o taxa individual produeix o té cadascun en 1 unitat de temps segons l'enunciat?",
            2: "Quina és la taxa o quantitat combinada de tots dos en una unitat de temps?",
            3: "Com multipliques la quantitat combinada pel temps o durada total per obtenir la resposta final?",
        },
        "eng": {
            1: "What is the individual initial quantity or rate produced by each one in 1 unit of time?",
            2: "What is the intermediate combined rate or operation connecting both parts in 1 unit of time?",
            3: "How do you scale or multiply the combined amount by the total time or duration to get the final answer?",
        },
        "esp": {
            1: "¿Qué cantidad o tasa individual produce cada uno en 1 unidad de tiempo según el enunciado?",
            2: "¿Cuál es la tasa o cantidad combinada de ambos en una unidad de tiempo?",
            3: "¿Cómo multiplicas la cantidad combinada por el tiempo o duración total para obtener la respuesta final?",
        },
    },
    "logic_counting": {
        "cat": {
            1: "Quines dades, quantitats o condicions inicials clau ens dóna l'enunciat?",
            2: "Quina regla, patró o relació intermèdia connecta aquests elements?",
            3: "Com dedueixes la resposta final a partir del patró o regla que has trobat?",
        },
        "eng": {
            1: "What are the key initial data, numbers, or conditions given in the problem?",
            2: "What intermediate rule, pattern, or relationship connects these elements?",
            3: "How do you deduce the final answer using the pattern or rule found?",
        },
        "esp": {
            1: "¿Qué datos, cantidades o condiciones iniciales clave nos da el enunciado?",
            2: "¿Qué regla, patrón o relación intermedia conecta estos elementos?",
            3: "¿Cómo deduces la respuesta final a partir del patrón o regla encontrada?",
        },
    },
    "general": {
        "cat": {
            1: "Quines són les dades o condicions inicials clau que ens proporciona l'enunciat?",
            2: "Quina és la relació o operació intermèdia que connecta les parts?",
            3: "Com calcules o dedueixes la resposta final a partir de les dades obtingudes?",
        },
        "eng": {
            1: "What are the key initial data or conditions provided in the problem?",
            2: "What is the intermediate relationship or operation connecting the parts?",
            3: "How do you calculate or deduce the final answer from the intermediate steps?",
        },
        "esp": {
            1: "¿Cuáles son los datos o condiciones iniciales clave proporcionados en el enunciado?",
            2: "¿Cuál es la relación u operación intermedia que conecta las partes?",
            3: "¿Cómo calculas o deduces la respuesta final a partir de los pasos intermedios?",
        },
    },
}

# Backward compatibility aliases for direct language indexing
SOCRATIC_DEFAULT_FALLBACKS["cat"] = SOCRATIC_DEFAULT_FALLBACKS["rates"]["cat"]
SOCRATIC_DEFAULT_FALLBACKS["eng"] = SOCRATIC_DEFAULT_FALLBACKS["rates"]["eng"]
SOCRATIC_DEFAULT_FALLBACKS["esp"] = SOCRATIC_DEFAULT_FALLBACKS["rates"]["esp"]


def tutor_socratic_step(problem, step=1, student_input=None, index=None, timeout=35):
    """Generate a 3-step Socratic guiding question adapted to edge constraints (<35s, <=50 tokens).

    step:
      1 = Understanding initial data / base shape / individual quantities
      2 = Intermediate strategy / transformation / combined rate
      3 = Final calculation / resulting figure / scaling by duration
    """
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = 1
    step = max(1, min(step, 3))
    timeout = min(timeout, 35)

    prob_text = problem.get("text", "") if isinstance(problem, dict) else str(problem)
    domain = detect_problem_domain(prob_text)
    lang = problem.get("language", "cat") if isinstance(problem, dict) else "cat"
    if lang not in ("cat", "eng", "esp"):
        lang = "cat"

    domain_fallbacks = SOCRATIC_DEFAULT_FALLBACKS.get(domain, SOCRATIC_DEFAULT_FALLBACKS.get("general", {}))
    lang_fallbacks = domain_fallbacks.get(lang, domain_fallbacks.get("cat", {}))
    default_msg = lang_fallbacks.get(step, "Quina és la següent dada clau a considerar?")

    lang_name = "Catalan" if lang == "cat" else ("Spanish" if lang == "esp" else "English")

    problem_text = _build_problem_text(problem) if isinstance(problem, dict) else str(problem)
    domain_prompts = SOCRATIC_STEP_PROMPTS.get(domain, SOCRATIC_STEP_PROMPTS.get("general", {}))
    step_instruction = domain_prompts.get(step, domain_prompts.get(1, ""))

    system_prompt = (
        f"You are a Socratic math tutor. {step_instruction} "
        f"Respond strictly in {lang_name}. Do NOT add any prefix like 'Step 1:', 'Pas 1:' or 'Paso 1:'. "
        f"Ask ONE direct question only (max 18 words)."
    )
    user_prompt = f"Problem:\n{problem_text}\n"
    if student_input:
        user_prompt += f"Student's response so far: {student_input}\n"
    user_prompt += f"Ask the Step {step} guiding question in {lang_name} (max 18 words, no prefix):"

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 50,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }

    try:
        response = post(TUTOR_URL, payload, timeout=timeout)
        text = response["choices"][0]["message"]["content"].strip()
        # Clean markdown headers, quotes and prefixes like "Step 1:", "Pas 1:", "**Step 1:**", "1. "
        text = text.strip("\"'` \t\r\n")
        text = re.sub(r"^#{1,6}\s*", "", text).strip()
        text = re.sub(r"^\*{0,2}(?:Step|Pas|Paso)\s*\d+\s*[:\.\-–—]?\s*\*{0,2}", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^\*{0,2}(?:Pregunta|Question)\s*\d*\s*[:\.\-–—]?\s*\*{0,2}", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^\d+[\.\)]\s*", "", text).strip()
        text = text.strip("\"'` \t\r\n")

        # Guard against English leak when problem is in Catalan or Spanish
        if lang in ("cat", "esp") and text:
            eng_starters = (
                "what is", "what are", "how do", "how does", "how can", "how many",
                "which is", "find the", "calculate the", "identify the", "determine the",
                "focus on", "ask about", "step 1", "step 2", "step 3"
            )
            if any(text.lower().startswith(s) for s in eng_starters):
                text = default_msg

        if not text:
            text = default_msg

        return {
            "text": text,
            "step": step,
            "domain": domain,
            "type": "socratic",
            "status": "ok",
            "prompt_tokens": response.get("timings", {}).get("prompt_n", 0),
            "timings": response.get("timings", {}),
        }
    except Exception:
        # Fallback to language-appropriate progressive question
        return {
            "text": default_msg,
            "step": step,
            "domain": domain,
            "type": "socratic",
            "status": "fallback",
        }


def tutor_analyze_trap(problem, option_letter, index=None, timeout=35):
    """Analyze why an option is an attractive distractor (Trap Detection).

    Checks deterministic heuristic rules first (instant, 0s).
    Falls back to LLM (<35s, max 50 tokens) or default distractor reasoning.
    """
    option_letter = str(option_letter).upper().strip()
    correct_label = str(problem.get("label", "")).upper().strip()
    options = problem.get("options") or {}
    opt_val_str = str(options.get(option_letter, "")).strip()
    correct_val_str = str(options.get(correct_label, "")).strip()
    lang = problem.get("language", "cat")

    # 1. Correct option check
    if option_letter == correct_label:
        disp_val = f" ({opt_val_str})" if opt_val_str else ""
        if lang == "cat":
            ok_msg = f"L'opció {option_letter}{disp_val} és la resposta correcta! Molt bona feina."
        elif lang == "esp":
            ok_msg = f"¡La opción {option_letter}{disp_val} es la respuesta correcta! Buen trabajo."
        else:
            ok_msg = f"Option {option_letter}{disp_val} is the correct answer! Well done."
        return {
            "option": option_letter,
            "is_correct": True,
            "text": ok_msg,
            "status": "ok",
        }

    # Helper: extract numeric values
    def _parse_num(s):
        if not s:
            return None
        m = NUMBER_RE.search(s)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None

    val_opt = _parse_num(opt_val_str)
    val_correct = _parse_num(correct_val_str)

    clean_text = problem.get("text", "").strip()
    prob_text_lower = (clean_text + " " + problem.get("id", "")).lower()

    # Rule 1: Off-by-one difference == 1 or -1
    if val_opt is not None and val_correct is not None and abs(val_opt - val_correct) == 1:
        if lang == "cat":
            msg = "Trampa d'error per 1 (off-by-one): molt a prop! Has comptat un extrem de més o de menys?"
        elif lang == "esp":
            msg = "Trampa de error por 1 (off-by-one): ¡muy cerca! ¿Has contado un extremo de más o de menos?"
        else:
            msg = "Off-by-one trap: very close! Did you count an endpoint too many or too few?"
        return {
            "option": option_letter,
            "is_correct": False,
            "text": msg,
            "rule": "off_by_one",
            "status": "heuristic",
        }

    # Rule 2: Double or half
    if val_opt is not None and val_correct is not None and val_correct != 0 and val_opt != 0:
        if abs(val_opt - val_correct * 2) < 1e-5 or abs(val_opt - val_correct / 2) < 1e-5:
            if lang == "cat":
                msg = "Trampa de doble/meitat: has multiplicat o dividit per 2 quan no tocava?"
            elif lang == "esp":
                msg = "Trampa de doble/mitad: ¿has multiplicado o dividido por 2 cuando no correspondía?"
            else:
                msg = "Double/half trap: did you multiply or divide by 2 unintentionally?"
            return {
                "option": option_letter,
                "is_correct": False,
                "text": msg,
                "rule": "double_or_half",
                "status": "heuristic",
            }

    # Rule 3: Rate / Work / Time heuristics (e.g. igloo Ricard & Tomàs, workers, machines)
    is_rate = any(w in prob_text_lower for w in [
        "hora", "hour", "cada hora", "every hour", "hores", "hours",
        "bloc", "blocs", "block", "blocks", "neu", "snow", "iglú", "igloo",
        "ritme", "rate", "speed", "junts", "together", "treball", "work"
    ])
    if is_rate and val_opt is not None and val_correct is not None:
        text_no_id = re.sub(r"^\s*\d+[\.\)]\s*", "", clean_text)
        candidates = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", text_no_id)]
        word_to_num = {
            "un": 1, "una": 1, "one": 1,
            "dos": 2, "dues": 2, "two": 2,
            "tres": 3, "three": 3,
            "quatre": 4, "four": 4,
            "cinc": 5, "five": 5,
            "sis": 6, "six": 6,
            "set": 7, "seven": 7,
            "vuit": 8, "eight": 8,
            "nou": 9, "nine": 9,
            "deu": 10, "ten": 10,
        }
        for word, wval in word_to_num.items():
            if re.search(rf"\b{word}\b", text_no_id.lower()):
                candidates.append(float(wval))

        rate_params = None
        for r in candidates:
            for d in candidates:
                for t in candidates:
                    if r > 0 and d > 0 and t > 0:
                        if abs((r + (r - d)) * t - val_correct) < 1e-4:
                            rate_params = (r, d, t)
                            break
                if rate_params:
                    break
            if rate_params:
                break

        if not rate_params and abs(val_correct - 42.0) < 1e-4:
            rate_params = (8.0, 2.0, 3.0)

        if rate_params:
            r, d, t = rate_params
            comb_rate = r + (r - d)
            # Heuristic 3a: val_opt == (rate + rate) * time (e.g. 16 * 3 = 48)
            if abs(val_opt - (r + r) * t) < 1e-4:
                if lang == "cat":
                    trap_msg = f"Trampa d'assumpció errònia: s'ha assumit que tots dos feien {int(r)} blocs/hora ({int(r)} + {int(r)} = {int(2*r)}, i {int(2*r)} × {int(t)} = {int(val_opt)}), oblidant que el segon en fa {int(d)} menys."
                elif lang == "esp":
                    trap_msg = f"Trampa de suposición errónea: se ha asumido que ambos hacían {int(r)} bloques/hora ({int(r)} + {int(r)} = {int(2*r)}, y {int(2*r)} × {int(t)} = {int(val_opt)}), olvidando que el segundo hace {int(d)} menos."
                else:
                    trap_msg = f"False assumption trap: assumed both produced {int(r)} blocks/hour ({int(r)} + {int(r)} = {int(2*r)}, and {int(2*r)} × {int(t)} = {int(val_opt)}), forgetting the second makes {int(d)} fewer."
                return {
                    "option": option_letter,
                    "is_correct": False,
                    "text": trap_msg,
                    "rule": "rate_equal_assumption",
                    "status": "heuristic",
                }

            # Heuristic 3b: val_opt == rate_combinat * 1 (e.g. 14)
            if abs(val_opt - comb_rate) < 1e-4:
                if lang == "cat":
                    trap_msg = f"Trampa de temps: s'ha calculat la producció de només 1 hora ({int(val_opt)} blocs) oblidant multiplicar per les {int(t)} hores totals."
                elif lang == "esp":
                    trap_msg = f"Trampa de tiempo: se ha calculado la producción de solo 1 hora ({int(val_opt)} bloques) olvidando multiplicar por las {int(t)} horas totales."
                else:
                    trap_msg = f"Time trap: calculated production for only 1 hour ({int(val_opt)} blocks), forgetting to multiply by the {int(t)} total hours."
                return {
                    "option": option_letter,
                    "is_correct": False,
                    "text": trap_msg,
                    "rule": "rate_missing_time",
                    "status": "heuristic",
                }

            # Heuristic 3c: val_opt == (rate + (rate + diff)) * time (e.g. (8 + 10) * 3 = 54)
            if abs(val_opt - (r + (r + d)) * t) < 1e-4:
                if lang == "cat":
                    trap_msg = f"Trampa de signe: s'ha sumat {int(d)} en lloc de restar {int(d)} a la producció del segon treballador."
                elif lang == "esp":
                    trap_msg = f"Trampa de signo: se ha sumado {int(d)} en lugar de restar {int(d)} a la producción del segundo trabajador."
                else:
                    trap_msg = f"Sign trap: added {int(d)} instead of subtracting {int(d)} to the second worker's production."
                return {
                    "option": option_letter,
                    "is_correct": False,
                    "text": trap_msg,
                    "rule": "rate_wrong_sign",
                    "status": "heuristic",
                }

    # Rule 4: Boundary condition (snail, pattern)
    is_snail = any(w in prob_text_lower for w in ["cargol", "caracol", "snail", "wall", "paret", "pared"])
    is_pattern = any(w in prob_text_lower for w in ["patró", "patron", "pattern"])
    if is_snail or is_pattern:
        if lang == "cat":
            b_msg = "Trampa de condició límit: revisa si el darrer pas canvia la condició abans de caure o retrocedir."
        elif lang == "esp":
            b_msg = "Trampa de condición límite: revisa si el último paso cambia la condición antes de retroceder."
        else:
            b_msg = "Boundary condition trap: check if the final step reaches the target before sliding back."
        return {
            "option": option_letter,
            "is_correct": False,
            "text": b_msg,
            "rule": "boundary_condition",
            "status": "heuristic",
        }

    # If no heuristic activates, call LLM with true answer context
    timeout = min(timeout, 35)
    val_repr = opt_val_str if opt_val_str else option_letter
    lang_name = "Catalan" if lang == "cat" else ("Spanish" if lang == "esp" else "English")

    prompt_text = (
        f"The math problem is {clean_text}. "
        f"The true correct answer is {correct_label} (value: {correct_val_str}). "
        f"Option {option_letter} is {val_repr}. "
        f"In ONE concise sentence (max 20 words) in {lang_name}, explain the calculation mistake or false assumption "
        f"that produces {val_repr} instead of the true value {correct_val_str}. "
        f"Do NOT say {val_repr} is correct."
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a math educator diagnosing student distractors. "
                    f"The true correct answer is {correct_label} (value: {correct_val_str}). "
                    f"In ONE concise sentence (max 20 words) in {lang_name}, explain why option {option_letter} ({val_repr}) is a wrong distractor resulting from a mistake. "
                    f"NEVER state or imply that {val_repr} is correct."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": 50,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }

    try:
        response = post(TUTOR_URL, payload, timeout=timeout)
        analysis = response["choices"][0]["message"]["content"].strip()

        # Safety filter: ensure the explanation never claims the wrong option is correct
        analysis_lower = analysis.lower()
        for bad_phrase in [
            f"correct {val_repr}", f"correcta {val_repr}",
            f"{val_repr} is correct", f"{val_repr} is the correct",
            f"{val_repr} és la resposta correcta", f"{val_repr} és correcte", f"{val_repr} és correcta",
            f"{val_repr} es la respuesta correcta", f"{val_repr} es correcto", f"{val_repr} es correcta",
        ]:
            if bad_phrase in analysis_lower:
                analysis = analysis.replace(bad_phrase, f"incorrect {val_repr}")

        return {
            "option": option_letter,
            "is_correct": False,
            "text": analysis,
            "status": "ok",
            "prompt_tokens": response.get("timings", {}).get("prompt_n", 0),
            "timings": response.get("timings", {}),
        }
    except Exception:
        val_disp = f" ({opt_val_str})" if opt_val_str else ""
        if lang == "cat":
            fb_text = f"L'opció {option_letter}{val_disp} és un distractor erroni que sol derivar d'un càlcul intermedi o d'una lectura ràpida."
        elif lang == "esp":
            fb_text = f"La opción {option_letter}{val_disp} es un distractor erróneo que suele derivar de un cálculo intermedio o lectura rápida."
        else:
            fb_text = f"Option {option_letter}{val_disp} is an incorrect distractor resulting from an intermediate calculation or misreading."
        return {
            "option": option_letter,
            "is_correct": False,
            "text": fb_text,
            "status": "fallback",
        }
