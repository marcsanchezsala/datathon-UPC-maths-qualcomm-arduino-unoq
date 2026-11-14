"""Teach track: guide the student without giving the answer away.

Kept apart from predict.py, so the official run never changes by accident.
Standard library only. Every function that needs neighbours takes an index
with the same interface as rag.Index: search(problem, k, exclude_id).

Flow when the student answers wrong (on_wrong_answer):
  1. Rules diagnose a typical slip (off by 1-2, missing a datum, x/÷, ², √, sign).
  2. Counterexample from train: a similar problem where the SAME kind of slip
     lands on one of its wrong options. No model, well under a second.
  3. If none and the student's explanation passes the filter: the LLM applies
     that reasoning to a similar problem. A wrong letter is a counterexample.
  4. Otherwise: an easier similar problem (hint ladder, rung 1), or just "try again".
The current problem's correct answer is never shown.
"""
import json
import math
import os
import re
import sqlite3
import time

from .client import post
from .config import CONFIG
from .llm import URL, format_problem

SEED = int(CONFIG["SEED"])
MIN_SCORE = 0.6          # similarity threshold for bge-small; tune it with teach_eval.py
K_NEIGHBOURS = 30
SAME_LEVEL = True        # rung 1: allow a similar problem at the SAME level when none is easier

# ---------------------------------------------------------------------------
# Messages (eng, cat, esp). {placeholders} are filled in by the functions below.
# ---------------------------------------------------------------------------
MSG = {
    "eng": {
        "counter": "In this similar problem:\n{problem}\nIf you solve it the same way as yours, "
                   "you get {wrong}) {wrong_val}, but the correct answer is {right}) {right_val}. "
                   "What do you think you're missing? Try yours again.",
        "counter_llm": "In this similar problem:\n{problem}\nWith your reasoning you would choose "
                       "{wrong}) {wrong_val}, but the correct answer is {right}) {right_val}. "
                       "Where does your reasoning break? Try yours again.",
        "easier": "Look at this similar, easier problem:\n{problem}\nIts answer is {right}) {right_val}. "
                  "Now try yours again.",
        "similar": "Look at this similar problem:\n{problem}\nIts answer is {right}) {right_val}. "
                   "Now try yours again.",
        "saved": "Thanks, we've saved your explanation for your teacher.",
        "retry": "Not quite. Try again.",
        "topic": "This looks like a {topic} problem. {tip}",
        "no_more": "That's all the hints. Ask for the solution if you want it.",
        "solution": "The answer is {right}) {right_val}.",
    },
    "cat": {
        "counter": "En aquest problema semblant:\n{problem}\nSi el resols igual que el teu, "
                   "et surt {wrong}) {wrong_val}, però la resposta correcta és {right}) {right_val}. "
                   "Què creus que et deixes? Torna a provar el teu.",
        "counter_llm": "En aquest problema semblant:\n{problem}\nAmb el teu raonament triaries "
                       "{wrong}) {wrong_val}, però la resposta correcta és {right}) {right_val}. "
                       "On falla el raonament? Torna a provar el teu.",
        "easier": "Mira aquest problema semblant i més fàcil:\n{problem}\nLa resposta és {right}) {right_val}. "
                  "Ara torna a provar el teu.",
        "similar": "Mira aquest problema semblant:\n{problem}\nLa resposta és {right}) {right_val}. "
                   "Ara torna a provar el teu.",
        "saved": "Gràcies, hem desat la teva explicació per al professor.",
        "retry": "No és correcta. Torna-ho a provar.",
        "topic": "Sembla un problema de {topic}. {tip}",
        "no_more": "Ja no queden més pistes. Demana la solució si la vols.",
        "solution": "La resposta és {right}) {right_val}.",
    },
    "esp": {
        "counter": "En este problema parecido:\n{problem}\nSi lo resuelves igual que el tuyo, "
                   "te sale {wrong}) {wrong_val}, pero la respuesta correcta es {right}) {right_val}. "
                   "¿Qué crees que te dejas? Vuelve a intentar el tuyo.",
        "counter_llm": "En este problema parecido:\n{problem}\nCon tu razonamiento elegirías "
                       "{wrong}) {wrong_val}, pero la respuesta correcta es {right}) {right_val}. "
                       "¿Dónde falla tu razonamiento? Vuelve a intentar el tuyo.",
        "easier": "Mira este problema parecido y más fácil:\n{problem}\nSu respuesta es {right}) {right_val}. "
                  "Ahora vuelve a intentar el tuyo.",
        "similar": "Mira este problema parecido:\n{problem}\nSu respuesta es {right}) {right_val}. "
                   "Ahora vuelve a intentar el tuyo.",
        "saved": "Gracias, hemos guardado tu explicación para tu profesor.",
        "retry": "No es correcta. Vuelve a intentarlo.",
        "topic": "Parece un problema de {topic}. {tip}",
        "no_more": "No quedan más pistas. Pide la solución si la quieres.",
        "solution": "La respuesta es {right}) {right_val}.",
    },
}

# Topics by keywords (there is no topic field in M3Kang). Order = priority.
TOPICS = [
    ("geometry", r"\b(area|perimeter|triangle|square|rectangle|circle|angle|side|cube|"
                 r"àrea|perímetre|triangle|quadrat|rectangle|cercle|angle|costat|cub|"
                 r"área|perímetro|triángulo|cuadrado|rectángulo|círculo|ángulo|lado)\b|cm²|m²|°"),
    ("digits", r"\b(digit|digits|xifra|xifres|cifra|cifras)\b"),
    ("time", r"\b(clock|hour|hours|minute|minutes|day|days|week|rellotge|hora|hores|minut|minuts|"
             r"dia|dies|setmana|reloj|horas|minuto|minutos|día|días|semana)\b"),
    ("logic", r"\b(true|false|lie|lies|liar|truth|veritat|mentida|menteix|verdad|mentira|miente)\b"),
    ("counting", r"\b(how many|quants|quantes|cuántos|cuántas|ways|maneres|maneras)\b"),
    ("arithmetic", r"[+×÷*/=]|\b(sum|product|difference|suma|producte|diferència|producto|diferencia)\b"),
]
TOPIC_NAME = {
    "eng": {"geometry": "geometry", "digits": "digits", "time": "time", "logic": "logic",
            "counting": "counting", "arithmetic": "arithmetic"},
    "cat": {"geometry": "geometria", "digits": "xifres", "time": "temps", "logic": "lògica",
            "counting": "comptar", "arithmetic": "aritmètica"},
    "esp": {"geometry": "geometría", "digits": "cifras", "time": "tiempo", "logic": "lógica",
            "counting": "contar", "arithmetic": "aritmética"},
}
TOPIC_TIP = {
    "eng": {"geometry": "Draw the figure and write the lengths you know on it.",
            "digits": "Write the digits in positions (tens, units) and go one by one.",
            "time": "Convert everything to the same unit first.",
            "logic": "Suppose one statement is true and check whether the others agree.",
            "counting": "List the cases in order, so you don't miss or repeat any.",
            "arithmetic": "Do the operations step by step, in the right order."},
    "cat": {"geometry": "Dibuixa la figura i escriu-hi les longituds que coneixes.",
            "digits": "Escriu les xifres per posicions (desenes, unitats) i ves una a una.",
            "time": "Primer passa-ho tot a la mateixa unitat.",
            "logic": "Suposa que una afirmació és certa i comprova si les altres hi encaixen.",
            "counting": "Fes la llista dels casos en ordre, per no deixar-te'n ni repetir-ne cap.",
            "arithmetic": "Fes les operacions pas a pas, en l'ordre correcte."},
    "esp": {"geometry": "Dibuja la figura y escribe en ella las longitudes que conoces.",
            "digits": "Escribe las cifras por posiciones (decenas, unidades) y ve una a una.",
            "time": "Primero pásalo todo a la misma unidad.",
            "logic": "Supón que una afirmación es cierta y comprueba si las demás encajan.",
            "counting": "Haz la lista de los casos en orden, para no saltarte ni repetir ninguno.",
            "arithmetic": "Haz las operaciones paso a paso, en el orden correcto."},
}


def msg(lang, key, **kw):
    return MSG.get(lang, MSG["eng"])[key].format(**kw)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
WORDNUM = {  # no "one"/"un"/"una": in Catalan and Spanish they are articles
    "eng": {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100},
    "cat": {"dos": 2, "dues": 2, "tres": 3, "quatre": 4, "cinc": 5, "sis": 6, "set": 7,
            "vuit": 8, "nou": 9, "deu": 10, "onze": 11, "dotze": 12, "vint": 20, "cent": 100},
    "esp": {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
            "nueve": 9, "diez": 10, "once": 11, "doce": 12, "veinte": 20, "cien": 100},
}
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _norm(text):
    return str(text).replace("−", "-").replace("–", "-").replace(",", ".")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def parse_num(value):
    """'12 cm' -> 12.0, '1/2' -> 0.5, '3,5' -> 3.5, 'a square' -> None."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)(?:\s*/\s*(\d+))?", _norm(value))
    if not m:
        return None
    num = float(m.group(1))
    if m.group(2):
        den = float(m.group(2))
        return num / den if den else None
    return num


def strip_header(text):
    """Drop the '13. ' that numbers the statement."""
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", text)


def statement_numbers(text, lang):
    t = _norm(strip_header(text))
    nums = {parse_num(x) for x in _NUM.findall(t)}
    words = WORDNUM.get(lang, {})
    nums |= {float(words[w]) for w in re.findall(r"[a-zà-ÿ]+", t.lower()) if w in words}
    return {n for n in nums if n is not None}


def fmt(x):
    return f"{x:g}"


# ---------------------------------------------------------------------------
# 1. Diagnose typical slips with rules
# ---------------------------------------------------------------------------
ERROR_TYPES = ("despiste", "suma_dato", "mult_dato", "cuadrado", "raiz", "signo")


def classify(x, c, data):
    """Slips compatible with choosing x when the answer is c, by priority: [(type, param)]."""
    errs, d = [], x - c
    if abs(d) in (1, 2) and not ({1.0, 2.0} & data):
        errs.append(("despiste", d))
    if any(v and (close(d, v) or close(d, -v)) for v in data):
        errs.append(("suma_dato", math.copysign(1, d)))
    for v in sorted(data):
        if v in (0, 1):
            continue
        if close(x, c * v):
            errs.append(("mult_dato", "mul"))
            break
        if close(x, c / v):
            errs.append(("mult_dato", "div"))
            break
    if c not in (0, 1) and close(x, c ** 2):
        errs.append(("cuadrado", None))
    if c > 0 and c != 1 and close(x, math.sqrt(c)):
        errs.append(("raiz", None))
    if c != 0 and close(x, -c):
        errs.append(("signo", None))
    return errs


def wrong_values(err, right, ndata):
    """Apply the same KIND of slip to a neighbour's correct value."""
    kind, p = err
    if kind == "despiste":
        return [right + p]
    if kind == "suma_dato":
        return [right + p * v for v in ndata if v]
    if kind == "mult_dato":
        return [right * v if p == "mul" else right / v for v in ndata if v not in (0, 1)]
    if kind == "cuadrado":
        return [right ** 2]
    if kind == "raiz":
        return [math.sqrt(right)] if right > 0 else []
    if kind == "signo":
        return [-right]
    return []


def diagnose(problem, chosen):
    """Return (x, c, errors) or None when the options aren't numbers."""
    opts = problem.get("options") or {}
    x, c = parse_num(opts.get(chosen, "")), parse_num(opts.get(problem.get("label"), ""))
    if x is None or c is None:
        return None
    return x, c, classify(x, c, statement_numbers(problem["text"], problem["language"]))


# ---------------------------------------------------------------------------
# Library: show neighbours in the student's language (same id, other language)
# ---------------------------------------------------------------------------
class Library:
    """Problems by (id, language): load train AND the bank of problems shown to students
    (e.g. val), so any problem can be found in another language by its id."""

    def __init__(self, problems=()):
        self.by_key = {(p["id"], p["language"]): p for p in problems}
        # Known words, for the invented-word check on the student's explanation
        self.vocab = {w.lower() for p in self.by_key.values()
                      for w in re.findall(r"[^\W\d_]+", p["text"]) if len(w) > 2}

    @classmethod
    def load(cls, *paths):
        paths = paths or ("data/train.jsonl", "data/val.jsonl")
        rows = []
        for path in paths:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    rows += [json.loads(line) for line in f if line.strip()]
        return cls(rows)

    def localize(self, problem, lang):
        return self.by_key.get((problem["id"], lang), problem)


def show(problem):
    return format_problem(compact_problem(problem))


SEARCH_LANG = "eng"  # bge-small only understands English


def neighbours(problem, index, k=None, library=None):
    """Similar TRAINING problems, never the problem itself in any language.

    bge-small is an English model: if the English version of the same problem
    (same id) is known, search with it. The neighbours are shown later in the
    student's language with Library.localize.
    """
    query = library.localize(problem, SEARCH_LANG) if library else problem
    return index.search(query, k=k or K_NEIGHBOURS, exclude_id=problem["id"])


def safe_to_show(neighbour, problem):
    """A neighbour whose correct answer has the same value as the current one would give it away."""
    opts = problem.get("options") or {}
    mine = str(opts.get(problem.get("label"), "")).strip()
    theirs = str((neighbour.get("options") or {}).get(neighbour.get("label"), "")).strip()
    if not mine or not theirs:
        return True
    a, b = parse_num(mine), parse_num(theirs)
    if a is not None and b is not None:
        return not close(a, b)
    return mine.lower() != theirs.lower()


# ---------------------------------------------------------------------------
# 2. Counterexample without the model
# ---------------------------------------------------------------------------
def counterexample(problem, chosen, hits, min_score=None):
    min_score = MIN_SCORE if min_score is None else min_score
    found = diagnose(problem, chosen)
    if not found or not found[2]:
        return None
    _x, _c, errs = found
    for score, p in hits:
        if score < min_score or not p.get("options") or not safe_to_show(p, problem):
            continue
        right = parse_num(p["options"].get(p["label"], ""))
        if right is None:
            continue
        ndata = statement_numbers(p["text"], p["language"])
        for err in errs:
            for w in wrong_values(err, right, ndata):
                for letter, v in p["options"].items():
                    nv = parse_num(v)
                    if letter != p["label"] and nv is not None and close(nv, w):
                        return {"neighbour": p, "score": score, "error": err[0],
                                "wrong": letter, "right": p["label"]}
    return None


# ---------------------------------------------------------------------------
# 3. Input preprocessing with grounding: fewer tokens, same numbers
# ---------------------------------------------------------------------------
def compact(text):
    """Drop the header number ('13. ') and repeated spaces. Words are never removed:
    'not', 'except', 'only'... change the problem."""
    return re.sub(r"\s+", " ", strip_header(text)).strip()


def numbers_and_symbols(text):
    return sorted(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?|[+\-×÷=<>%°/*^²³√]", _norm(text)))


def compact_problem(problem):
    """Grounding on the input: the text sent to the model keeps every number and
    symbol of the statement (header excluded). If anything were lost, the
    original text is kept instead."""
    original = problem["text"]
    short = compact(original)
    if numbers_and_symbols(strip_header(original)) != numbers_and_symbols(short):
        short = original
    return {**problem, "text": short}


def grounded(text, problem, allowed=(0.0, 1.0, 2.0)):
    """Grounding on the output: every number the model writes must be in the
    statement (or be 0, 1, 2). A hint with an invented number is rejected."""
    data = statement_numbers(problem["text"], problem["language"]) | set(allowed)
    for m in _NUM.findall(_norm(text)):
        v = parse_num(m)
        if v is not None and not any(close(v, d) for d in data):
            return False
    return True


# ---------------------------------------------------------------------------
# 4. Filters: the student's explanation (rules + regex) and the model's hints
# ---------------------------------------------------------------------------
IDK = re.compile(
    r"^\W*("
    r"no\s*(lo|ho|le)?\s*s[eé]|nos[eé]|n\.?\s*s\.?|ns\s*nc|nidea|ni\s*idea|"
    r"no\s*(tinc|tengo)\s*(ni\s*)?(idea|ni\s*idea)|cap\s*idea|ninguna\s*idea|"
    r"i\s*(do\s*)?n'?o?t\s*know|i\s*dont\s*know|idk|dunno|no\s*idea|"
    r"porque\s*s[ií]|perqu[eè]\s*s[ií]|because|por\s*que\s*s[ií]|"
    r"nada|res|nothing|nose\s*nose|ok|vale|si|s[ií]|no|\?+|\.+"
    r")\W*$", re.I)
GUESS = re.compile(r"\b(al\s*azar|a\s*l'?atzar|he\s*adivinado|adivin\w*|endevin\w*|"
                   r"guess\w*|random|a\s*boleo|de\s*cap)\b", re.I)
REASON_WORDS = re.compile(
    r"\b(because|so|then|add\w*|subtract\w*|times|multipl\w*|divid\w*|count\w*|total|each|every|"
    r"perquè|per\s*tant|llavors|sum\w*|rest\w*|compt\w*|cada|vegades|"
    r"porque|entonces|luego|cont\w*|veces|mitad|meitat|doble|half|double)\b", re.I)
KEYBOARD_MASH = re.compile(r"(qwer|wert|asdf|sdfg|zxcv|hjkl|jkl[ñ;]|uiop|fghj|xcvb)", re.I)
VOWELS = "aeiouyàèéíòóúïüáâêôûœ"
# Common words in eng/cat/esp that may not appear in the M3Kang statements
BASE_WORDS = set("""
the and are was for that this with have has had then because there one two three four five six
seven eight nine ten each every all not but think thought did got get gives give make makes made
first last more less half double times plus minus equal equals result number numbers answer posts
que creo pensé pensaba porque entonces luego sumado restado multiplicado dividido sale salía
salió da dado tengo tiene hay son era eran cada todos uno dos tres cuatro cinco seis siete ocho
nueve diez más menos mitad doble veces igual resultado número números primero después también
crec pensava perquè llavors després surt sortia dona dóna tinc té són eren cada tots dues quatre
cinc sis set vuit nou deu més menys meitat vegades igual resultat nombre nombres primer també
per tant amb els les del dels una uns unes això aquest aquesta aquí perquè
los las del una unos unas con para por pero así esto este esta aquí
its into from than what when how which they them
""".split())


def _plausible_word(w):
    """A shape test for an invented word: vowels, no long consonant runs, no aaaa."""
    lw = w.lower()
    if lw.isdigit() or len(lw) <= 2:
        return True
    if not any(c in VOWELS for c in lw):
        return False
    if re.search(rf"[^{VOWELS}\d]{{5,}}", lw) or re.search(r"(.)\1\1", lw):
        return False
    return not KEYBOARD_MASH.search(lw)


def explanation_ok(text, vocab=None):
    """(ok, reason). Rules only, no model. Blocks empty text, 'I don't know' and its
    variations, guesses, keyboard mashing and invented words. vocab (optional) is a
    set of known words, e.g. Library.vocab built from the training statements."""
    t = (text or "").strip()
    if not t:
        return False, "empty"
    if IDK.match(t):
        return False, "dont_know"
    if GUESS.search(t):
        return False, "guess"          # honest and useful for the teacher, but no reasoning
    words = re.findall(r"[^\W\d_]+", t)
    if len(t) < 10 or len(words) + len(re.findall(r"\d+", t)) < 3:
        return False, "too_short"
    long_words = [w for w in words if len(w) > 2]
    if long_words:
        shape = sum(_plausible_word(w) for w in long_words) / len(long_words)
        if shape < 0.7:
            return False, "invented_words"
        if vocab:
            known = sum(w.lower() in vocab or w.lower() in BASE_WORDS or bool(REASON_WORDS.match(w))
                        for w in long_words) / len(long_words)
            if known < 0.3:
                return False, "invented_words"
    if not (re.search(r"\d", t) or REASON_WORDS.search(t)):
        return False, "no_reasoning"
    return True, "ok"


ANSWER_TALK = re.compile(r"\b(answer|correct|option|resposta|correcta|opció|respuesta|opción)\b"
                         r"|\b[A-E]\)|=\s*-?\d", re.I)


def leaks(text, problem):
    """True if text gives away the current problem's answer.

    Repeating a number from the statement is fine ("gaps of 5 m"), but naming
    the answer's value when it isn't a datum, an option letter, or talking
    about "the answer" is rejected.
    """
    if ANSWER_TALK.search(text):
        return True
    opts = problem.get("options") or {}
    value = str(opts.get(problem.get("label"), "")).strip()
    if not value:
        return False
    num = parse_num(value)
    if num is not None:
        data = statement_numbers(problem["text"], problem["language"])
        if any(close(num, d) for d in data):
            return False
        return any(close(parse_num(m) or math.inf, num) for m in _NUM.findall(_norm(text)))
    return len(value) >= 2 and value.lower() in text.lower()


# ---------------------------------------------------------------------------
# 5. The LLM, only where it's needed (short outputs, guarded)
# ---------------------------------------------------------------------------
APPLY_SYSTEM = ("A student solved a problem with the reasoning below. Apply EXACTLY the same "
                "reasoning to the new problem, even if the reasoning is wrong. "
                "Reply with the letter of the option it leads to.")
HINT_SYSTEM = ("You are a math tutor for school students. Give ONE short sentence that helps the "
               "student start the problem. Never give the answer, a final number, or a letter. "
               "Reply in {lang}.")
LANG_NAME = {"eng": "English", "cat": "Catalan", "esp": "Spanish"}


def llm_apply_reasoning(neighbour, explanation, timeout=40):
    response = post(URL, {
        "messages": [{"role": "system", "content": APPLY_SYSTEM},
                     {"role": "user", "content": f"Student's reasoning: {explanation.strip()}\n\n"
                                                 f"New problem:\n{format_problem(compact_problem(neighbour))}"
                                                 f"\nAnswer:"}],
        "grammar": "root ::= [A-E]", "max_tokens": 1, "temperature": 0, "seed": SEED,
        "cache_prompt": True,
    }, timeout=timeout)
    return response["choices"][0]["message"]["content"].strip()


def llm_hint(problem, timeout=40):
    """Rung 3. Returns the hint, or None if it leaks the answer or invents a number."""
    lang = LANG_NAME.get(problem["language"], "English")
    response = post(URL, {
        "messages": [{"role": "system", "content": HINT_SYSTEM.format(lang=lang)},
                     {"role": "user", "content": format_problem(compact_problem(problem))}],
        "max_tokens": 40, "temperature": 0, "seed": SEED, "cache_prompt": True,
    }, timeout=timeout)
    hint = response["choices"][0]["message"]["content"].strip().split("\n")[0]
    if leaks(hint, problem) or not grounded(hint, problem):
        return None
    return hint


# ---------------------------------------------------------------------------
# Hint ladder
# ---------------------------------------------------------------------------
def topic_of(problem):
    text = problem["text"].lower()
    for name, pattern in TOPICS:
        if re.search(pattern, text):
            return name
    return None


def easier_example(problem, hits, min_score=None, same_level=None):
    """Rung 1: the most similar solved problem at a lower level. At level 0, or with
    same_level (default SAME_LEVEL), a similar problem at the same level is accepted."""
    min_score = MIN_SCORE if min_score is None else min_score
    same_level = SAME_LEVEL if same_level is None else same_level
    good = [(s, p) for s, p in hits
            if s >= min_score and p.get("options") and safe_to_show(p, problem)]
    lower = [(s, p) for s, p in good if p.get("level", 0) < problem.get("level", 0)]
    same = [(s, p) for s, p in good if p.get("level", 0) == problem.get("level", 0)]
    pick = (lower or (same if problem.get("level", 0) == 0 or same_level else []) or [None])[0]
    return pick


def hint(problem, rung, index, library=None, use_llm=True):
    """Return {"rung", "kind", "message"} for help request number `rung` (1..4)."""
    lang = problem["language"]
    if rung <= 1:
        pick = easier_example(problem, neighbours(problem, index, library=library))
        if pick:
            p = library.localize(pick[1], lang) if library else pick[1]
            kind = "easier" if p.get("level", 0) < problem.get("level", 0) else "similar"
            return {"rung": 1, "kind": kind, "score": pick[0], "neighbour_id": p["id"],
                    "message": msg(lang, kind, problem=show(p), right=p["label"],
                                   right_val=p["options"].get(p["label"], ""))}
        rung = 2
    if rung == 2:
        topic = topic_of(problem)
        if topic is None:  # use what the neighbours have in common
            topics = [topic_of(p) for _s, p in neighbours(problem, index, k=5, library=library)]
            topics = [t for t in topics if t]
            topic = max(set(topics), key=topics.count) if topics else None
        if topic:
            return {"rung": 2, "kind": "topic", "topic": topic,
                    "message": msg(lang, "topic", topic=TOPIC_NAME[lang][topic],
                                   tip=TOPIC_TIP[lang][topic])}
        rung = 3
    if rung == 3 and use_llm:
        try:
            text = llm_hint(problem)
        except OSError:
            text = None
        if text:
            return {"rung": 3, "kind": "llm_hint", "message": text}
        return {"rung": 3, "kind": "llm_hint_rejected", "message": msg(lang, "no_more")}
    if rung >= 4:
        opts = problem.get("options") or {}
        return {"rung": 4, "kind": "solution",
                "message": msg(lang, "solution", right=problem["label"],
                               right_val=opts.get(problem["label"], ""))}
    return {"rung": rung, "kind": "none", "message": msg(lang, "no_more")}


# ---------------------------------------------------------------------------
# Wrong answer: the full flow
# ---------------------------------------------------------------------------
def on_wrong_answer(problem, chosen, explanation, index, library=None, use_llm=True, store=None,
                    student="anon"):
    start = time.perf_counter()
    lang = problem["language"]
    hits = neighbours(problem, index, library=library)
    parts, out = [], {"kind": "retry"}
    ok, why = explanation_ok(explanation, library.vocab if library else None)
    if explanation and explanation.strip():
        parts.append(msg(lang, "saved"))

    ce = counterexample(problem, chosen, hits)
    if ce:
        p = library.localize(ce["neighbour"], lang) if library else ce["neighbour"]
        parts.append(msg(lang, "counter", problem=show(p), wrong=ce["wrong"],
                         wrong_val=p["options"].get(ce["wrong"], ""), right=ce["right"],
                         right_val=p["options"].get(ce["right"], "")))
        out = {"kind": "counterexample", "error": ce["error"], "neighbour_id": p["id"],
               "score": ce["score"]}
    else:
        pick = easier_example(problem, hits) or next(
            ((s, p) for s, p in hits
             if s >= MIN_SCORE and p.get("options") and safe_to_show(p, problem)), None)
        if use_llm and ok and pick:
            try:
                letter = llm_apply_reasoning(pick[1], explanation)
            except OSError:
                letter = None
            p = library.localize(pick[1], lang) if library else pick[1]
            if letter and letter != p["label"] and letter in p["options"]:
                parts.append(msg(lang, "counter_llm", problem=show(p), wrong=letter,
                                 wrong_val=p["options"][letter], right=p["label"],
                                 right_val=p["options"][p["label"]]))
                out = {"kind": "counterexample_llm", "neighbour_id": p["id"], "score": pick[0]}
        if out["kind"] == "retry":
            h = hint(problem, 1, index, library, use_llm=False)
            if h["kind"] in ("easier", "similar"):
                parts.append(h["message"])
                out = {"kind": h["kind"], "neighbour_id": h["neighbour_id"], "score": h["score"]}
            else:
                parts.append(msg(lang, "retry"))

    out["message"] = "\n\n".join(parts)
    out["explanation_check"] = why
    out["seconds"] = round(time.perf_counter() - start, 3)
    if store:
        store.log(student, problem, chosen, explanation, out["kind"], why)
    return out


# ---------------------------------------------------------------------------
# Teacher log (SQLite on the board): every answer and explanation
# ---------------------------------------------------------------------------
class Store:
    SCHEMA = """CREATE TABLE IF NOT EXISTS answers (
        ts REAL, student TEXT, id TEXT, language TEXT, level INTEGER,
        chosen TEXT, correct TEXT, ok INTEGER, explanation TEXT, explanation_check TEXT,
        feedback TEXT)"""

    def __init__(self, path="data/tutor.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(self.SCHEMA)

    def log(self, student, problem, chosen, explanation, feedback, explanation_check=""):
        self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            time.time(), student, problem["id"], problem["language"], problem.get("level", 0),
            chosen, problem.get("label"), int(chosen == problem.get("label")),
            explanation or "", explanation_check, feedback))
        self.db.commit()
