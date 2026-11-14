"""Ask the small language model for one letter, A to E."""
import math
import re

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")

BOXED_GRAMMAR = 'root ::= "\\\\boxed{" [A-E] "}"'

TRAIN_BIAS_CAT_3B = {'A': -0.470, 'B': 0.313, 'C': 0.288, 'D': -0.672, 'E': 0.061}
TRAIN_BIAS_ENG_MATH = {'A': 0.376, 'B': 0.297, 'C': 0.113, 'D': -1.225, 'E': 0.004}

_PRIOR_CACHE = {}


def extract_letter_logprobs(response):
    letter_lps = {}
    if not isinstance(response, dict):
        return letter_lps
    choices = response.get("choices") or []
    content_list = []
    if choices and isinstance(choices[0], dict):
        logprobs = choices[0].get("logprobs") or {}
        content_list = logprobs.get("content") or []

    entries = []
    if content_list:
        tok_info = next(
            (t for t in content_list if isinstance(t, dict) and str(t.get("token") or "") in "ABCDE" and len(str(t.get("token") or "")) == 1),
            content_list[0] if isinstance(content_list[0], dict) else None,
        )
        if tok_info:
            top = tok_info.get("top_logprobs") or []
            entries = list(top) if top else [tok_info]
    elif response.get("completion_probabilities"):
        comp_probs = response.get("completion_probabilities") or []
        tok_info = next(
            (t for t in comp_probs if isinstance(t, dict) and str(t.get("token") or t.get("tok_str") or "") in "ABCDE" and len(str(t.get("token") or t.get("tok_str") or "")) == 1),
            comp_probs[0] if comp_probs and isinstance(comp_probs[0], dict) else None,
        )
        if tok_info:
            entries = tok_info.get("probs") or tok_info.get("top_logprobs") or []

    for item in entries:
        if not isinstance(item, dict):
            continue
        tok = str(item.get("token") or item.get("tok_str") or "")
        if tok in "ABCDE" and len(tok) == 1:
            if "logprob" in item and item["logprob"] is not None:
                lp = float(item["logprob"])
            elif "prob" in item and item["prob"] is not None:
                lp = math.log(max(float(item["prob"]), 1e-12))
            else:
                continue
            if tok not in letter_lps or lp > letter_lps[tok]:
                letter_lps[tok] = lp
    return letter_lps


def get_prior_logprobs(url=URL, timeout=10):
    if url in _PRIOR_CACHE:
        return _PRIOR_CACHE[url]
    neutral_problem = {"text": "Question?", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}
    try:
        resp = post(url, {
            "messages": build_messages(neutral_problem),
            "grammar": "root ::= [A-E]",
            "max_tokens": 1,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "logprobs": True,
            "top_logprobs": 5,
        }, timeout=timeout)
        lp = extract_letter_logprobs(resp)
        if lp:
            _PRIOR_CACHE[url] = lp
            return _PRIOR_CACHE[url]
    except Exception:
        pass
    return {}


def format_problem(problem, sees_image=False):
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append("(The options are in the picture.)" if sees_image else
                     "(The options are in a picture that you can't see. Pick the most likely letter.)")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it."""
    knowledge_texts = [ex["text"].strip() for ex in examples if not ex.get("options") and not ex.get("label") and ex.get("text")]
    problem_examples = [ex for ex in examples if ex.get("options") or ex.get("label")]

    messages = [{"role": "system", "content": SYSTEM}]
    for ex in problem_examples:  # solved problems retrieved with RAG, if any
        label = ex.get("label", "")
        options = ex.get("options") or {}
        if label in options and str(options[label]).strip():
            val = str(options[label]).strip()
            ans_text = f"The value is \\boxed{{{val}}}."
        elif label:
            ans_text = f"\\boxed{{{label}}}"
        else:
            ans_text = ""
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ans_text})
    rule_prefix = f"\n[Math rule: {' | '.join(knowledge_texts)}]\n" if knowledge_texts else ""
    text = rule_prefix + format_problem(problem, sees_image=bool(image_b64)) + "\nAnswer:"
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                   {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL, calibrate=True, debias_alpha=None):
    """Return (letter, timings). The grammar makes any other output impossible."""
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64),
        "grammar": BOXED_GRAMMAR,
        "max_tokens": 5,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": 20,
    }, timeout=timeout)
    timings = dict(response.get("timings") or {})
    choices = response.get("choices") or []
    content = choices[0]["message"]["content"] if choices else ""
    m_box = re.search(r"\\boxed\{\s*([A-E])\s*\}", content)
    gen_letter = m_box.group(1) if m_box else (re.findall(r"[A-E]", content)[-1] if re.findall(r"[A-E]", content) else "D")
    raw_lp = extract_letter_logprobs(response)
    if debias_alpha is None:
        debias_alpha = float(CONFIG.get("DEBIAS_ALPHA", 0.0))

    if debias_alpha > 0.0 and raw_lp:
        bias = TRAIN_BIAS_ENG_MATH if (problem.get("language") == "eng" or "math" in url or "8083" in url) else TRAIN_BIAS_CAT_3B
        scores = {c: raw_lp.get(c, -15.0) - debias_alpha * bias.get(c, 0.0) for c in "ABCDE"}
        m = max(scores.values())
        exps = {c: math.exp(scores[c] - m) for c in "ABCDE"}
        total = sum(exps.values())
        probs = {c: exps[c] / total for c in "ABCDE"} if total > 0 else {c: 0.2 for c in "ABCDE"}
        letter = max("ABCDE", key=scores.get)
        timings["confidence"] = round(probs.get(letter, probs[max("ABCDE", key=probs.get)]), 4)
    elif raw_lp:
        prior = get_prior_logprobs(url=url, timeout=min(timeout, 10)) if calibrate else {}
        scores = {c: raw_lp.get(c, -10.0) - prior.get(c, 0.0) for c in "ABCDE"}
        m = max(scores.values())
        exps = {c: math.exp(scores[c] - m) for c in "ABCDE"}
        total = sum(exps.values())
        probs = {c: exps[c] / total for c in "ABCDE"} if total > 0 else {c: 0.2 for c in "ABCDE"}
        letter = gen_letter if gen_letter in "ABCDE" else max("ABCDE", key=probs.get)
        timings["confidence"] = round(probs.get(letter, probs[max("ABCDE", key=probs.get)]), 4)
    else:
        letter = gen_letter
        timings["confidence"] = 1.0
    return letter, timings


def answer_jev(problem, examples=(), timeout=60, image_b64=None, url=URL, debias_alpha=None):
    letter, timings = answer_letter(problem, examples=examples, timeout=timeout,
                                    image_b64=image_b64, url=url, calibrate=True, debias_alpha=debias_alpha)
    return letter, timings.get("confidence", 1.0), timings


def match_option_from_text(text, options=None, allow_numeric=True):
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{\s*(?:\\text\{\s*)?([A-E])\s*\}?\s*\}", text)
    if boxed:
        return boxed[-1]
    explicit_matches = list(re.finditer(r"(?i)(?:answer|option|lletra|resposta)\s*(?:is\b|[:=])?\s*\(?([A-E])\)?\b", text))
    for m in reversed(explicit_matches):
        candidate = m.group(1).upper()
        post = text[m.end():]
        if m.group(1) == 'a' and re.match(r"^\s+[a-zA-Z]", post):
            continue
        return candidate

    if not allow_numeric:
        return None

    m_end = re.search(r"(?:^|[\s(])([A-E])\)?\.?\s*$", text)
    if m_end:
        return m_end.group(1)

    if options:
        nums = re.findall(r"-?\d+(?:/\d+|\.\d+)?", text)
        if nums:
            last_num = nums[-1]
            matched = [k for k, v in options.items() if str(v).strip() == last_num]
            if len(matched) == 1:
                return matched[0]
    return None


def answer_micro_cot(problem, examples=(), timeout=60, image_b64=None, url=URL, temperature=0, seed=None):
    messages = build_messages(problem, examples, image_b64)
    suffix = "\nSolve in at most 3 short lines of equations, then write \\boxed{X} where X is the correct option letter (A, B, C, D, or E)."
    last_content = messages[-1]["content"]
    if isinstance(last_content, str):
        messages[-1]["content"] = last_content.removesuffix("\nAnswer:") + suffix
    elif isinstance(last_content, list):
        for part in reversed(last_content):
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = part.get("text", "").removesuffix("\nAnswer:") + suffix
                break
        else:
            last_content.append({"type": "text", "text": suffix})
    actual_seed = int(CONFIG["SEED"]) if seed is None else int(seed)
    response = post(url, {
        "messages": messages,
        "max_tokens": int(CONFIG.get("COT_MAX_TOKENS", 220)),
        "temperature": temperature,
        "seed": actual_seed,
        "cache_prompt": True,
    }, timeout=timeout)
    choices = response.get("choices") or []
    text = choices[0]["message"]["content"].strip() if choices else ""
    finish_reason = choices[0].get("finish_reason") if choices else None
    matched = match_option_from_text(text, problem.get("options") or {}, allow_numeric=(finish_reason != "length"))
    timings = dict(response.get("timings") or {})
    if matched is None and text:
        step2_messages = list(messages) + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "State the final option letter in \\boxed{X}:"},
        ]
        try:
            resp2 = post(url, {
                "messages": step2_messages,
                "grammar": BOXED_GRAMMAR,
                "max_tokens": 5,
                "temperature": 0,
                "seed": int(CONFIG["SEED"]),
                "cache_prompt": True,
                "logprobs": True,
                "top_logprobs": 20,
            }, timeout=min(timeout, 10))
            raw_lp2 = extract_letter_logprobs(resp2)
            if raw_lp2:
                matched = max(raw_lp2, key=raw_lp2.get)
            else:
                choices2 = resp2.get("choices") or []
                c2 = choices2[0]["message"]["content"] if choices2 else ""
                letters2 = re.findall(r"[A-E]", c2)
                if letters2:
                    matched = letters2[-1]
            timings2 = resp2.get("timings") or {}
            if "prompt_n" in timings2:
                timings["prompt_n"] = timings.get("prompt_n", 0) + timings2["prompt_n"]
        except Exception:
            pass
    return matched, timings


def answer_elimination(problem, examples=(), timeout=60, image_b64=None, url=URL):
    messages = build_messages(problem, examples, image_b64)
    suffix = ("\nStep 1: Evaluate each option and eliminate those that are mathematically impossible.\n"
              "Step 2: Choose the only correct option.\n"
              "Final answer: \\boxed{LETTER}")
    last_content = messages[-1]["content"]
    if isinstance(last_content, str):
        messages[-1]["content"] = last_content.removesuffix("\nAnswer:") + suffix
    elif isinstance(last_content, list):
        for part in reversed(last_content):
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = part.get("text", "").removesuffix("\nAnswer:") + suffix
                break
        else:
            last_content.append({"type": "text", "text": suffix})

    response = post(url, {
        "messages": messages,
        "max_tokens": 180,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)
    choices = response.get("choices") or []
    text = choices[0]["message"]["content"].strip() if choices else ""
    finish_reason = choices[0].get("finish_reason") if choices else None
    timings = dict(response.get("timings") or {})

    matched = match_option_from_text(text, allow_numeric=False)
    has_boxed = bool(re.search(r"\\boxed\{\s*(?:\\text\{\s*)?([A-E])\s*\}?\s*\}", text))
    if finish_reason == "length" or not has_boxed:
        if text:
            step2_messages = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "State the final letter in \\boxed{X}:"},
            ]
            try:
                resp2 = post(url, {
                    "messages": step2_messages,
                    "grammar": BOXED_GRAMMAR,
                    "max_tokens": 5,
                    "temperature": 0,
                    "seed": int(CONFIG["SEED"]),
                    "cache_prompt": True,
                    "logprobs": True,
                    "top_logprobs": 20,
                }, timeout=min(timeout, 10))
                raw_lp2 = extract_letter_logprobs(resp2)
                if raw_lp2:
                    matched = max(raw_lp2, key=raw_lp2.get)
                else:
                    choices2 = resp2.get("choices") or []
                    c2 = choices2[0]["message"]["content"] if choices2 else ""
                    letters2 = re.findall(r"[A-E]", c2)
                    if letters2:
                        matched = letters2[-1]
                timings2 = resp2.get("timings") or {}
                if "prompt_n" in timings2:
                    timings["prompt_n"] = timings.get("prompt_n", 0) + timings2["prompt_n"]
            except Exception:
                pass
    return matched, timings


def verify_answer(problem, candidate_letter, examples=(), timeout=15, image_b64=None, url=URL):
    """Quick 5-token verification of candidate_letter using BOXED_GRAMMAR and KV-cache."""
    messages = build_messages(problem, examples, image_b64)
    messages.extend([
        {"role": "assistant", "content": f"\\boxed{{{candidate_letter}}}"},
        {"role": "user", "content": f"Carefully re-verify option {candidate_letter}. State the final correct option letter in \\boxed{{X}}:"}
    ])
    response = post(url, {
        "messages": messages,
        "grammar": BOXED_GRAMMAR,
        "max_tokens": 5,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": 20,
    }, timeout=timeout)
    timings = dict(response.get("timings") or {})
    raw_lp = extract_letter_logprobs(response)
    if raw_lp:
        letter = max(raw_lp, key=raw_lp.get)
    else:
        choices = response.get("choices") or []
        content = choices[0]["message"]["content"] if choices else ""
        letters = re.findall(r"[A-E]", content)
        letter = letters[-1] if letters else candidate_letter
    return letter, timings

