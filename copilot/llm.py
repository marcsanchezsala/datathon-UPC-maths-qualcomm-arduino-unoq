"""Ask the small language model for one letter, A to E."""
from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")


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
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    text = format_problem(problem, sees_image=bool(image_b64)) + "\nAnswer:"
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                   {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL):
    """Return (letter, timings). The grammar makes any other output impossible."""
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)
    letter = response["choices"][0]["message"]["content"].strip()
    return letter, response.get("timings", {})


def answer_micro_cot(problem, examples=(), timeout=35, image_b64=None, url=None, temperature=0, seed=None):
    """Solve using Micro-CoT: 2-3 lines of math equations/logic concluding with \\boxed{LETTER}."""
    timeout = min(timeout, 35)
    target_url = url or CONFIG.get("ROUTER_MATH_URL") or URL
    actual_seed = int(CONFIG["SEED"]) if seed is None else int(seed)
    max_tokens = int(CONFIG.get("COT_MAX_TOKENS", 120))
    messages = build_messages(problem, examples, image_b64)
    suffix = "\nSolve in at most 3 short lines of equations and logic, then write \\boxed{X} where X is the correct option letter (A, B, C, D, or E)."
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

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": actual_seed,
        "cache_prompt": True,
    }

    try:
        response = post(target_url, payload, timeout=timeout)
    except Exception:
        # Fallback to default LLM port (8081) if router or target URL fails
        if target_url != URL:
            response = post(URL, payload, timeout=timeout)
        else:
            raise

    choices = response.get("choices") or []
    text = choices[0]["message"]["content"].strip() if choices else ""
    timings = dict(response.get("timings") or {})
    return text, timings

