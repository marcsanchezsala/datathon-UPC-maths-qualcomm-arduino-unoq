"""Ask the small language model for one letter, A to E."""
import re

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Reasoning mode: the model writes a few lines of working, then "Answer: X".
# Off by default. finetune/ trains the model on exactly this prompt format.
REASONING = CONFIG.get("REASONING", "0") == "1"
READ_SPEED = float(CONFIG.get("READ_TOKENS_PER_SECOND", "11"))
WRITE_SPEED = float(CONFIG.get("WRITE_TOKENS_PER_SECOND", "5.5"))
MIN_REASON_TOKENS = 16  # below this, reasoning can't help: answer one letter instead

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")
SYSTEM_REASON = ("You solve multiple-choice math competition problems for school students. "
                 "Work out the answer in a few short lines, then write the letter of the "
                 "correct option as \"Answer: X\".")


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


def build_messages(problem, examples=(), image_b64=None, reason=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it.

    reason picks the reasoning prompt; by default, REASONING for text-only problems.
    """
    if reason is None:
        reason = REASONING and not image_b64
    system, cue, reply = (SYSTEM_REASON, "", "Answer: {}") if reason else (SYSTEM, "\nAnswer:", "{}")
    messages = [{"role": "system", "content": system}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + cue})
        messages.append({"role": "assistant", "content": reply.format(ex["label"])})
    text = format_problem(problem, sees_image=bool(image_b64)) + cue
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
        "messages": build_messages(problem, examples, image_b64, reason=False),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)
    letter = response["choices"][0]["message"]["content"].strip()
    return letter, response.get("timings", {})


def reason_budget(prompt_tokens, time_limit):
    """How many tokens the model may write and still answer in time on the board.

    Counted in tokens, not seconds, so a fast laptop measures what the board will do.
    Reserves the read of a second short call ("Answer:" and one letter) at the end.
    """
    seconds = time_limit - prompt_tokens / READ_SPEED - 8 / READ_SPEED - 2 / WRITE_SPEED
    return max(0, int(seconds * WRITE_SPEED))


def find_letter(text):
    """The last "Answer: X" in the text, or None."""
    found = re.findall(r"Answer\s*:?\s*\(?([A-E])\b", text)
    return found[-1] if found else None


def answer_reasoning(problem, examples=(), timeout=60, time_limit=57, url=URL):
    """Return (letter, info): the model reasons within a token budget, then answers.

    Uses the raw /completion endpoint, so a second call can continue the same text
    and force the letter with the grammar when the reasoning didn't reach "Answer:".
    """
    root = url.split("/v1/")[0]
    prompt = post(root + "/apply-template", {"messages": build_messages(problem, examples, reason=True)},
                  timeout=timeout)["prompt"]
    prompt_tokens = len(post(root + "/tokenize", {"content": prompt}, timeout=timeout)["tokens"])
    budget = reason_budget(prompt_tokens, time_limit)
    info = {"prompt_n": prompt_tokens, "budget": budget, "reason_tokens": 0, "calls": 0, "reasoning": ""}
    if budget < MIN_REASON_TOKENS:
        letter, _timings = answer_letter(problem, examples, timeout=timeout, url=url)
        return letter, {**info, "calls": 1}
    common = {"temperature": 0, "seed": int(CONFIG["SEED"]), "cache_prompt": True}
    first = post(root + "/completion", {"prompt": prompt, "n_predict": budget, **common}, timeout=timeout)
    text = first["content"]
    info.update(reason_tokens=first["timings"]["predicted_n"], calls=1, reasoning=text)
    letter = find_letter(text)
    if letter:
        return letter, info
    # No "Answer: X" in the text: finish it with a grammar that only allows a letter
    second = post(root + "/completion", {"prompt": prompt + text + "\nAnswer:", "n_predict": 2,
                                         "grammar": 'root ::= " "? [A-E]', **common}, timeout=timeout)
    info.update(calls=2, second_prompt_n=second["timings"]["prompt_n"])
    return second["content"].strip(), info


if __name__ == "__main__":  # python3 -m copilot.llm: check the reasoning helpers
    assert find_letter("4 * 3 = 12\nAnswer: C") == "C"
    assert find_letter("Answer: (B)") == "B" and find_letter("Answer:E") == "E"
    assert find_letter("Answer: A, no wait.\nAnswer: D") == "D"
    assert find_letter("A square has four sides.") is None
    # 110 prompt tokens on the board: 10 s to read, about 46 s to write, 1 s for the letter
    assert reason_budget(110, 57) == 252 and reason_budget(700, 57) == 0
    print("ok")
