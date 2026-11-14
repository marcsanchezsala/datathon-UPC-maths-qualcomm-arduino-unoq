"""Teaching features for the M3Kang explorer.

Separate from copilot/pipeline.py so the reasoning team can optimise predict.py
without touching teaching code. Uses the same LLM server but different prompts.
"""
from .client import post
from .config import CONFIG

TUTOR_URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

TUTOR_SYSTEM = (
    "You are a patient math tutor. The student is solving a multiple-choice math "
    "competition problem. Guide them without giving the answer directly. "
    "Explain the method, point out what to look for, or give a similar example. "
    "Keep responses under 80 words and use clear, simple language."
)

SOLUTION_SYSTEM = (
    "The student needs the full solution now. Explain the reasoning step by step, "
    "show the calculation or logic, and state the correct letter clearly."
)


def _build_problem_text(problem):
    """Format a problem into plain text, same style as the explorer."""
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for l in "ABCDE":
        if l in options:
            lines.append(f"{l}) {options[l]}")
    return "\n".join(lines)


def tutor_messages(problem, hint_level=3):
    """Return the messages for a tutor request."""
    problem_text = _build_problem_text(problem)

    if hint_level == 4:
        system_content = SOLUTION_SYSTEM
        user_content = (
            f"The student is ready for the solution. Explain step by step and give the correct letter.\n"
            f"{problem_text}"
        )
    else:
        system_content = TUTOR_SYSTEM
        user_content = (
            f"The student is stuck on this problem. Give a helpful hint or explanation of the method. "
            f"Do NOT give the answer directly. Focus on guiding their thinking.\n"
            f"{problem_text}"
        )

    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


def find_rag_examples(problem, index, k=3):
    """Find similar problems at a lower level for RAG-based hints."""
    if not index:
        return []
    hits = index.search(problem, k=k * 2, exclude_id=problem["id"])
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


def tutor_generate(problem, index=None, hint_level=3, timeout=45):
    """Ask the LLM for a tutor response.
    
    hint_level: 1=similar problem (RAG), 2=method hint, 3=model hint, 4=solution
    """
    messages = tutor_messages(problem, hint_level)

    # Add RAG examples for level 1 hints
    if hint_level == 1 and index:
        examples = find_rag_examples(problem, index, k=2)
        if examples:
            rag_text = "Here is a similar problem that was already solved. Think about how you would solve it:\n"
            for _score, p in examples:
                rag_text += _build_problem_text(p) + "\n"
            messages.append({"role": "assistant", "content": rag_text})
            messages.append({"role": "user", "content": f"Now try this one: {problem['text'].strip()}. Give a hint."})

    payload = {
        "messages": messages,
        "max_tokens": 250,
        "temperature": 0.5 if hint_level <= 3 else 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": False,
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
        return {"text": f"Error generating hint: {e}", "type": "error", "level": hint_level}