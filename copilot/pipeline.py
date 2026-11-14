"""The baseline copilot: one problem in, one letter out.

This is the file to change first. predict.py and eval.py both call solve(),
so whatever you build here is exactly what the official run measures.
"""
from collections import Counter
import socket
import time
import urllib.error

from .config import CONFIG
from .llm import URL, answer_elimination, answer_letter, answer_micro_cot, verify_answer

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]
MARGIN = 3.0  # seconds kept free for Python and for writing the answer


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT, mode=None):
    """Return {"answer", "seconds", "status", ...} for one problem.

    index and k turn on retrieval: the k most similar solved problems are
    added to the prompt as examples. Each example costs reading time.
    image_b64 and llm_url send the problem image to a vision server.
    """
    start = time.perf_counter()
    result = {"answer": FALLBACK, "status": "ok", "prompt_tokens": 0, "retrieved": "", "examples": []}
    mode = mode or CONFIG.get("SOLVE_MODE", "hybrid")
    try:
        if mode == "router":
            math_url = CONFIG.get("ROUTER_MATH_URL", "http://127.0.0.1:8083/v1/chat/completions")
            target_url = math_url if problem.get("language") == "eng" else llm_url
            return solve(problem, index=index, k=k, exclude_id=exclude_id, image_b64=image_b64,
                         llm_url=target_url, time_limit=time_limit, mode="hybrid")

        if "options" in problem and not problem.get("options") and not image_b64:
            if index is not None and k > 0:
                hits = index.search(problem, k=1, exclude_id=exclude_id, min_score=0.98)
                if hits and hits[0][0] >= 0.98 and hits[0][1].get("label") in "ABCDE":
                    result["answer"] = hits[0][1]["label"]
            result["seconds"] = round(time.perf_counter() - start, 2)
            return result

        examples = []
        hits = []
        if index is not None and k > 0:
            hits = index.search(problem, k=k, exclude_id=exclude_id,
                                min_score=float(CONFIG.get("RAG_MIN_SCORE", 0.55)))
            examples = [p for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)

        remaining = time_limit - (time.perf_counter() - start) - MARGIN
        use_prior = CONFIG.get("JEV_USE_NEUTRAL_PRIOR") == "1"

        if mode == "tier2":
            cot_letter, timings = answer_micro_cot(problem, examples, timeout=max(remaining, 1.0),
                                                   image_b64=image_b64, url=llm_url)
            result["prompt_tokens"] = timings.get("prompt_n", 0)
            if cot_letter in ("A", "B", "C", "D", "E"):
                result["answer"] = cot_letter
            else:
                remaining = time_limit - (time.perf_counter() - start) - MARGIN
                letter, _t2 = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url, calibrate=False)
                result["answer"] = letter
        elif mode == "hybrid":
            letter, timings = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url, calibrate=use_prior)
            result["answer"] = letter
            result["prompt_tokens"] = timings.get("prompt_n", 0)
            confidence = timings.get("confidence", 1.0)
            remaining = time_limit - (time.perf_counter() - start) - MARGIN
            if confidence < float(CONFIG.get("JEV_CONFIDENCE_THRESHOLD", 0.38)) and remaining > 25.0:
                cot_letter, _t2 = answer_micro_cot(problem, examples, timeout=remaining,
                                                   image_b64=image_b64, url=llm_url)
                if cot_letter in ("A", "B", "C", "D", "E"):
                    result["answer"] = cot_letter
        elif mode == "consistency":
            letters = []
            total_prompt_tokens = 0
            for i in range(3):
                remaining = time_limit - (time.perf_counter() - start) - MARGIN
                cot_letter, timings = answer_micro_cot(
                    problem, examples, timeout=max(remaining / 3, 1.0),
                    image_b64=image_b64, url=llm_url,
                    temperature=0.6, seed=int(CONFIG["SEED"]) + i,
                )
                total_prompt_tokens += timings.get("prompt_n", 0)
                if cot_letter in ("A", "B", "C", "D", "E"):
                    letters.append(cot_letter)
            result["prompt_tokens"] = total_prompt_tokens
            if letters:
                result["answer"] = Counter(letters).most_common(1)[0][0]
            else:
                remaining = time_limit - (time.perf_counter() - start) - MARGIN
                fallback_letter, _t = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url, calibrate=False,
                )
                result["answer"] = fallback_letter if fallback_letter in "ABCDE" else FALLBACK
        elif mode == "elimination":
            letter, timings = answer_elimination(problem, examples, timeout=max(remaining, 1.0),
                                                 image_b64=image_b64, url=llm_url)
            result["prompt_tokens"] = timings.get("prompt_n", 0)
            if letter in ("A", "B", "C", "D", "E"):
                result["answer"] = letter
            else:
                remaining = time_limit - (time.perf_counter() - start) - MARGIN
                fb_letter, _t = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                              image_b64=image_b64, url=llm_url, calibrate=False)
                result["answer"] = fb_letter if fb_letter in "ABCDE" else FALLBACK
        else:
            letter, timings = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url,
                                            calibrate=(use_prior and mode != "baseline"))
            result["answer"] = letter
            result["prompt_tokens"] = timings.get("prompt_n", 0)

    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result
