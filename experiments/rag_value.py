#!/usr/bin/env python3
"""Experiment: does a retrieved M3Kang problem help if it shows the VALUE of its answer?

M3Kang has no worked solutions, so a retrieved example used to end in a bare
letter ("Answer: C"), which says nothing about a different problem. Here the
example also shows the value of its correct option ("18"), and some
conditions ask the model for the value of the option instead of its letter.

    LLM_PORT=8091 EMBED_PORT=8092 python3 experiments/rag_value.py \
        --model 0.8b --input data/demo/train.jsonl --conditions all

Conditions:
    letter-k0         one letter, grammar A-E (the baseline)
    letter-k1         + the nearest solved problem, its answer as a letter (the old RAG)
    letter-k1v        + the nearest solved problem, its answer as "C (18)"
    value-k0          the model writes one option's text, forced by a grammar
    value-k1v         + the nearest solved problem, its answer as a value
    reason-k0         a few lines of working within the board's token budget, then the letter
    reason-k1         + the nearest solved problem, "Answer: C"
    reason-k1v        + the nearest solved problem, "Answer: C (18)"

No board here: board_s is an estimate, full prompt tokens / READ + written
tokens / WRITE, with the speeds in config.env. The reasoning budget is counted
in tokens with the same speeds, so it is what the board would allow.
Writes results/rag-exp/<model>_<condition>_<input>.csv and one line per run
in results/rag-exp/summary.csv. Standard library only, like the copilot.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from copilot import llm  # noqa: E402  (ports come from the environment)
from copilot.client import post  # noqa: E402
from copilot.config import CONFIG  # noqa: E402

CONDITIONS = ["letter-k0", "letter-k1", "letter-k1v", "value-k0", "value-k1v",
              "reason-k0", "reason-k1", "reason-k1v", "reason-ctxv"]
# reason-ctxv: the solved problem goes inside the user message, as a reference,
# not as an earlier assistant turn. As a turn, its bare "Answer: C (18)" teaches
# the model to answer at once without working.
REFERENCE = "A similar solved problem, for reference:\n{example}\nIts answer: {letter}) {value}\n\nNow solve this problem:\n"
SYSTEM_VALUE = ("You solve multiple-choice math competition problems for school students. "
                "Read the problem and the options, then reply with the correct option, "
                "exactly as it is written.")
OUT = os.path.join(ROOT, "results", "rag-exp")
ROOT_URL = llm.URL.split("/v1/")[0]
COMMON = {"temperature": 0, "seed": int(CONFIG["SEED"]), "cache_prompt": True}


def with_value(example):
    """The example with its answer shown as "C (18)" instead of "C"."""
    value = example["options"][example["label"]]
    return {**example, "label": f"{example['label']} ({value})"}


def gbnf_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def prompt_tokens(messages):
    prompt = post(ROOT_URL + "/apply-template", {"messages": messages}, timeout=60)["prompt"]
    return len(post(ROOT_URL + "/tokenize", {"content": prompt}, timeout=60)["tokens"])


def value_messages(problem, examples):
    messages = [{"role": "system", "content": SYSTEM_VALUE}]
    for ex in examples:
        messages.append({"role": "user", "content": llm.format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["options"][ex["label"]]})
    messages.append({"role": "user", "content": llm.format_problem(problem) + "\nAnswer:"})
    return messages


def answer_value(problem, examples):
    """The model writes one of the five option texts; the grammar allows nothing else."""
    options = problem["options"]
    texts = {letter: options[letter].strip() for letter in "ABCDE"}
    if len(set(texts.values())) < 5:  # two identical options: the text can't name one letter
        letter, _ = llm.answer_letter(problem, (), url=llm.URL)
        return letter, {"written": 1, "note": "duplicate options, letter"}
    messages = value_messages(problem, examples)
    grammar = "root ::= " + " | ".join(gbnf_string(t) for t in texts.values())
    response = post(llm.URL, {"messages": messages, "grammar": grammar, "max_tokens": 48, **COMMON}, timeout=120)
    written = response["choices"][0]["message"]["content"].strip()
    letter = next((l for l, t in texts.items() if t == written), None)
    info = {"written": response.get("timings", {}).get("predicted_n", 0), "note": written}
    return letter or CONFIG["FALLBACK_ANSWER"], info


def run_reference(problem, examples, retrieved):
    """Reasoning, with the solved problem quoted in the user message."""
    original = llm.build_messages

    def build(problem_, examples_=(), image_b64=None, reason=None):
        messages = original(problem_, (), image_b64, reason)
        if examples:
            ex = examples[0]
            note = REFERENCE.format(example=llm.format_problem(ex), letter=ex["label"],
                                    value=ex["options"][ex["label"]])
            messages[-1] = {**messages[-1], "content": note + messages[-1]["content"]}
        return messages

    llm.build_messages = build  # answer_reasoning and its letter fallback both build through it
    try:
        letter, info = llm.answer_reasoning(problem, (), timeout=300, time_limit=57, url=llm.URL)
    finally:
        llm.build_messages = original
    n_prompt = info["prompt_n"] + (8 if info.get("calls") == 2 else 0)
    written = info.get("reason_tokens", 0) + (1 if info.get("calls") == 2 else 0)
    if letter not in ("A", "B", "C", "D", "E"):
        letter = CONFIG["FALLBACK_ANSWER"]
    return letter, {"prompt_tokens": n_prompt, "written": written,
                    "board_s": round(n_prompt / llm.READ_SPEED + written / llm.WRITE_SPEED, 1),
                    "retrieved": retrieved, "budget": info.get("budget", ""), "calls": info.get("calls", ""),
                    "note": "", "reasoning": info.get("reasoning", "")}


def run_one(problem, condition, index):
    examples, retrieved = [], ""
    if condition.split("-")[1] != "k0" and index is not None:
        hits = index.search(problem, k=1, exclude_id=problem["id"])
        examples = [p for _score, p in hits]
        retrieved = ";".join(f"{p['id']}|{p['language']}:{s:.3f}" for s, p in hits)
    if condition.endswith("v"):
        shown = [with_value(ex) for ex in examples]
    else:
        shown = examples
    mode = condition.split("-")[0]
    info = {}
    if condition == "reason-ctxv":
        return run_reference(problem, examples, retrieved)
    if mode == "value" and problem.get("options"):
        letter, info = answer_value(problem, examples)
        n_prompt = prompt_tokens(value_messages(problem, examples))
        written = info["written"]
    elif mode == "reason" and problem.get("options") is not None:
        letter, info = llm.answer_reasoning(problem, shown, timeout=300, time_limit=57, url=llm.URL)
        n_prompt = info["prompt_n"]
        written = info.get("reason_tokens", 0) + (1 if info.get("calls") == 2 else 0)
        if info.get("calls") == 2:
            n_prompt += 8  # the short second call re-reads "Answer:" on the board
    else:  # letter mode, or a value condition on a problem with no options
        letter, _ = llm.answer_letter(problem, shown, timeout=120, url=llm.URL)
        n_prompt = prompt_tokens(llm.build_messages(problem, shown, reason=False))
        written = 1
    if letter not in ("A", "B", "C", "D", "E"):
        letter = CONFIG["FALLBACK_ANSWER"]
    board_s = n_prompt / llm.READ_SPEED + written / llm.WRITE_SPEED
    return letter, {"prompt_tokens": n_prompt, "written": written, "board_s": round(board_s, 1),
                    "retrieved": retrieved, "budget": info.get("budget", ""),
                    "calls": info.get("calls", ""), "note": info.get("note", ""),
                    "reasoning": info.get("reasoning", "")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="a label for the model the server runs")
    p.add_argument("--input", required=True)
    p.add_argument("--index", default="data/index-v2.db")
    p.add_argument("--conditions", default="all")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    conditions = CONDITIONS if args.conditions == "all" else args.conditions.split(",")
    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        problems = problems[:args.limit]
    index = None
    if any(not c.endswith("k0") for c in conditions):
        from copilot.rag import Index
        index = Index(args.index)

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    stem = os.path.basename(os.path.dirname(args.input)) + "-" + stem if stem == "train" else stem
    for condition in conditions:
        path = os.path.join(OUT, f"{args.model}_{condition}_{stem}.csv")
        rows, start = [], time.time()
        with open(path, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            fields = ["prompt_tokens", "written", "board_s", "retrieved", "budget", "calls", "note", "reasoning"]
            writer.writerow(["id", "language", "level", "text_only", "label", "answer", "correct"] + fields)
            for problem in problems:
                letter, info = run_one(problem, condition, index)
                correct = letter == problem["label"]
                rows.append((problem, letter, correct, info))
                writer.writerow([problem["id"], problem["language"], problem["level"], problem.get("text_only"),
                                 problem["label"], letter, int(correct)] + [info[k] for k in fields])
                out.flush()
        n = len(rows)
        hits = sum(r[2] for r in rows)
        board = [r[3]["board_s"] for r in rows]
        letters = Counter(r[1] for r in rows)
        line = {"time": time.strftime("%Y%m%d-%H%M%S"), "model": args.model, "condition": condition,
                "input": args.input, "n": n, "correct": hits, "accuracy": f"{100 * hits / n:.1f}",
                "mean_prompt_tokens": f"{sum(r[3]['prompt_tokens'] for r in rows) / n:.0f}",
                "mean_board_s": f"{sum(board) / n:.1f}", "max_board_s": f"{max(board):.1f}",
                "over_57s": sum(b > 57 for b in board),
                "letters": " ".join(f"{l}:{letters[l]}" for l in "ABCDE"),
                "laptop_minutes": f"{(time.time() - start) / 60:.1f}"}
        summary = os.path.join(OUT, "summary.csv")
        new = not os.path.exists(summary)
        with open(summary, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(line))
            if new:
                writer.writeheader()
            writer.writerow(line)
        print(f"{args.model} {condition} {stem}: {hits}/{n} = {line['accuracy']}%, "
              f"board ~{line['mean_board_s']} s mean, {line['max_board_s']} s max, letters {line['letters']}",
              flush=True)


if __name__ == "__main__":
    main()
