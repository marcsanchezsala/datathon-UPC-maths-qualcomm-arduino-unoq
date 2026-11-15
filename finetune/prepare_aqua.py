#!/usr/bin/env python3
"""Build the fine-tuning set from AQuA-RAT: short worked solutions ending in "Answer: X".

    python prepare_aqua.py --out work/aqua.jsonl --n 15000

Only AQuA-RAT (deepmind/aqua_rat, Apache 2.0) goes in: never M3Kang.
Each line holds the problem, its new label and the chat messages, built with
copilot.llm.build_messages, so the model trains on the prompt it sees at inference.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from copilot.config import CONFIG  # noqa: E402
from copilot.llm import build_messages  # noqa: E402

LETTERS = "ABCDE"
MAX_SOLUTION_TOKENS = 150
MAX_TOTAL_TOKENS = 512  # the training length: longer examples would be cut
# The last line must only name the answer: "Answer: B", "ANSWER:C", "The answer is B.", "B",
# "Correct Answer:A)10", "Option D"
ANSWER_LINE = re.compile(r"^\W*(?:(?:the\s+)?(?:correct\s+)?(?:answer|ans|option|choice)\w*"
                         r"(?:\s+(?:is|will be|should be))?\s*[:.\-=]*\s*)?\(?([A-E])\b\)?[^A-Za-z]{0,15}$", re.I)
# A calculation: digits around an operator or an equals sign
CALCULATION = re.compile(r"\d\s*[-+*/×÷^=]\s*[(\d.]|=\s*-?\d")
# The working mustn't point at a letter: the options get shuffled
LETTER_REFERENCE = re.compile(r"\b(?:answers?|options?|choices?|ans)\b\W*[A-E]\b|\([A-E]\)|\b[A-E]\)", re.I)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
HEADER = re.compile(r"^(?:sol(?:ution)?|explanation|ans(?:wer)?)\W*$", re.I)  # "Sol.", "Explanation:"


def squash(text):
    return re.sub(r"\s+", "", text).replace(",", "").lower()


def parse_options(raw):
    """["A)21", ...] -> {"A": "21", ...}, or None if they aren't five clean options."""
    options = {}
    for letter, item in zip(LETTERS, raw):
        if not item.startswith(letter + ")") or not item[2:].strip():
            return None
        options[letter] = item[2:].strip()
    if len(raw) != 5 or len(set(map(squash, options.values()))) != 5:
        return None
    return options


def mentions(body, value):
    """Does the working reach the value of the correct option?"""
    numbers = NUMBER.findall(value)
    if numbers:  # "350,000" or "7.5 min": the number is enough
        found = {n.replace(",", "") for n in NUMBER.findall(body)}
        return all(n.replace(",", "") in found for n in numbers)
    return squash(value) in squash(body)


def clean(row):
    """Return (problem, label, solution) or None. Keeps few but good examples."""
    options = parse_options(row["options"])
    if not options or row["correct"] not in LETTERS:
        return None
    lines = [line.strip() for line in row["rationale"].strip().split("\n")
             if line.strip() and not HEADER.match(line.strip())]
    if not 2 <= len(lines) <= 6:
        return None
    match = ANSWER_LINE.match(lines[-1])
    if not match or match.group(1).upper() != row["correct"]:
        return None
    body = "\n".join(lines[:-1])
    if not CALCULATION.search(body) or LETTER_REFERENCE.search(body):
        return None
    if not mentions(lines[-2], options[row["correct"]]):  # the conclusion reaches the answer
        return None
    return {"text": row["question"].strip(), "options": options}, row["correct"], body


def shuffle_options(problem, label, rng):
    """Put the options in a random order: the base model loves some letters."""
    values = [problem["options"][l] for l in LETTERS]
    order = list(range(5))
    rng.shuffle(order)
    options = {LETTERS[i]: values[j] for i, j in enumerate(order)}
    new_label = LETTERS[order.index(LETTERS.index(label))]
    return {**problem, "options": options}, new_label


def example(problem, label, body):
    messages = build_messages(problem, reason=True)
    messages.append({"role": "assistant", "content": f"{body}\nAnswer: {label}"})
    return {"problem": problem, "label": label, "messages": messages}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="work/aqua.jsonl")
    p.add_argument("--n", type=int, default=15000, help="training examples to keep")
    p.add_argument("--eval", type=int, default=300, help="held-out examples, in <out>.eval.jsonl")
    p.add_argument("--seed", type=int, default=int(CONFIG["SEED"]))
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["LLM_BASE_REPO"], revision=CONFIG["LLM_BASE_REVISION"])
    rows = load_dataset("deepmind/aqua_rat", "raw", split="train")

    rng = random.Random(args.seed)
    seen, kept = set(), []
    for row in rows:
        key = re.sub(r"[^a-z0-9]", "", row["question"].lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned = clean(row)
        if not cleaned:
            continue
        problem, label, body = cleaned
        if len(tokenizer(body + "\nAnswer: A")["input_ids"]) >= MAX_SOLUTION_TOKENS:
            continue
        problem, label = shuffle_options(problem, label, rng)
        ex = example(problem, label, body)
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
        if len(tokenizer(text)["input_ids"]) > MAX_TOTAL_TOKENS:
            continue
        kept.append(ex)
    print(f"AQuA-RAT train: {len(rows)} rows, {len(seen)} unique questions, {len(kept)} pass the filter")

    rng.shuffle(kept)
    held_out, train = kept[:args.eval], kept[args.eval:args.eval + args.n]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for path, items in ((args.out, train), (args.out.replace(".jsonl", ".eval.jsonl"), held_out)):
        with open(path, "w") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{path}: {len(items)} examples")
    letters = {l: sum(ex["label"] == l for ex in train) for l in LETTERS}
    print(f"Labels after shuffling: {letters}")


if __name__ == "__main__":
    main()
