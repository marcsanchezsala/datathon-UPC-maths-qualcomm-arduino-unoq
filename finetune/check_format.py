#!/usr/bin/env python3
"""Fail if the training prompts differ from what the copilot sends at inference.

    python check_format.py --data work/aqua.jsonl                             # Hugging Face side only
    python check_format.py --data work/aqua.jsonl --server http://127.0.0.1:8092  # and llama-server

Checks, on a few training examples:
1. The stored messages are what copilot.llm.build_messages(reason=True) builds today.
2. The prompt train.py learns from is the prompt llama-server renders from those messages,
   as text and as tokens (needs --server: any llama-server with this model family).
"""
import argparse
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from copilot.client import post  # noqa: E402
from copilot.config import CONFIG  # noqa: E402
from copilot.llm import build_messages  # noqa: E402
from train import render_prompt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="work/aqua.jsonl")
    p.add_argument("--server", default="", help="e.g. http://127.0.0.1:8092")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["LLM_BASE_REPO"], revision=CONFIG["LLM_BASE_REVISION"])
    with open(args.data) as f:
        examples = [json.loads(line) for _, line in zip(range(args.n), f)]
    for ex in examples:
        expected = build_messages(ex["problem"], reason=True)
        assert ex["messages"][:-1] == expected, f"stale data, run prepare_aqua.py again:\n{expected}"
        assert ex["messages"][-1]["content"].endswith(f"\nAnswer: {ex['label']}")
        prompt = render_prompt(tokenizer, expected)
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(prompt[-80:])
        if args.server:
            served = post(args.server + "/apply-template", {"messages": expected}, timeout=30)["prompt"]
            assert served == prompt, f"templates differ:\nHF:     {prompt!r}\nserver: {served!r}"
            tokens = post(args.server + "/tokenize", {"content": served}, timeout=30)["tokens"]
            assert tokens == tokenizer(prompt, add_special_tokens=False)["input_ids"], "tokenizers differ"
    print(f"OK: {len(examples)} examples match build_messages"
          + (f" and {args.server} renders and tokenizes them the same" if args.server else ""))


if __name__ == "__main__":
    main()
