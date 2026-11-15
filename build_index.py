#!/usr/bin/env python3
"""Embed the training problems into a SQLite index for retrieval (RAG).

    python3 build_index.py                       # data/train.jsonl -> data/index.db
    python3 build_index.py --only-with-options   # skip problems whose options are pictures

Run it once on the board, after you copy the data. Only training problems go
in, so validation problems can never retrieve themselves.
"""
import argparse
import json
import os
import time

from copilot import rag


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/train.jsonl")
    p.add_argument("--db", default="data/index.db")
    p.add_argument("--languages", default="", help="e.g. eng,cat (default: all in the file)")
    p.add_argument("--only-with-options", action="store_true")
    args = p.parse_args()

    languages = set(filter(None, args.languages.split(",")))
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems
                if (not languages or x.get("language") in languages)
                and (x.get("options") or not args.only_with_options)]

    db_dir = os.path.dirname(os.path.abspath(args.db))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    if os.path.exists(args.db):
        os.remove(args.db)

    if not problems:
        print(f"No problems matched criteria from {args.input}")
        return

    start = time.perf_counter()
    for done in rag.build(args.db, problems):
        elapsed = max(0.001, time.perf_counter() - start)
        rate = done / elapsed
        print(f"\r{done}/{len(problems)} problems embedded, {rate:.1f}/s", end="", flush=True)
    size = os.path.getsize(args.db) / 1e6
    print(f"\nWrote {args.db} ({size:.1f} MB) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
