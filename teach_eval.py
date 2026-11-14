#!/usr/bin/env python3
"""Measure the Teach features on the validation set, without a student.

For every validation problem with numeric options, pretend the student chose
each wrong option in turn, and record whether the rules diagnose a slip and
whether a counterexample from train is found. Also measures rung 1 (an easier
similar problem) and rung 2 (topic) coverage. No model is called.

    python teach_eval.py                                 # laptop: keyword index, no servers
    python3 teach_eval.py --index data/index.db --min-score 0.6   # board: bge-small index
    python teach_eval.py --languages eng --limit 50 --name eng50

Writes results/teach-<time>-<name>.csv (one row per simulated wrong answer)
and appends one summary line to results/teach_ablations.csv.
"""
import argparse
import csv
import json
import os
import time
from collections import Counter

from copilot import tutor


def load(path, languages):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [p for p in rows if not languages or p["language"] in languages]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--index", default="lexical", help="'lexical' (laptop) or a path to index.db (board)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="similarity threshold (default 0.3 lexical, 0.6 bge)")
    ap.add_argument("--languages", default="", help="e.g. eng,cat")
    ap.add_argument("--limit", type=int, default=0, help="first N validation problems (0 = all)")
    ap.add_argument("--name", default="run")
    ap.add_argument("--k", type=int, default=10, help="neighbours searched per problem")
    ap.add_argument("--same-level", action="store_true",
                    help="rung 1 accepts a similar problem at the same level when none is easier")
    ap.add_argument("--search-lang", default="eng",
                    help="search with this language version of each problem (same id); '' = as given")
    args = ap.parse_args()

    languages = set(filter(None, args.languages.split(",")))
    train, val = load(args.train, languages), load(args.val, languages)
    if args.limit:
        val = val[:args.limit]

    if args.index == "lexical":
        from copilot.lexindex import LexicalIndex
        index, default_score = LexicalIndex(train), 0.3
    else:
        from copilot.rag import Index
        index, default_score = Index(args.index), 0.6
    tutor.MIN_SCORE = args.min_score if args.min_score is not None else default_score
    tutor.SEARCH_LANG = args.search_lang
    tutor.K_NEIGHBOURS = args.k
    tutor.SAME_LEVEL = args.same_level
    library = tutor.Library.load(args.train, args.val) if args.search_lang else None

    rows, problems_numeric, rung1, rung1_easier, rung2 = [], 0, 0, 0, 0
    search_secs = []
    for n, p in enumerate(val, 1):
        t0 = time.perf_counter()
        hits = tutor.neighbours(p, index, library=library)
        search_secs.append(time.perf_counter() - t0)
        pick = tutor.easier_example(p, hits)
        if pick:
            rung1 += 1
            rung1_easier += int(pick[1].get("level", 0) < p.get("level", 0))
        if tutor.topic_of(p) or any(tutor.topic_of(q) for _s, q in hits[:5]):
            rung2 += 1
        opts = p.get("options") or {}
        if tutor.parse_num(opts.get(p.get("label"), "")) is None:
            continue
        problems_numeric += 1
        for letter in opts:
            if letter == p["label"] or tutor.parse_num(opts[letter]) is None:
                continue
            t1 = time.perf_counter()
            diag = tutor.diagnose(p, letter)
            ce = tutor.counterexample(p, letter, hits)
            rows.append({
                "id": p["id"], "language": p["language"], "level": p.get("level", 0),
                "chosen": letter, "correct": p["label"],
                "errors": "|".join(e[0] for e in diag[2]) if diag else "",
                "diagnosed": int(bool(diag and diag[2])),
                "counterexample": int(bool(ce)),
                "ce_error": ce["error"] if ce else "",
                "ce_score": round(ce["score"], 3) if ce else "",
                "ce_neighbour": ce["neighbour"]["id"] if ce else "",
                "ms": round(1000 * (time.perf_counter() - t1), 2),
            })
        print(f"\r{n}/{len(val)} problems", end="", flush=True)
    print()

    sims = len(rows)
    diagnosed = sum(r["diagnosed"] for r in rows)
    found = sum(r["counterexample"] for r in rows)
    by_type = Counter(r["ce_error"] for r in rows if r["counterexample"])
    diag_types = Counter(e for r in rows for e in r["errors"].split("|") if e)
    by_level = Counter()
    by_level_found = Counter()
    for r in rows:
        by_level[r["level"]] += 1
        by_level_found[r["level"]] += r["counterexample"]
    mean_search = 1000 * sum(search_secs) / max(len(search_secs), 1)

    print(f"Index: {args.index}   threshold: {tutor.MIN_SCORE}   k: {args.k}   same level: {args.same_level}"
          f"   search language: {args.search_lang or 'as given'}")
    print(f"Validation problems:               {len(val)}")
    print(f"  with a numeric answer:           {problems_numeric} ({pct(problems_numeric, len(val))})")
    print(f"  rung 1 found:                    {rung1} ({pct(rung1, len(val))}), "
          f"of which easier: {rung1_easier}, same level: {rung1 - rung1_easier}")
    print(f"  rung 2 (topic) found:            {rung2} ({pct(rung2, len(val))})")
    print(f"Simulated wrong answers:           {sims}")
    print(f"  slip diagnosed by rules:         {diagnosed} ({pct(diagnosed, sims)})")
    print(f"  counterexample found:            {found} ({pct(found, sims)})")
    print(f"  of diagnosed, with counterexample: {pct(found, diagnosed)}")
    print("Diagnosed slip types:  " + ", ".join(f"{k} {v}" for k, v in diag_types.most_common()))
    print("Counterexamples by type: " + ", ".join(f"{k} {v}" for k, v in by_type.most_common()))
    print("Counterexample rate by level: " + ", ".join(
        f"L{lv} {pct(by_level_found[lv], by_level[lv])} (n={by_level[lv]})" for lv in sorted(by_level)))
    print(f"Mean search time per problem: {mean_search:.1f} ms")

    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = f"results/teach-{stamp}-{args.name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        fields = list(rows[0]) if rows else ["id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = "results/teach_ablations.csv"
    new = not os.path.exists(summary)
    with open(summary, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "index", "threshold", "k", "same_level", "languages", "problems",
                        "numeric", "rung1", "rung1_easier", "rung2", "simulated", "diagnosed",
                        "counterexample", "search_ms"])
        w.writerow([stamp, args.name, args.index, tutor.MIN_SCORE, args.k, int(args.same_level),
                    args.languages or "all", len(val), problems_numeric, rung1, rung1_easier, rung2,
                    sims, diagnosed, found, round(mean_search, 1)])
    print(f"Wrote {out} and one line in {summary}")


if __name__ == "__main__":
    main()
