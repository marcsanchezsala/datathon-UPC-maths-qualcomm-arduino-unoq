#!/usr/bin/env python3
"""Prepare M3Kang for the Arduino UNO Q, on your laptop.

M3Kang stores each problem as text plus an image. The text has the question
but not the five answer options: those only appear in the image. This script:

1. downloads the parquet shards you ask for from Hugging Face,
2. keeps the languages you ask for,
3. reads the options A to E from each image with OCR (RapidOCR, offline),
4. splits the problems into train and validation by problem ID, so a
   problem and its translations never land on both sides,
5. writes small JSONL files (no images) that you copy to the board.

Usage:
    python3 laptop/prepare_m3kang.py                      # English and Catalan
    python3 laptop/prepare_m3kang.py --languages eng,cat,esp
    python3 laptop/prepare_m3kang.py --val-years 2024     # hold out a whole year
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np  # installed with rapidocr_onnxruntime, like Pillow
from PIL import Image

DATASET = "qualcomm/M3Kang"
SHARD = "data/{split}-{index:05d}-of-{total:05d}.parquet"
SHARDS = {"high_quality": 4, "standard": 10}
LETTERS = "ABCDE"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="high_quality", choices=sorted(SHARDS))
    p.add_argument("--shards", default="0",
                   help="comma-separated shard numbers, or 'all'. Shard 0 of high_quality has "
                        "eng, cat, fra, esp, ind, deu, swh, vie, tur, zho, kor, arb, jpn, lit")
    p.add_argument("--parquet", nargs="*", help="use local parquet files instead of downloading")
    p.add_argument("--languages", default="eng,cat", help="M3Kang language codes, e.g. eng,cat,esp")
    p.add_argument("--val-fraction", type=float, default=0.15, help="share of problem IDs held out")
    p.add_argument("--val-years", default="", help="hold out these years instead, e.g. 2023,2024")
    p.add_argument("--out", default="data", help="output folder")
    p.add_argument("--no-ocr", action="store_true", help="skip OCR; options will be empty")
    p.add_argument("--save-images", action="store_true", help="also save problem images as PNG")
    p.add_argument("--limit", type=int, default=0, help="only process the first N rows (for testing)")
    return p.parse_args()


def shard_paths(args):
    if args.parquet:
        return args.parquet
    from huggingface_hub import hf_hub_download
    total = SHARDS[args.split]
    indexes = range(total) if args.shards == "all" else [int(s) for s in args.shards.split(",")]
    paths = []
    for i in indexes:
        name = SHARD.format(split=args.split, index=i, total=total)
        print(f"Downloading {name} (cached after the first time)...", flush=True)
        paths.append(hf_hub_download(DATASET, name, repo_type="dataset"))
    return paths


# --- Reading the options from the image -------------------------------------
#
# Two passes. First, OCR on the whole image finds the markers "A)" to "E)":
# the last A..E sequence in reading order, usually one row at the bottom
# (OCR noise seen on real M3Kang images: "A)21", "E16", "C" and "13" in
# separate boxes, and a serif "C)" not detected at all). Second, each option
# is cut out of the image along its ink and read again on its own, marker
# included. Reading the whole cell fixes most errors of the first pass: the
# marker is read correctly ("C)8" instead of "18"), figure labels on the same
# row are left out, and stacked fractions are read as "3/4" instead of "34".

MARKER = re.compile(r"^\s*([A-E])\s*([\)）\]:.]?)\s*(.*)$")


def reading_order(boxes):
    """Sort OCR boxes into rows (top to bottom), then left to right."""
    items = []
    for box, text, _score in boxes:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append({"x": min(xs), "y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "text": text})
    items.sort(key=lambda it: it["y"])
    rows, current = [], []
    for it in items:
        if current and abs(it["y"] - current[-1]["y"]) > 0.6 * max(it["h"], current[-1]["h"]):
            rows.append(current)
            current = []
        current.append(it)
    if current:
        rows.append(current)
    return [it for row in rows for it in sorted(row, key=lambda it: it["x"])]


def is_marker(text, letter, strict):
    m = MARKER.match(text)
    if not m or m.group(1) != letter:
        return None
    paren, rest = m.group(2), m.group(3)
    if paren:
        return rest
    if strict:
        return None
    # Without a parenthesis, only accept "C" alone or "E16"-style values,
    # so words such as "ADA" or figure labels with text are not markers.
    if rest == "" or re.match(r"^[-+−\d]", rest):
        return rest
    return None


def find_markers(items):
    """Find the option markers in OCR boxes sorted by reading_order.

    Returns {letter: item} for the last A..E sequence with A and at least three
    of B..E, or {}. The text detector often drops one serif marker, usually
    "C)", so one letter may be missing; option_cells finds its cell anyway.
    """
    starts = [i for i, it in enumerate(items) if is_marker(it["text"], "A", strict=True) is not None]
    starts += [i for i, it in enumerate(items) if i not in starts and is_marker(it["text"], "A", strict=False) is not None]
    for start in sorted(starts, reverse=True):  # the options are the last sequence
        found = {"A": items[start]}
        for it in items[start + 1:]:
            nxt = LETTERS.index(max(found)) + 1
            for letter in LETTERS[nxt:nxt + 2]:  # skip at most one letter at a time
                if is_marker(it["text"], letter, strict=False) is not None:
                    found[letter] = it
                    break
        heights = sorted(it["h"] for it in found.values())
        median = heights[len(heights) // 2]
        if len(found) >= 4 and all(0.4 * median <= h <= 2 * median for h in heights):
            return found
    return {}


def option_rows(found):
    """Place the five letters in rows: ([{"y": centre, "cells": [(letter, x or None), ...]}, ...], height).

    Kangaroo layouts are 5 in a row, 3+2, 2+2+1 or one per row. A missing
    letter goes between its neighbours in their row; when they are in
    different rows, it ends the first row unless the next letter is not in
    the left column, in which case it starts the next row.
    """
    h = sorted(it["h"] for it in found.values())[len(found) // 2]
    rows = []
    for letter in LETTERS:
        it = found.get(letter)
        if it and rows and rows[-1]["y"] is not None and abs(it["y"] - rows[-1]["y"]) <= 0.6 * h:
            rows[-1]["cells"].append((letter, it["x"]))
        elif it:
            rows.append({"y": it["y"], "cells": [(letter, it["x"])]})
        elif letter == "E":
            rows[-1]["cells"].append((letter, None))
        else:  # missing B, C or D: decide once the next marker is known
            rows.append({"y": None, "cells": [(letter, None)]})
    # Resolve a row that holds only a missing letter.
    left = min(it["x"] for it in found.values())
    for i, row in enumerate(rows):
        if row["y"] is not None:
            continue
        prev, nxt = rows[i - 1], rows[i + 1]
        if abs(nxt["y"] - prev["y"]) <= 0.6 * h:  # neighbours on one row: merge
            prev["cells"] += row["cells"] + nxt["cells"]
            nxt["cells"] = []
        elif abs(nxt["cells"][0][1] - left) <= h:  # next row starts at the left: end this row
            prev["cells"] += row["cells"]
        else:  # next row starts later: the missing letter opens it
            nxt["cells"] = [(row["cells"][0][0], left)] + nxt["cells"]
        row["cells"] = []
    rows = [r for r in rows if r["cells"]]
    return rows, h


def ink_runs(mask, max_gap):
    """[(start, end)] of True runs in a 1-D mask, joining gaps up to max_gap."""
    runs = []
    for i in np.flatnonzero(mask):
        if runs and i - runs[-1][1] <= max_gap + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b + 1) for a, b in runs]


def option_cells(ink, rows, h):
    """Cut each option out of the image: {letter: (x0, y0, x1, y1)}.

    The extent comes from the ink, not from OCR boxes: a cell starts at its
    marker and ends at the first gap wider than 1.5 marker heights, so a
    figure label on the same row is not glued to the last option.
    """
    height, width = ink.shape
    cells = {}
    for r, row in enumerate(rows):
        y = row["y"]
        top = int(max(0, y - 2.5 * h, (rows[r - 1]["y"] + y) / 2 if r else 0))
        bottom = int(min(height, y + 2.5 * h, (rows[r + 1]["y"] + y) / 2 if r + 1 < len(rows) else height))
        strip = ink[int(max(top, y - 0.5 * h)):int(min(bottom, y + 0.5 * h))].any(axis=0)
        end = 0
        for k, (letter, x) in enumerate(row["cells"]):
            x0 = int(x - 0.3 * h) if x is not None else end
            nxt = row["cells"][k + 1][1] if k + 1 < len(row["cells"]) else None
            x1 = int(nxt - 0.3 * h) if nxt is not None else width
            runs = ink_runs(strip[max(0, x0):x1], 1.5 * h)
            if not runs:
                return {}
            cs, ce = max(0, x0) + runs[0][0], max(0, x0) + runs[0][1]
            end = ce
            rows_ink = ink[top:bottom, cs:ce].any(axis=1)
            centre = int(y) - top
            block = [run for run in ink_runs(rows_ink, 0.35 * h) if run[0] <= centre + 0.3 * h and run[1] >= centre - 0.3 * h]
            if not block:
                return {}
            cells[letter] = (cs, top + block[0][0], ce, top + block[-1][1])
    return cells if len(cells) == 5 else {}


def recognize(ocr, gray, ink, y0, y1, x0, x1):
    """Read one line of text in a region, trimmed to its ink, with the recognizer only."""
    rows, cols = np.flatnonzero(ink[y0:y1, x0:x1].any(axis=1)), np.flatnonzero(ink[y0:y1, x0:x1].any(axis=0))
    if not len(rows):
        return "", 0.0
    crop = gray[y0 + rows[0]:y0 + rows[-1] + 1, x0 + cols[0]:x0 + cols[-1] + 1]
    padded = np.pad(crop, 6, constant_values=255)
    result, _ = ocr(np.stack([padded] * 3, axis=-1), use_det=False, use_cls=False)
    text, score = result[0] if result else ("", 0.0)
    return text.strip(), float(score)


SUPERSCRIPT = {"2": "²", "3": "³", "。": "°", "°": "°", "º": "°", "o": "°", "'": "′", "′": "′", "’": "′"}


def read_line(ocr, gray, ink, y0, y1, x0, x1, base, cap):
    """Read one line of an option, with raised glyphs read on their own.

    Read as one piece, "n³ - 2" comes out as "n3 - 2" and "2¹³" as "213". A
    glyph whose bottom is well above the baseline and whose top is above the
    capitals is raised. The line is read once with the raised glyphs blanked,
    and each raised glyph is put back after as many characters as the line
    has up to it. "²" and "³" stay as they are, the degree sign is "°", other
    exponents are written "^13". Returns None when a raised glyph is not read
    cleanly, the line and its prefix read differently, or a square root was
    lost.
    """
    raised, prev = [], False
    for a, b in ink_runs(ink[y0:y1, x0:x1].any(axis=0), 0):
        rows = np.flatnonzero(ink[y0:y1, x0 + a:x0 + b].any(axis=1))
        up = rows[-1] < base - 0.45 * cap and rows[0] < base - 0.75 * cap
        if up and prev:
            raised[-1][1] = b
        elif up:
            raised.append([a, b])
        prev = up
    # A square root's bar runs above the capitals, which no letter does. The
    # recognizer sometimes drops the root sign: "√2" read as "2".
    above = ink[y0:y0 + max(0, int(base - 1.05 * cap)), x0:x1]
    root = any(b - a >= 0.5 * cap for row in above for a, b in ink_runs(row, 0))
    if not raised:
        text = recognize(ocr, gray, ink, y0, y1, x0, x1)[0]
        return None if root and not re.search("[√Vv]", text) else text
    blank_gray, blank_ink = gray.copy(), ink.copy()
    for a, b in raised:
        blank_gray[y0:y1, x0 + a:x0 + b], blank_ink[y0:y1, x0 + a:x0 + b] = 255, False
    text, inserted = recognize(ocr, blank_gray, blank_ink, y0, y1, x0, x1)[0], 0
    for a, b in raised:
        sup, sup_score = recognize(ocr, gray, ink, y0, y1, x0 + a, x0 + b)
        if sup in SUPERSCRIPT:
            sup = SUPERSCRIPT[sup]
        elif sup.isalnum() and sup_score >= 0.9:
            sup = "^" + sup
        else:
            return None
        prefix = recognize(ocr, blank_gray, blank_ink, y0, y1, x0, x0 + a)[0].replace(" ", "")
        plain = re.sub(r"\^\w+|[²³°′\s]", "", text)  # the line without what was put back
        if not prefix or not plain.startswith(prefix):
            return None  # the two readings do not line up
        n = len(prefix) + inserted
        chars = [i for i, c in enumerate(text) if c != " "]
        at = chars[n - 1] + 1
        if sup.startswith("^") and text[at:at + 1].isalnum():
            sup += " "  # "2^100 3^20", not "2^1003^20"
        text = text[:at] + sup + text[at:]
        inserted += len(sup.replace(" ", ""))
    return None if root and not re.search("[√Vv]", text) else text


def read_cell(ocr, gray, ink, cell, letter):
    """Read one option cell, marker included. Returns the value, or None.

    The cell is split into glyph groups. A group taller than 1.5 times the
    marker letter with a free-standing horizontal bar that has wide ink above
    and below is a stacked fraction, read as "numerator/denominator". A group
    taller than twice the letter that is not a fraction is a picture: the cell
    is unreadable. Runs of the other groups are read as one line. The first part
    must start with this cell's marker, such as "C)".
    """
    x0, y0, x1, y1 = cell
    sub = ink[y0:y1, x0:x1]
    groups = ink_runs(sub.any(axis=0), 2)
    if not groups:
        return None
    first = ink_runs(sub.any(axis=0), 0)[0]  # the marker letter
    letter_rows = np.flatnonzero(sub[:, first[0]:first[1]].any(axis=1))
    letter_h = letter_rows[-1] - letter_rows[0] + 1
    parts, run = [], None
    for a, b in groups + [(None, None)]:
        if a is not None:
            g = sub[:, a:b]
            rows = np.flatnonzero(g.any(axis=1))
            tall = rows[-1] - rows[0] + 1
            wide = lambda part: part.any(axis=0).sum() >= 0.2 * (b - a)
            # The fraction bar has blank rows right above and below it; the
            # crossbar of a "4" runs into the digit's stem.
            bars = [(t, u) for t, u in ink_runs(g.mean(axis=1) >= 0.85, 0)
                    if 0 < t and u < len(g) and not g[t - 1].any() and not g[u].any()
                    and wide(g[:t - 1]) and wide(g[u + 1:])]
            fraction = tall > 1.5 * letter_h and bars
            if not fraction and tall <= 2 * letter_h:
                run = (run[0], b) if run else (a, b)
                continue
        if run:  # read the line of normal groups before this one
            line = read_line(ocr, gray, ink, y0, y1, x0 + run[0], x0 + run[1], letter_rows[-1], letter_h)
            if line is None:
                return None
            parts.append(line)
            run = None
        if a is None:
            break
        if not fraction:
            return None  # a picture
        top, bottom = bars[0]
        num = recognize(ocr, gray, ink, y0, y0 + top - 1, x0 + a, x0 + b)[0]
        den = recognize(ocr, gray, ink, y0 + bottom + 1, y1, x0 + a, x0 + b)[0]
        if not num or not den:
            return None
        wrap = lambda t: f"({t})" if re.search(r"[\s+\-−]", t) else t
        parts.append(f"{wrap(num)}/{wrap(den)}")
    m = MARKER.match(parts[0]) if parts else None
    if not m or m.group(1) != letter or not m.group(2):
        return None
    return " ".join([m.group(3)] + parts[1:]).strip() or None


def read_options(ocr, image):
    """Read the options A..E from a problem image (PNG bytes).

    Returns {"A": "12", ..., "E": "24"}, or {} when the options are pictures
    or cannot be read with confidence.
    """
    gray = np.asarray(Image.open(io.BytesIO(image)).convert("L"))
    boxes, _ = ocr(gray)
    found = find_markers(reading_order(boxes or []))
    if not found:
        return {}
    rows, h = option_rows(found)
    ink = gray < 128
    options = {}
    for letter, cell in sorted(option_cells(ink, rows, h).items()):
        read = read_cell(ocr, gray, ink, cell, letter)
        if read is None:
            return {}
        options[letter] = normalize(read)
    return options if len(options) == 5 else {}


UNIT = r"(?:mm|cm|dm|km|m|kg|g|ml|l|min|h|s)(?![^\W\d_²³])"


def normalize(value):
    """Tidy OCR text so that the same option reads the same in every image."""
    value = value.translate(str.maketrans("（）［］，：∈", "()[],:€"))  # "€" is read as "∈"
    value = re.sub(r"\s*([=+×÷<>])\s*", r" \1 ", value)
    value = re.sub(r"(\d)\s*(" + UNIT + ")", r"\1 \2", value)  # "16cm" -> "16 cm"
    value = re.sub(r"\s+%", "%", value)
    value = re.sub(r"(?<![^\W\d_])(mm|cm|dm|km|m)([23])(?!\d)", lambda m: m[1] + "²³"[int(m[2]) - 2], value)
    return re.sub(r"\s+", " ", value).strip()


def fix_roots(options, text):
    """The recognizer often reads a square root as "V" or "v": "V2" is "√2", unless
    the question itself names things V1, V2..."""
    if re.search(r"[Vv]\d", text):
        return options
    return {k: re.sub(r"[Vv](?=[\d(])", "√", v) for k, v in options.items()}


def language_free(options):
    """True when the options have no words, only numbers, symbols, units and
    single letters, so they are the same in every language."""
    text = re.sub(r"(?<![^\W\d_])" + UNIT, "", " ".join(options.values()))
    return not re.search(r"[^\W\d_²³]{2,}", text)


def cross_check(rows):
    """Use the translations of a problem to check and complete its options.

    Language-free options must read the same in every language. Sets
    options_check to "agree" (two or more languages read the same),
    "conflict" (they differ, so the options are dropped: one of them is
    wrong), "single" (only one reading) or "none" (no options). A language
    whose image could not be read gets the agreed options, and
    options_source says which language they were read from.
    """
    by_id = {}
    for r in rows:
        r["options_source"] = r["language"] if r["options"] else ""
        r["options_check"] = "single" if r["options"] else "none"
        by_id.setdefault(r["id"], []).append(r)
    key = lambda options: re.sub(r"\s+", "", json.dumps(options, ensure_ascii=False))
    for group in by_id.values():
        free = [r for r in group if r["options"] and language_free(r["options"])]
        if len({key(r["options"]) for r in free}) > 1:
            for r in free:
                r["options"], r["options_source"], r["options_check"] = {}, "", "conflict"
            continue
        if not free:
            continue
        check = "agree" if len(free) > 1 else "single"
        for r in group:
            if r in free:
                r["options_check"] = check
            elif not r["options"]:
                r["options"] = dict(free[0]["options"])
                r["options_source"], r["options_check"] = free[0]["language"], check


def main():
    args = parse_args()
    import pyarrow.parquet as pq

    languages = set(args.languages.split(","))
    val_years = set(y for y in args.val_years.split(",") if y)
    os.makedirs(args.out, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    ocr = None
    if not args.no_ocr:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

    cache_path = os.path.join(args.out, ".ocr_cache_v2.jsonl")  # v1 held reads of the old parser
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row["options"]
    cache_file = open(cache_path, "a")

    rows = []
    t0 = time.time()
    for path in shard_paths(args):
        table = pq.read_table(path, columns=["language"])
        wanted = [i for i, lang in enumerate(table.column(0).to_pylist()) if lang in languages]
        if not wanted:
            continue
        full = pq.read_table(path).take(wanted)
        for i in range(full.num_rows):
            if args.limit and len(rows) >= args.limit:
                break
            r = {c: full.column(c)[i].as_py() for c in ("id", "language", "level", "text_only", "label", "text")}
            key = f"{r['id']}|{r['language']}"
            image = full.column("image")[i].as_py()["bytes"]
            if args.save_images:
                with open(os.path.join(args.out, "images", key.replace("|", "_") + ".png"), "wb") as f:
                    f.write(image)
            if key in cache:
                options = cache[key]
            elif ocr:
                options = read_options(ocr, image)
                cache[key] = options
                cache_file.write(json.dumps({"key": key, "options": options}, ensure_ascii=False) + "\n")
            else:
                options = {}
            year = r["id"].split("_")[1]
            if val_years:
                split = "val" if year in val_years else "train"
            else:
                bucket = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000
                split = "val" if bucket < args.val_fraction * 1000 else "train"
            rows.append({
                "id": r["id"], "language": r["language"], "level": r["level"], "year": int(year),
                "text_only": r["text_only"], "text": r["text"].strip(), "options": fix_roots(options, r["text"]),
                "label": r["label"], "split": split,
            })
            if len(rows) % 100 == 0:
                rate = len(rows) / (time.time() - t0)
                print(f"  {len(rows)} rows, {rate:.1f} rows/s", flush=True)
    cache_file.close()
    cross_check(rows)

    for split in ("train", "val"):
        out = os.path.join(args.out, f"{split}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "split"}, ensure_ascii=False) + "\n")

    report(rows, args.out)


def report(rows, out):
    if not rows:
        sys.exit("No rows matched. Check --languages and --shards.")
    by = Counter((r["split"], r["language"]) for r in rows)
    with_options = sum(1 for r in rows if r["options"])
    text_only = [r for r in rows if r["text_only"]]
    labels = Counter(r["label"] for r in rows if r["split"] == "train")
    top, count = labels.most_common(1)[0]
    print()
    print(f"Wrote {out}/train.jsonl and {out}/val.jsonl")
    for (split, lang), n in sorted(by.items()):
        print(f"  {split:5} {lang}: {n}")
    print(f"Options read from the image: {with_options}/{len(rows)} ({100 * with_options / len(rows):.0f}%)")
    if text_only:
        ok = sum(1 for r in text_only if r["options"])
        print(f"  of which text-only problems: {ok}/{len(text_only)} ({100 * ok / len(text_only):.0f}%)")
    checks = Counter(r["options_check"] for r in rows)
    copied = sum(1 for r in rows if r["options"] and r["options_source"] != r["language"])
    print(f"  same in two languages: {checks['agree']}, one reading only: {checks['single']}, "
          f"copied from another language: {copied}, dropped because languages disagree: {checks['conflict']}")
    print(f"Most common answer in train: {top} ({100 * count / sum(labels.values()):.1f}%). "
          f"Always answering {top} is the floor to beat.")


if __name__ == "__main__":
    main()
