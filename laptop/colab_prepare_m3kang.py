#!/usr/bin/env python3
"""Colab-friendly M3Kang dataset preparation and option extraction.

Reuses laptop/prepare_m3kang.py OCR pipeline and supplements missing visual
options (A..E) using Qwen/Qwen2.5-VL-7B-Instruct on GPU. Preserves train/val
schema for predict.py and eval.py, tracking empty option rates before and after.

Usage:
    python colab_prepare_m3kang.py --self-test
    python colab_prepare_m3kang.py --mode ocr --limit 50
    python colab_prepare_m3kang.py --mode hybrid --model Qwen/Qwen2.5-VL-7B-Instruct
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

# Ensure laptop/ directory is in sys.path whether run from repo root or laptop/
_LAPTOP_DIR = os.path.dirname(os.path.abspath(__file__))
if _LAPTOP_DIR not in sys.path:
    sys.path.insert(0, _LAPTOP_DIR)

from prepare_m3kang import LETTERS, SHARDS, parse_options, shard_paths


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
    p.add_argument("--save-images", action="store_true", help="also save problem images as PNG")
    p.add_argument("--limit", type=int, default=0, help="only process the first N rows (for testing)")
    p.add_argument("--mode", choices=["ocr", "hybrid"], default="hybrid",
                   help="ocr: RapidOCR only; hybrid: RapidOCR + VLM for missing options")
    p.add_argument("--vlm-all", action="store_true",
                   help="run VLM even on rows where RapidOCR already found options")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Hugging Face VLM model identifier")
    p.add_argument("--batch-size", type=int, default=16, help="VLM batch size on GPU")
    p.add_argument("--drop-empty-train", action="store_true",
                   help="omit rows with options == {} when writing train.jsonl (keeps val.jsonl intact)")
    p.add_argument("--self-test", action="store_true", help="run unit self-checks and exit")
    return p.parse_args()


# --- VLM Extraction & Helpers -----------------------------------------------


def is_language_independent(options: dict) -> bool:
    """Return True if options are empty (picture options) or contain no multi-letter words."""
    if not options:
        return True
    return all(re.search(r"[A-Za-zÀ-ÿ]{2,}", str(v)) is None for v in options.values())


def _validate_options_dict(val: dict) -> dict:
    res = {}
    for letter in LETTERS:
        if letter not in val:
            return {}
        v = val[letter]
        if v is None:
            return {}
        v_str = str(v).strip()
        if not v_str or v_str.lower() in ("null", "none"):
            return {}
        res[letter] = v_str
    return res


def extract_vlm_json(raw_text: str) -> dict:
    """Extract a 5-option JSON dictionary (A..E) from raw VLM output, or return {}."""
    if not raw_text or not isinstance(raw_text, str):
        return {}

    # Try fenced code blocks first
    fence_matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, re.IGNORECASE)
    for cand in fence_matches:
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                res = _validate_options_dict(val)
                if res:
                    return res
        except Exception:
            pass

    # Try outer brace match
    outer = re.search(r"\{[\s\S]*\}", raw_text)
    if outer:
        try:
            val = json.loads(outer.group(0))
            if isinstance(val, dict):
                res = _validate_options_dict(val)
                if res:
                    return res
        except Exception:
            pass

    # Scan for any valid JSON object starting with '{'
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw_text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw_text[i:])
                if isinstance(obj, dict):
                    res = _validate_options_dict(obj)
                    if res:
                        return res
            except Exception:
                pass

    return {}


def load_vlm(model_id: str):
    """Load Qwen2.5-VL model and processor onto CUDA in bfloat16."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run on a GPU runtime (e.g. Colab A100/T4) "
            "or use '--mode ocr' to run offline OCR only."
        )
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass

    print(f"Loading VLM {model_id} on CUDA...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=512 * 28 * 28)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


def vlm_extract_options_batch(model, processor, image_bytes_list: list) -> list[dict]:
    """Batch prompt VLM to extract options A..E, returning {} if options are pictures."""
    if not image_bytes_list:
        return []
    from PIL import Image
    import torch

    prompt = (
        "Extract ONLY the 5 multiple-choice options A, B, C, D, E from this math problem image "
        "as a single JSON object:\n"
        '{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}\n'
        "Only extract options if they are text, numbers, or mathematical formulas.\n"
        "If the options A-E are drawings, geometric figures, shapes, or pictures, "
        "immediately output {} and nothing else."
    )
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
    texts = []
    for img in imgs:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)

    trimmed_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [extract_vlm_json(t) for t in decoded]


def vlm_extract_options(model, processor, image_bytes: bytes) -> dict:
    """Extract options for a single image by wrapping vlm_extract_options_batch."""
    return vlm_extract_options_batch(model, processor, [image_bytes])[0]


# --- Metrics & Reporting ----------------------------------------------------


def compute_empty_stats(rows: list) -> dict:
    """Compute overall, type-based, and per-language stats for questions without options."""
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "final_empty": 0,
            "final_empty_pct": 0.0,
            "final_with": 0,
            "final_with_pct": 0.0,
            "ocr_empty": 0,
            "ocr_empty_pct": 0.0,
            "ocr_with": 0,
            "ocr_with_pct": 0.0,
            "by_type": {},
            "by_language": {},
        }

    final_empty = sum(1 for r in rows if not r.get("options"))
    final_with = total - final_empty
    ocr_empty = sum(1 for r in rows if not r.get("ocr_has_options", False))
    ocr_with = total - ocr_empty

    by_type = {}
    for is_text in (True, False):
        sub = [r for r in rows if r.get("text_only") == is_text]
        n_sub = len(sub)
        n_empty = sum(1 for r in sub if not r.get("options"))
        n_with = n_sub - n_empty
        by_type["text_only" if is_text else "visual"] = {
            "total": n_sub,
            "empty": n_empty,
            "empty_pct": (100.0 * n_empty / n_sub) if n_sub else 0.0,
            "with": n_with,
            "with_pct": (100.0 * n_with / n_sub) if n_sub else 0.0,
        }

    by_language = {}
    languages = sorted(set(r.get("language", "unknown") for r in rows))
    for lang in languages:
        sub = [r for r in rows if r.get("language") == lang]
        n_sub = len(sub)
        n_empty = sum(1 for r in sub if not r.get("options"))
        n_with = n_sub - n_empty
        by_language[lang] = {
            "total": n_sub,
            "empty": n_empty,
            "empty_pct": (100.0 * n_empty / n_sub) if n_sub else 0.0,
            "with": n_with,
            "with_pct": (100.0 * n_with / n_sub) if n_sub else 0.0,
        }

    return {
        "total": total,
        "final_empty": final_empty,
        "final_empty_pct": 100.0 * final_empty / total,
        "final_with": final_with,
        "final_with_pct": 100.0 * final_with / total,
        "ocr_empty": ocr_empty,
        "ocr_empty_pct": 100.0 * ocr_empty / total,
        "ocr_with": ocr_with,
        "ocr_with_pct": 100.0 * ocr_with / total,
        "by_type": by_type,
        "by_language": by_language,
    }


def report(rows: list, out: str, mode: str = "hybrid", reused_cross_lang: int = 0, dropped_train: int | None = None):
    """Print split counts, OCR vs VLM option recovery metrics, and floor accuracy."""
    if not rows:
        sys.exit("No rows matched. Check --languages and --shards.")
    by = Counter((r["split"], r["language"]) for r in rows)
    stats = compute_empty_stats(rows)
    total = stats["total"]

    print()
    print(f"Wrote {out}/train.jsonl and {out}/val.jsonl")
    for (split, lang), n in sorted(by.items()):
        print(f"  {split:5} {lang}: {n}")
    if dropped_train is not None:
        print(f"  Empty-option rows dropped from train (--drop-empty-train): {dropped_train}")

    print("\nOptions Extraction:")
    print(f"  Baseline OCR:")
    print(f"    Questions without options (options == {{}}): {stats['ocr_empty']}/{total} ({stats['ocr_empty_pct']:.1f}%)")
    print(f"    Questions with options:                     {stats['ocr_with']}/{total} ({stats['ocr_with_pct']:.1f}%)")

    stage_name = "Final (Hybrid: RapidOCR + VLM)" if mode == "hybrid" else "Final (RapidOCR only)"
    print(f"  {stage_name}:")
    print(f"    Questions without options (options == {{}}): {stats['final_empty']}/{total} ({stats['final_empty_pct']:.1f}%)")
    print(f"    Questions with options:                     {stats['final_with']}/{total} ({stats['final_with_pct']:.1f}%)")
    if reused_cross_lang > 0:
        print(f"    Cross-language option reuses:               {reused_cross_lang}")

    print("  Breakdown by problem type:")
    for type_name, s in stats["by_type"].items():
        if s["total"] > 0:
            print(f"    - {type_name:10}: without options {s['empty']}/{s['total']} ({s['empty_pct']:.1f}%), with options {s['with']}/{s['total']} ({s['with_pct']:.1f}%)")

    print("  Breakdown per language:")
    for lang, s in stats["by_language"].items():
        if s["total"] > 0:
            print(f"    - {lang:5}: without options {s['empty']}/{s['total']} ({s['empty_pct']:.1f}%), with options {s['with']}/{s['total']} ({s['with_pct']:.1f}%)")

    train_labels = [r["label"] for r in rows if r["split"] == "train"]
    if train_labels:
        labels = Counter(train_labels)
        top, count = labels.most_common(1)[0]
        print(f"\nMost common answer in train: {top} ({100 * count / len(train_labels):.1f}%). "
              f"Always answering {top} is the floor to beat.")
    else:
        print("\nNo rows in train split; floor to beat skipped.")


# --- Self-Test --------------------------------------------------------------


def run_self_test():
    """Run standalone self-checks without external dependencies."""
    # 1. Clean JSON
    clean_json = '{"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}'
    assert extract_vlm_json(clean_json) == {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}

    # 2. Markdown fenced with extra prose and UTF-8 math symbols
    fenced_json = (
        "Here are the extracted options from the image:\n```json\n"
        "{\n"
        '  "A": "√2",\n'
        '  "B": "π",\n'
        '  "C": "1/2",\n'
        '  "D": "\\u4e00",\n'
        '  "E": "x^2 + y^2"\n'
        "}\n"
        "```\nHope this helps!"
    )
    res = extract_vlm_json(fenced_json)
    assert res == {"A": "√2", "B": "π", "C": "1/2", "D": "\u4e00", "E": "x^2 + y^2"}

    # 3. Incomplete JSON (missing E)
    missing_e = '{"A": "1", "B": "2", "C": "3", "D": "4"}'
    assert extract_vlm_json(missing_e) == {}

    # 4. Incomplete JSON (empty/null value for C)
    assert extract_vlm_json('{"A": "1", "B": "2", "C": "", "D": "4", "E": "5"}') == {}
    assert extract_vlm_json('{"A": "1", "B": "2", "C": null, "D": "4", "E": "5"}') == {}
    assert extract_vlm_json('{"A": "1", "B": "2", "C": "None", "D": "4", "E": "5"}') == {}
    assert extract_vlm_json('{"A": "1", "B": "2", "C": "null", "D": "4", "E": "5"}') == {}

    # 5. Invalid / empty text / empty dict
    assert extract_vlm_json("No options found in this image.") == {}
    assert extract_vlm_json("") == {}
    assert extract_vlm_json(None) == {}
    assert extract_vlm_json("{}") == {}
    assert extract_vlm_json("```json\n{}\n```") == {}

    # 6. is_language_independent checks
    assert is_language_independent({}) is True
    assert is_language_independent({"A": "12", "B": "-3.5", "C": "1/2", "D": "2^10", "E": "x + 1"}) is True
    assert is_language_independent({"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}) is True
    assert is_language_independent({"A": "Monday", "B": "Tuesday", "C": "Wednesday", "D": "Thursday", "E": "Friday"}) is False
    assert is_language_independent({"A": "Dilluns", "B": "Dimarts", "C": "Dimecres", "D": "Dijous", "E": "Divendres"}) is False
    assert is_language_independent({"A": "3 cm", "B": "4 cm", "C": "5 cm", "D": "6 cm", "E": "7 cm"}) is False

    # 7. compute_empty_stats checks
    assert compute_empty_stats([])["total"] == 0
    assert compute_empty_stats([])["final_empty_pct"] == 0.0

    dummy_rows = [
        {"id": "1", "language": "eng", "text_only": True, "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}, "ocr_has_options": True},
        {"id": "2", "language": "eng", "text_only": True, "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}, "ocr_has_options": False},
        {"id": "3", "language": "cat", "text_only": False, "options": {}, "ocr_has_options": False},
        {"id": "4", "language": "cat", "text_only": False, "options": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}, "ocr_has_options": True},
    ]
    stats = compute_empty_stats(dummy_rows)
    assert stats["total"] == 4
    assert stats["final_empty"] == 1
    assert abs(stats["final_empty_pct"] - 25.0) < 1e-6
    assert stats["final_with"] == 3
    assert abs(stats["final_with_pct"] - 75.0) < 1e-6
    assert stats["ocr_empty"] == 2
    assert abs(stats["ocr_empty_pct"] - 50.0) < 1e-6
    assert stats["by_type"]["text_only"]["empty"] == 0
    assert stats["by_type"]["visual"]["empty"] == 1
    assert stats["by_language"]["eng"]["empty"] == 0
    assert stats["by_language"]["cat"]["empty"] == 1

    print("Self-test passed.")


# --- Main Pipeline ----------------------------------------------------------


def main(args=None):
    if args is None:
        args = parse_args()

    if args.self_test:
        run_self_test()
        return

    import pyarrow.parquet as pq

    languages = set(args.languages.split(","))
    val_years = set(y for y in args.val_years.split(",") if y)
    os.makedirs(args.out, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    # Initialize RapidOCR
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    # Shared OCR cache
    ocr_cache_path = os.path.join(args.out, ".ocr_cache.jsonl")
    ocr_cache = {}
    if os.path.exists(ocr_cache_path):
        with open(ocr_cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ocr_cache[row["key"]] = row["options"]
    ocr_cache_file = open(ocr_cache_path, "a", encoding="utf-8")

    # VLM cache (in hybrid mode)
    vlm_cache = {}
    vlm_cache_file = None
    vlm_model, vlm_processor = None, None
    if args.mode == "hybrid":
        vlm_cache_path = os.path.join(args.out, ".vlm_cache.jsonl")
        if os.path.exists(vlm_cache_path):
            with open(vlm_cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        vlm_cache[(row["key"], row.get("model", args.model))] = row["options"]
        vlm_cache_file = open(vlm_cache_path, "a", encoding="utf-8")

    def get_vlm():
        nonlocal vlm_model, vlm_processor
        if vlm_model is None:
            vlm_model, vlm_processor = load_vlm(args.model)
        return vlm_model, vlm_processor

    rows = []
    pending_images = {}
    id_ocr_numeric = {}
    t0 = time.time()
    for path in shard_paths(args):
        if args.limit and len(rows) >= args.limit:
            break
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

            # 1. OCR Stage
            if key in ocr_cache:
                ocr_options = ocr_cache[key]
            else:
                boxes, _ = ocr(image)
                ocr_options = parse_options(boxes or [])
                ocr_cache[key] = ocr_options
                ocr_cache_file.write(json.dumps({"key": key, "options": ocr_options}, ensure_ascii=False) + "\n")
                ocr_cache_file.flush()

            if ocr_options and is_language_independent(ocr_options):
                id_ocr_numeric[r["id"]] = ocr_options

            if args.mode == "hybrid" and (args.vlm_all or not ocr_options):
                pending_images[key] = image

            year = r["id"].split("_")[1]
            if val_years:
                split = "val" if year in val_years else "train"
            else:
                bucket = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000
                split = "val" if bucket < args.val_fraction * 1000 else "train"

            rows.append({
                "id": r["id"],
                "language": r["language"],
                "level": r["level"],
                "year": int(year),
                "text_only": r["text_only"],
                "text": r["text"].strip(),
                "options": ocr_options,
                "ocr_has_options": bool(ocr_options),
                "label": r["label"],
                "split": split,
            })
            if len(rows) % 100 == 0:
                rate = len(rows) / (time.time() - t0)
                print(f"  {len(rows)} rows, {rate:.1f} rows/s", flush=True)

    ocr_cache_file.close()

    reused_cross_lang = 0
    if args.mode == "hybrid":
        # 1. Reuse OCR numeric options across languages
        for r in rows:
            if not args.vlm_all and not r["options"] and r["id"] in id_ocr_numeric:
                r["options"] = id_ocr_numeric[r["id"]]
                reused_cross_lang += 1

        # 2. Check existing VLM cache
        id_vlm_resolved = {}
        for r in rows:
            if args.vlm_all or not r["options"]:
                key = f"{r['id']}|{r['language']}"
                cache_key = (key, args.model)
                if cache_key in vlm_cache:
                    cached_opts = vlm_cache[cache_key]
                    if cached_opts:
                        r["options"] = cached_opts
                    if is_language_independent(cached_opts):
                        if r["id"] not in id_vlm_resolved or (not id_vlm_resolved[r["id"]] and cached_opts):
                            id_vlm_resolved[r["id"]] = cached_opts

        # 3. Find remaining rows needing VLM
        pending_rows = [
            r for r in rows
            if (args.vlm_all or not r["options"]) and (f"{r['id']}|{r['language']}", args.model) not in vlm_cache
        ]

        # 4. Group uncached rows by ID
        uncached_by_id = {}
        for r in pending_rows:
            uncached_by_id.setdefault(r["id"], []).append(r)

        first_pass_reps = []
        second_pass_pending = []

        for pid, group in uncached_by_id.items():
            if pid in id_vlm_resolved and is_language_independent(id_vlm_resolved[pid]):
                second_pass_pending.extend(group)
            else:
                rep = next((r for r in group if r["language"] == "eng"), group[0])
                first_pass_reps.append(rep)
                for r in group:
                    if r is not rep:
                        second_pass_pending.append(r)

        # 5. Process first pass (one representative per ID)
        if first_pass_reps:
            model, processor = get_vlm()
            batch_size = max(1, args.batch_size)
            total_batches = (len(first_pass_reps) + batch_size - 1) // batch_size
            for b_idx in range(total_batches):
                batch = first_pass_reps[b_idx * batch_size : (b_idx + 1) * batch_size]
                imgs = [pending_images[f"{r['id']}|{r['language']}"] for r in batch]
                results = vlm_extract_options_batch(model, processor, imgs)
                for r, vlm_opts in zip(batch, results):
                    key = f"{r['id']}|{r['language']}"
                    vlm_cache[(key, args.model)] = vlm_opts
                    if vlm_cache_file:
                        vlm_cache_file.write(
                            json.dumps({"key": key, "model": args.model, "options": vlm_opts}, ensure_ascii=False) + "\n"
                        )
                    if vlm_opts:
                        r["options"] = vlm_opts
                    id_vlm_resolved[r["id"]] = vlm_opts
                if vlm_cache_file:
                    vlm_cache_file.flush()
                print(f"  VLM batch {b_idx + 1}/{total_batches} done ({min((b_idx + 1) * batch_size, len(first_pass_reps))}/{len(first_pass_reps)} reps)", flush=True)

        # 6. Process second pass (remaining languages for same ID)
        second_pass_vlm_rows = []
        for r in second_pass_pending:
            pid = r["id"]
            cached_val = id_vlm_resolved.get(pid)
            if cached_val is not None and is_language_independent(cached_val):
                key = f"{r['id']}|{r['language']}"
                vlm_cache[(key, args.model)] = cached_val
                if vlm_cache_file:
                    vlm_cache_file.write(
                        json.dumps({"key": key, "model": args.model, "options": cached_val}, ensure_ascii=False) + "\n"
                    )
                if cached_val:
                    r["options"] = cached_val
                reused_cross_lang += 1
            else:
                second_pass_vlm_rows.append(r)

        if vlm_cache_file:
            vlm_cache_file.flush()

        if second_pass_vlm_rows:
            model, processor = get_vlm()
            batch_size = max(1, args.batch_size)
            total_batches = (len(second_pass_vlm_rows) + batch_size - 1) // batch_size
            for b_idx in range(total_batches):
                batch = second_pass_vlm_rows[b_idx * batch_size : (b_idx + 1) * batch_size]
                imgs = [pending_images[f"{r['id']}|{r['language']}"] for r in batch]
                results = vlm_extract_options_batch(model, processor, imgs)
                for r, vlm_opts in zip(batch, results):
                    key = f"{r['id']}|{r['language']}"
                    vlm_cache[(key, args.model)] = vlm_opts
                    if vlm_cache_file:
                        vlm_cache_file.write(
                            json.dumps({"key": key, "model": args.model, "options": vlm_opts}, ensure_ascii=False) + "\n"
                        )
                    if vlm_opts:
                        r["options"] = vlm_opts
                if vlm_cache_file:
                    vlm_cache_file.flush()
                print(f"  VLM pass 2 batch {b_idx + 1}/{total_batches} done ({min((b_idx + 1) * batch_size, len(second_pass_vlm_rows))}/{len(second_pass_vlm_rows)} rows)", flush=True)

        pending_images.clear()

    if vlm_cache_file:
        vlm_cache_file.close()

    dropped_train = 0
    output_keys = ("id", "language", "level", "year", "text_only", "text", "options", "label")
    for split in ("train", "val"):
        out_path = os.path.join(args.out, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split:
                    if split == "train" and args.drop_empty_train and not r["options"]:
                        dropped_train += 1
                        continue
                    f.write(json.dumps({k: r[k] for k in output_keys}, ensure_ascii=False) + "\n")

    report(
        rows,
        args.out,
        mode=args.mode,
        reused_cross_lang=reused_cross_lang,
        dropped_train=dropped_train if args.drop_empty_train else None,
    )


if __name__ == "__main__":
    main()
