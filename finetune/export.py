#!/usr/bin/env python3
"""Merge the LoRA adapter into the original weights and make the Q8_0 GGUF the board runs.

    python export.py --adapter work/run/adapter --out work/run/qwen3.5-0.8b-ft-Q8_0.gguf

The merged model keeps the exact file layout of the original (vision tower and MTP head
included): only the tensors LoRA changed are replaced. Then convert_hf_to_gguf.py from
llama.cpp at LLAMA_CPP_REF (the board's commit) writes it as Q8_0.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from copilot.config import CONFIG  # noqa: E402


def llama_cpp(path):
    """A checkout of llama.cpp at the pinned commit. Only its Python converter is used."""
    ref = CONFIG["LLAMA_CPP_REF"]
    if not os.path.isdir(path):
        os.makedirs(path)
        for cmd in (["init", "-q"], ["remote", "add", "origin", CONFIG["LLAMA_CPP_REPO"]],
                    ["fetch", "-q", "--depth", "1", "origin", ref], ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", path, *cmd], check=True)
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    if head != ref:
        sys.exit(f"{path} is at {head}, not at LLAMA_CPP_REF {ref}")
    return path


def merge(base_dir, adapter, merged_dir):
    model = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.float32)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload().state_dict()
    os.makedirs(merged_dir, exist_ok=True)
    changed = 0
    for name in os.listdir(base_dir):
        source = os.path.join(base_dir, name)
        if name.endswith(".safetensors"):
            tensors = load_file(source)
            for key, original in tensors.items():
                if key in merged and not torch.equal(merged[key].to(original.dtype), original):
                    tensors[key] = merged[key].to(original.dtype).contiguous()
                    changed += 1
            save_file(tensors, os.path.join(merged_dir, name), metadata={"format": "pt"})
        elif os.path.isfile(source):
            shutil.copy(source, merged_dir)
    if changed == 0:
        sys.exit("The adapter changed no tensor: check the adapter and the tensor names")
    print(f"Merged: {changed} tensors changed, saved in {merged_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True, help="the .gguf file to write")
    p.add_argument("--base", default="", help="a local copy of the original model; downloaded if empty")
    p.add_argument("--llama-cpp", default=os.path.expanduser("~/llama.cpp-convert"))
    p.add_argument("--work", default="", help="where the merged model goes; next to --out if empty")
    args = p.parse_args()

    base = args.base or snapshot_download(CONFIG["LLM_BASE_REPO"], revision=CONFIG["LLM_BASE_REVISION"])
    merged_dir = args.work or os.path.splitext(args.out)[0] + "-merged"
    merge(base, args.adapter, merged_dir)
    converter = os.path.join(llama_cpp(args.llama_cpp), "convert_hf_to_gguf.py")
    subprocess.run([sys.executable, converter, merged_dir, "--outtype", "q8_0", "--outfile", args.out],
                   check=True)
    digest = hashlib.sha256()
    with open(args.out, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    print(f"\n{args.out}\n{os.path.getsize(args.out) / 1e6:.0f} MB\nSHA256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
