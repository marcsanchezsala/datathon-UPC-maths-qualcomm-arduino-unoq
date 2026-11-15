#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen3.5 0.8B on the examples from prepare_aqua.py.

    python train.py --data work/aqua.jsonl --out /content/drive/MyDrive/m3-finetune

Checkpoints go to <out>/checkpoints and the run resumes from the last one, so a
Colab session that dies loses at most --save-steps steps. The adapter ends in <out>/adapter.
The loss only counts the assistant's answer: the prompt is masked out.
"""
import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from copilot.config import CONFIG  # noqa: E402

# Only the language model: the vision tower stays as it is
TARGETS = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj"
           r"|gate_proj|up_proj|down_proj)")
END = "<|im_end|>"


def render_prompt(tokenizer, messages):
    """The prompt as llama-server renders it: the chat template with thinking off."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def encode(tokenizer, example, max_len):
    """Prompt and answer are tokenized apart, as at inference, and only the answer is learned."""
    prompt = tokenizer(render_prompt(tokenizer, example["messages"][:-1]), add_special_tokens=False)["input_ids"]
    answer = tokenizer(example["messages"][-1]["content"] + END, add_special_tokens=False)["input_ids"]
    ids = (prompt + answer)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": ([-100] * len(prompt) + answer)[:max_len]}


def load(path, tokenizer, max_len):
    with open(path) as f:
        return [encode(tokenizer, json.loads(line), max_len) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="work/aqua.jsonl")
    p.add_argument("--out", default="work/run")
    p.add_argument("--base", default=CONFIG["LLM_BASE_REPO"], help="a Hugging Face repo or a local folder")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1, help="for a quick test; -1 means use --epochs")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation: batch x accum per step")
    p.add_argument("--save-steps", type=int, default=100)
    args = p.parse_args()

    revision = CONFIG["LLM_BASE_REVISION"] if args.base == CONFIG["LLM_BASE_REPO"] else None
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=revision)
    tokenizer.padding_side = "right"
    train = load(args.data, tokenizer, args.max_len)
    eval_path = args.data.replace(".jsonl", ".eval.jsonl")
    held_out = load(eval_path, tokenizer, args.max_len) if os.path.exists(eval_path) else None
    print(f"{len(train)} training examples, {len(held_out or [])} held out")

    model = AutoModelForImageTextToText.from_pretrained(args.base, revision=revision, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    checkpoints = os.path.join(args.out, "checkpoints")
    training = TrainingArguments(
        output_dir=checkpoints, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, bf16=True, logging_steps=10, save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps" if held_out else "no", eval_steps=args.save_steps,
        seed=int(CONFIG["SEED"]), report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=train, eval_dataset=held_out,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    last = get_last_checkpoint(checkpoints) if os.path.isdir(checkpoints) else None
    if last:
        print(f"Resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(os.path.join(args.out, "adapter"))
    print(f"Adapter saved in {os.path.join(args.out, 'adapter')}")


if __name__ == "__main__":
    main()
