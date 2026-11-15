# Fine-tuning: teach Qwen3.5 0.8B to reason briefly

The copilot can let the model write a few lines of working before the letter
(`REASONING=1` in `config.env`). This folder trains a LoRA on AQuA-RAT so the
0.8B model writes short, useful working in exactly that format, then makes the
Q8_0 GGUF that the board runs.

Only AQuA-RAT (`deepmind/aqua_rat`, Apache 2.0) is used for training. M3Kang
never leaves the laptop: don't upload `data/` to Colab, Drive or any API.

| File | What it does |
| --- | --- |
| `prepare_aqua.py` | Filters AQuA-RAT to short worked solutions, rewrites the end to `Answer: X`, shuffles the options, writes the chat messages with `copilot.llm.build_messages` |
| `check_format.py` | Fails if the training prompt differs from the inference prompt (messages, template text, tokens) |
| `train.py` | LoRA (rank 16, alpha 32, lr 2e-4 cosine, 512 tokens), loss on the answer only, resumes from checkpoints |
| `export.py` | Merges the LoRA into the original weights, converts with llama.cpp at `LLAMA_CPP_REF` to Q8_0, prints the SHA256 |
| `colab.ipynb` | All of it, in order, on a Colab GPU |

## How reasoning mode works

`copilot/llm.py`, `answer_reasoning()`:

1. Renders the prompt with the server's chat template (`/apply-template`) and counts its tokens (`/tokenize`).
2. Computes a token budget so the whole problem fits in 60 s on the board:
   `(57 s - prompt / READ_TOKENS_PER_SECOND - one short call) x WRITE_TOKENS_PER_SECOND`.
   The budget is in tokens, so the laptop measures the same thing the board will do.
3. Lets the model write up to that budget. If the text contains `Answer: X`, that's the answer.
4. Otherwise, a second call continues the same text with `\nAnswer:` and the grammar `[A-E]`.
   The server reuses its cache, so it reads only about 4 new tokens.

If the budget is under 16 tokens (very long prompts), it answers one letter as before.
Any error or timeout still gives `FALLBACK_ANSWER`. Problems sent with an image use the one-letter path.

## Train on Colab

1. Commit and push this branch, with `finetune/`, `copilot/` and `config.env`, to GitHub.
   The notebook clones it: Colab needs the same `copilot/llm.py` to build the prompts.
2. Open `finetune/colab.ipynb` in Colab (File > Upload notebook, or from GitHub).
3. Runtime > Change runtime type > **L4** or **A100**.
4. Edit the first cell if needed: `REPO_BRANCH`, `DRIVE_DIR`, `N_EXAMPLES`, `EPOCHS`, `BATCH`, `ACCUM`.
5. Runtime > Run all. It asks for access to Google Drive.

What it does: data to `DRIVE_DIR/aqua.jsonl` (once), format check, training into
`DRIVE_DIR/run/checkpoints` (a checkpoint every 100 steps), adapter in `DRIVE_DIR/run/adapter`,
then the GGUF and its SHA256 in `DRIVE_DIR/run/`.
If the session dies, Run all again: it resumes from the last checkpoint.
The first training steps are slow while Triton compiles the kernels.

## Use the GGUF on the laptop

Download `Qwen3.5-0.8B-aqua-reason-Q8_0.gguf` from Drive into `~/models/` and check the SHA256 that the notebook printed:

```bash
sha256sum ~/models/Qwen3.5-0.8B-aqua-reason-Q8_0.gguf
~/llama.cpp/build/bin/llama-server -m ~/models/Qwen3.5-0.8B-aqua-reason-Q8_0.gguf \
  --host 127.0.0.1 --port 8092 --threads 4 --ctx-size 4096 --parallel 1 --cache-ram 0 \
  --ctx-checkpoints 2 --jinja --reasoning off --reasoning-budget 0 --alias qwen3.5-0.8b
# in another terminal, from the repository root
finetune/.venv/bin/python finetune/check_format.py --data finetune/work/aqua.jsonl --server http://127.0.0.1:8092
LLM_PORT=8092 REASONING=1 python3 eval.py --limit 0 --name ft-aqua-reason
```

`check_format.py` needs the training file; without it, run `python prepare_aqua.py` first (2 minutes).

## Use the GGUF on the board

```bash
scp ~/models/Qwen3.5-0.8B-aqua-reason-Q8_0.gguf arduino@<board-ip>:~/models/
ssh arduino@<board-ip>
sha256sum ~/models/Qwen3.5-0.8B-aqua-reason-Q8_0.gguf     # the same as in Colab
sudo sed -i 's|Qwen_Qwen3.5-0.8B-Q8_0.gguf|Qwen3.5-0.8B-aqua-reason-Q8_0.gguf|' /etc/systemd/system/math-llm.service
sudo systemctl daemon-reload && sudo systemctl restart math-llm
```

Then set `REASONING=1` in `config.env` on the board, and calibrate the two speeds there:
run the request from README step 5.5 with a prompt of about 150 tokens, and put
`prompt_per_second` in `READ_TOKENS_PER_SECOND` and `predicted_per_second` in
`WRITE_TOKENS_PER_SECOND` (round down: a lower value is safer).
Check with `python3 eval.py --limit 20 --name ft-board-check`: the worst time must stay under 60 s.
To go back, undo the `sed` (or swap the file names) and set `REASONING=0`.

## Local environment (laptop)

```bash
cd finetune
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch -r requirements.txt
```

On a 6 GB GPU, `train.py --batch 2 --accum 8` fits; a few steps are enough to test the chain.
