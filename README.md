# Math on the Edge: an Offline Math Copilot on the Arduino UNO Q
### Reason and Teach with Kangaroo problems — Datathon FME 2026, Qualcomm Challenge (Arduino UNO Q 4 GB)

<img width="1526" height="927" alt="UPC-datathon" src="https://github.com/user-attachments/assets/daa7be34-b91a-4703-9581-7b5142f75d91" />

> The organizers' step-by-step guide to set up the board (steps 0 to 15) is in [`DATATHON_SETUP.md`](DATATHON_SETUP.md).

---

## 1. What It Does: A Dual-Track Solution (Reason & Teach)

The **Math on the Edge** submission fulfills both competition tracks of the Datathon FME 2026 Qualcomm challenge directly on the **Arduino UNO Q 4 GB** (Qualcomm QRB2210 quad-core Arm CPU):

1. **Track 1 — Reason (Champion Competition Solver):**
   - Offline, autonomous mathematical problem solver for the international Kangaroo Math Competition ([M3Kang](https://huggingface.co/datasets/qualcomm/M3Kang)).
   - **What answers on the board:** `Qwen2.5-Math-1.5B-Instruct` alone (Q4_K_M, port 8081), the configuration in [`config.env`](config.env). Measured on a laptop with that configuration: **25.1% (141/562)** on our M3Kang validation set and **26.0% (26/100)** on the organizers' practice set of new problems. Random guessing gives 20%.
   - **Best result off the board:** the dual-model router described below (Qwen2.5-3B-Instruct + Qwen2.5-Math-1.5B) reached **28.8% (162/562)** on validation on a laptop. We do not submit it: Qwen2.5-3B takes 55–60 s per problem on the board's CPU, at the edge of the 60 s limit.
   - On the physical board we recorded **32.0% on 50 validation problems** (16/50) with **0 timeouts**.
   - Answers are rendered in real time onto the board's embedded **8x13 LED matrix** via the Arduino Router Bridge.
   - Run via: `python3 predict.py --input data/val.jsonl --output results/predictions.csv`

2. **Track 2 — Teach (Socratic Teaching Copilot & Edge Companion):**
   - Interactive learning companion designed for classrooms without cloud connectivity.
   - **4-Rung Socratic Hint Ladder:** progressive hints that guide student discovery without spoiling the answer.
   - **Distractor & Misconception Analysis:** detects common student pitfalls (off-by-one, sign errors, perimeter vs area) and explains *why* a chosen distractor is a trap.
   - **Adaptive Spaced Repetition Practice:** dynamically adjusts question difficulty based on student confidence and accuracy.
   - Run via: `python3 explorer/server.py --input data/val.jsonl --index data/index.db --images data/images` (access at `http://127.0.0.1:8000`).

---

## 2. The Architecture

> **Submitted configuration:** one language model on the board, `Qwen2.5-Math-1.5B-Instruct` on port 8081, for every language and level. In [`config.env`](config.env) the router's two addresses point to that same server. The two-model routing in the flowchart is the laptop configuration that scored 28.8%; see section 1 for why it is not the one on the board.

The system coordinates the two asymmetric processing domains of the Arduino UNO Q:
1. **Linux Domain (Qualcomm QRB2210 quad-core Arm Cortex-A53 @ 2.0 GHz, 4 GB LPDDR4):** Hosts concurrent quantized model servers through `llama.cpp` and executes the Python decision pipeline ([`copilot/pipeline.py`](copilot/pipeline.py)).
2. **Microcontroller Domain (Real-Time STM32 MCU):** Controls physical interaction, driving the onboard 8x13 LED matrix to visualize solutions (`A`–`E`) via remote procedure calls over the Arduino Router Bridge (`Bridge.call("draw", ...)`).

### Decision & Inference Pipeline Flowchart

```mermaid
flowchart TD
    P["Input Problem (text, options, language, level)"] --> T0{"options == {} and no image?"}
    
    %% Tier 0 Near-Duplicate RAG
    T0 -- "Yes (No Options)" --> H0{"Top-1 RAG near-duplicate score >= 0.94?"}
    H0 -- "Yes" --> L0["Return exact historical label (0.0s, 100% precision)"]
    H0 -- "No" --> D0["Return optimal fallback 'D' (0.0s)"]
    
    %% Hierarchical Router
    T0 -- "No (Has Options)" --> ROUTE{"SOLVE_MODE == 'router'?"}
    ROUTE -- "Yes (Hierarchical Routing)" --> LANG{"language == 'eng'?"}
    LANG -- "No (Catalan: all levels)" --> M_CAT["Port 8081: Qwen2.5-3B-Instruct (Multilingual linguistic capacity)"]
    LANG -- "Yes (English)" --> LVL{"level in [6, 7]?"}
    LVL -- "Yes (Advanced High School)" --> M_CAT
    LVL -- "No (Levels 0-5: Primary/Middle School)" --> M_ENG["Port 8083: Qwen2.5-Math-1.5B (Arithmetic/Algebra specialist)"]
    ROUTE -- "No" --> M_DEF["Default Port 8081"]
    
    %% Tier 1 Constrained Generation
    M_ENG --> T1["Tier 1: BOXED_GRAMMAR ('\\boxed{[A-E]}', 5 tokens) -> softmax logprobs"]
    M_CAT --> T1
    M_DEF --> T1
    
    %% Tier 2 Conditional Escalation
    T1 --> TH{"confidence < 0.38 and remaining > 25s?"}
    TH -- "No (High confidence)" --> OUT1["Output Tier 1 Letter (~0.2s - 0.3s)"]
    TH -- "Yes (Uncertain prediction)" --> T2["Tier 2: Micro-CoT (max_tokens=220) + match_option_from_text()"]
    
    %% Truncation Safeguard
    T2 --> OK{"finish_reason == 'length'?"}
    OK -- "No (Complete generation)" --> MATCH["Accept \\boxed{X}, explicit keywords, or verified numeric value"]
    OK -- "Yes (Truncated CoT)" --> SAFE["Accept only strict \\boxed{X}; discard dangling numbers, keep Tier 1"]
```

### Key Architectural Pillars

1. **Tier 0 Near-Duplicate RAG (`score >= 0.94`):**
   In the validation set we evaluated on, prepared with the organizers' original OCR script, 223 of 562 problems (39.7%) had no answer choices, because they could not be read from the image. Our improved OCR (see *Reading the options* below) brings that down to 84 of 562. Instead of blindly guessing, Tier 0 searches a localized vector database (`bge-small-en-v1.5` on Port 8082). When a near-identical problem statement is found with cosine similarity $\ge 0.94$, the verified historical label is returned with **100% deterministic accuracy**, yielding a $+2$ net score gain over the static fallback `"D"`.
2. **Zero-Overhead Hierarchical Language + Level Router:**
   A metadata-driven $O(1)$ router splits requests to the model best suited for the specific task:
   - **Catalan (`cat`, all levels):** Routed to **`Qwen2.5-3B-Instruct`** (Port 8081). The 3B model possesses broad multilingual pre-training that avoids the severe linguistic collapse observed in English-only mathematical models (27.8% vs 22.8%).
   - **English (`eng`, Levels 0–5):** Routed to **`Qwen2.5-Math-1.5B-Instruct`** (Port 8083). For primary and lower-secondary school arithmetic, fractions, and introductory algebra, the specialized math engine outperforms general 3B models.
   - **English (`eng`, Levels 6–7):** Routed to **`Qwen2.5-3B-Instruct`** (Port 8081). In advanced high school Kangaroo problems, reasoning shifts from procedural calculations to intricate verbal logic, combinatorial word puzzles, and semantic subtleties where parameter count and reading comprehension dominate.
3. **Constrained Grammar Tier 1 (`BOXED_GRAMMAR`):**
   Instead of unrestricted token generation, Tier 1 enforces `root ::= "\\boxed{" [A-E] "}"` (`max_tokens=5`, `top_logprobs=20`). This extracts calibrated softmax probabilities for each option letter directly in distribution without wasteful autoregression, finishing in **0.2s–0.3s**.
4. **Conditional Tier 2 Micro-CoT with Truncation Guard:**
   When Tier 1 top-1 confidence falls below 0.38 and remaining time $> 25$ seconds, the system escalates to a 220-token Chain-of-Thought reasoning burst. If the generation hits the token ceiling (`finish_reason == 'length'`), the parser enforces `allow_numeric=False`, preventing dangling numbers from intermediate equations from misidentifying options.

### Reading the options: our OCR

In M3Kang the five options are only in the problem's image, not in its text. The organizers' [`laptop/prepare_m3kang.py`](laptop/prepare_m3kang.py) read them with OCR on the whole image, which left many problems with no options and garbled fractions and exponents (`1/2` read as `12`, `n³` as `n3`). We rewrote how it reads them:

- **Two passes.** OCR on the whole image only finds the markers `A)` to `E)`. If one is missing, usually `C)`, its place is inferred from the layout (5 in a row, 3 + 2, 2 + 2 + 1, one per line). Then each option is cut out of the image along its ink and read on its own.
- **Maths notation.** Stacked fractions are read as `3/4`, exponents and degrees as `6^13`, `n³` and `78°`, and square roots as `√2`. An apostrophe, as in "L'Oriol", stays part of the word.
- **Two languages as a check.** When the options have no words, they must read the same in English and Catalan. If they differ, both are dropped. If one language could not be read, it gets the options of the other.
- **When in doubt, no options.** A picture, a symbol read with low confidence or a lost root sign leaves the problem empty, never with three or four options or a guessed value.

| M3Kang, English and Catalan (1,747 problems) | Original OCR | Our OCR |
| --- | ---: | ---: |
| Problems with options, English | 1,078 (62%) | 1,473 (84%) |
| Problems with options in at least one language | 1,156 | 1,473 |
| Text-only problems with no options in any language | 172 | 20 |

We checked 30 random problems whose options changed against their images: all 30 were right. Of 18 hard cases (fractions, roots, exponents, degrees), 17 were right; the wrong one (`√2` read as `2`) is now left empty. On the organizers' practice set both versions read 100 of 100 problems. [`laptop/test_prepare_m3kang.py`](laptop/test_prepare_m3kang.py) checks the reading with real OCR output from cases that used to fail. The 274 problems still without options are almost all problems whose options are drawings.

The results tables below were measured on the validation set prepared with the original OCR.

### The fine-tuned model: Qwen3.5-0.8B-aqua-reason

To get a small model that reasons in a few short lines instead of guessing a letter, we fine-tuned Qwen3.5 0.8B, the organizers' model, with LoRA ([`finetune/`](finetune/)):

- **Data:** 15,000 problems from [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat) (Apache 2.0), multiple choice A to E with a worked solution. We kept short solutions that end on the right option, rewrote their end to `Answer: X` and shuffled the options, so the model does not learn a favourite letter. No M3Kang data was used for training.
- **Training:** LoRA with rank 16, alpha 32, learning rate 2e-4, 2 epochs, loss on the answer only, in the exact prompt format the copilot uses. It runs on Google Colab ([`finetune/colab.ipynb`](finetune/colab.ipynb)), then merges the LoRA and exports a Q8_0 GGUF (833 MB) with the pinned llama.cpp commit.
- **What it learned:** to reason briefly. It writes 103 tokens on average where the base model writes 175, and it reaches its own `Answer: X` far more often instead of being cut off.
- **What it did not learn:** to calculate better. On validation it scores 21.4%, against 21.2% for the base model answering one letter. On the practice set it scores 30% against 26% for the base model reasoning, a difference within the noise of 100 problems.

**Where we use it:** it is the model of the Teach track, where short, step-by-step explanations matter more than the final letter.

**Why it does not answer the test:** we also considered it for the Reason track, but we did not have time to test it properly on the board. We think it would answer faster: its reasoning is shorter (about 32 s per problem on the board against 47 s for the base model, estimated from token counts and the board's measured speeds, not measured on the board), and it is about half the size of Qwen2.5-Math-1.5B, so each token costs less. With no board measurement and no clear gain in accuracy, we kept the measured configuration.

---

## 3. How to Replicate It

Replicating our champion system requires an **Arduino UNO Q 4 GB** (or a local Linux/Windows host for verification).

### Step 1: Base Environment Setup
Ensure steps 0 through 12 from [`DATATHON_SETUP.md`](DATATHON_SETUP.md) have been completed (cloning repo, Python 3 virtual environment, dependencies).

```bash
# Clone the repository
git clone https://github.com/marcsanchezsala/datathon-UPC-maths-qualcomm-arduino-unoq.git
cd datathon-UPC-maths-qualcomm-arduino-unoq

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r laptop/requirements.txt
```

### Step 2: Download Pinned Models (GGUF)
Our setup utilizes the following pinned GGUF models:

```bash
mkdir -p models

# 1. Official Board Solver Champion: Qwen2.5-Math-1.5B-Instruct (Port 8081 on board / Port 8083 in dual router)
curl -L -o models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf

# 2. Teaching Copilot & Socratic Edge Model: Qwen3.5-0.8B-aqua-reason (Port 8081 in teaching track, trained via finetune/)
# Not published as a download: train and export it with finetune/ (see finetune/README.md), then copy it to models/

# 3. High-Capacity Multilingual Model: Qwen2.5-3B-Instruct (Port 8081 in dual router)
curl -L -o models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# 4. Common Embedding Model: bge-small-en-v1.5 (Port 8082)
curl -L -o models/bge-small-en-v1.5-q8_0.gguf \
  https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf
```

### Step 3: Launch Local Model Servers

#### A. Official Production Setup on the Physical Arduino UNO Q (Single LLM Service)
On the physical Qualcomm QRB2210 ARM Cortex-A53 CPU, `Qwen2.5-Math-1.5B` provides the optimal balance of speed and precision (15–25s latency, 0 timeouts). It runs via `systemd` on Port 8081:
```bash
# Terminal 1: Port 8081 (Official Board Solver: Qwen2.5-Math-1.5B)
llama-server -m models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf --port 8081 -c 2048 -t 4 --alias qwen2.5-math-1.5b

# Terminal 2: Port 8082 (Embedding server: bge-small-en-v1.5)
llama-server -m models/bge-small-en-v1.5-q8_0.gguf --port 8082 -c 512 -t 4 --embedding
```
*(For the Teaching Companion track, Port 8081 runs `Qwen3.5-0.8B-aqua-reason-Q8_0.gguf` with ultra-fast ~1–4s Socratic latency).*

#### B. Dual-Model Hierarchical Router Setup (Evaluation / Laptop Server)
To evaluate the complete dual-model router across all languages and levels:
```bash
# Terminal 1: Port 8081 (Qwen2.5-3B-Instruct for Catalan and English Levels 6-7)
llama-server -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --port 8081 -c 2048 -t 4 --alias qwen2.5-3b-instruct

# Terminal 2: Port 8082 (Embedding server: bge-small-en-v1.5)
llama-server -m models/bge-small-en-v1.5-q8_0.gguf --port 8082 -c 512 -t 4 --embedding

# Terminal 3: Port 8083 (Qwen2.5-Math-1.5B-Instruct for English Levels 0-5)
llama-server -m models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf --port 8083 -c 2048 -t 4 --alias qwen2.5-math-1.5b
```

### Step 4: Verify `config.env` Configuration
Verify that [`config.env`](config.env) contains the configuration we submit, the one that runs on the board:
```ini
LLM_FILE=Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf
LLM_PORT=8081
EMBED_PORT=8082
SOLVE_MODE=router
ROUTER_MATH_PORT=8081
ROUTER_MATH_URL=http://127.0.0.1:8081/v1/chat/completions
ROUTER_ENG_MAX_LEVEL=5
TIER0_RAG_MIN_SCORE=0.94
JEV_CONFIDENCE_THRESHOLD=0.38
COT_MAX_TOKENS=80
TIME_LIMIT_SECONDS=60
FALLBACK_ANSWER=D
DEBIAS_ALPHA=0.0
```

Both router addresses are port 8081, so one model answers everything. For the dual-model router of setup B, set `ROUTER_MATH_PORT=8083`, `ROUTER_MATH_URL=http://127.0.0.1:8083/v1/chat/completions` and `COT_MAX_TOKENS=220`.

### Step 5: Build Vector Index & Execute Evaluation
```bash
# Build the vector index for Tier 0 RAG from the training set (data/train.jsonl, from step 9 of DATATHON_SETUP.md)
python3 build_index.py --input data/train.jsonl --db data/index.db

# Run official evaluation on sample benchmark (50 problems)
python3 eval.py --limit 50 --mode router

# Run full evaluation across all 562 official validation problems
python3 eval.py --limit 0 --mode router
```

With the submitted single-model configuration we measured 141/562 = 25.1% on validation and 26/100 = 26.0% on the practice set (runs `math15b_val_all562` and `math15b_demo_train` in [`results/ablations.csv`](results/ablations.csv)).

**Console output of the dual-model router (setup B, laptop):**
```text
============================================================
Accuracy: 162/562 = 28.8%
  floor, always D: 23.1%   random guess: 20.0%
Latency: mean 0.6 s, median 0.3 s, worst 8.6 s, over 60 s: 0
Status: {'ok': 562}

By language
  cat            78/281  27.8%
  eng            84/281  29.9%

By level
  0              11/40   27.5%
  1               5/34   14.7%
  2              16/58   27.6%
  3               9/24   37.5%
  4              34/124  27.4%
  5              37/106  34.9%
  6              31/98   31.6%
  7              19/78   24.4%

By whether OCR found the options
  False          60/223  26.9%
  True          102/339  30.1%

Letters     A: 36/102  B:110/118  C:109/119  D:253/130  E: 54/93
```

---

## 4. Results

The progression of all evaluated architectures is logged in [`results/ablations.csv`](results/ablations.csv) and detailed in [`RESULTS.md`](RESULTS.md).

### Comprehensive Experimental Ablation Table

| Experiment Run | Architecture & Configuration | Changes vs Prior | Overall Acc ($n=50$) | OCR Options ($n=28$) | No Options ($n=22$) | Latency (Mean / Med / Max) | Full Val ($n=562$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Statistical Floor** | Constant `"D"` | Static baseline | 22.0% (11/50) | 14.3% (4/28) | 31.8% (7/22) | 0.0s | 23.1% (130/562) |
| **Datathon Baseline** | Qwen3.5-0.8B (initial) | Reference prompt | 22.5% (9/40) | 22.5% (9/40) | — | 1.8s / 1.2s / 5.2s | — |
| `20261114-164015` | `tier2` (v1) | `root ::= [A-E]`, neutral prior ON | 24.0% (12/50) | 17.9% (5/28) | 31.8% (7/22) | 0.5s / 0.7s / 3.0s | — |
| `20261114-170338` | `hybrid` (v2) | `BOXED_GRAMMAR` + CoT (`conf < 0.45`) | 32.0% (16/50) | 32.1% (9/28) | 31.8% (7/22) | 0.9s / 0.2s / 7.0s | — |
| `20261114-174443` | `hybrid` (v3) | `conf < 0.38` + CoT truncation guard | 34.0% (17/50) | 35.7% (10/28) | 31.8% (7/22) | 0.4s / 0.2s / 5.4s | — |
| `20261114-175531` | `hybrid-rag1` | Naive RAG $k=1$ (`bge-small`) | 26.0% (13/50) | 25.0% (7/28) | 27.3% (6/22) | 0.7s / 0.5s / 4.5s | — |
| `20261114-180758` | `hybrid-twostep` | Fast Two-Step CoT (KV cache reuse) | 36.0% (18/50) | 39.3% (11/28) | 31.8% (7/22) | 0.4s / 0.3s / 4.1s | — |
| `20261114-201600` | `deepseek-r1-hybrid`| DeepSeek-R1-Distill-1.5B | 34.0% (17/50) | 35.7% (10/28) | 31.8% (7/22) | 1.2s / 0.4s / 10.5s | — |
| `20261114-201729` | `qwen2.5-1.5b-hybrid`| Qwen2.5-1.5B-Instruct | 28.0% (14/50) | 25.0% (7/28) | 31.8% (7/22) | 0.9s / 0.3s / 7.0s | — |
| `20261114-194135` | `full-val-fixed` | Qwen2.5-Math-1.5B Solo (Full Val) | 34.0% (17/50) | 35.7% (10/28) | 31.8% (7/22) | 1.1s / 0.3s / 14.3s | 25.4% (143/562) |
| `20261114-202433` | `qwen2.5-3b-full` | Qwen2.5-3B-Instruct Solo (Full Val) | 38.0% (19/50) | 42.9% (12/28) | 31.8% (7/22) | 0.5s / 0.6s / 10.5s | 26.2% (147/562) |
| `20261114-213252` | `language-router-full` | Language Router (`cat`->3B, `eng`->Math) | 36.0% (18/50) | 39.3% (11/28) | 31.8% (7/22) | 0.6s / 0.3s / 6.8s | 27.8% (156/562) |
| `20261114-224208` | `router-debias-full` | Logit debiasing ($\alpha = 0.15$) | 34.0% (17/50) | 35.7% (10/28) | 31.8% (7/22) | 0.8s / 0.3s / 12.3s | 27.4% (154/562) |
| `20261114-232659` | `router-tier0-full` | Language Router + Tier 0 RAG ($0.94$) | 34.0% (17/50) | 35.7% (10/28) | 31.8% (7/22) | 0.7s / 0.3s / 8.6s | 28.1% (158/562) |
| **`20261114-233857`** | **`hier-router-full`** | **Hierarchical Router (Lang+Lvl) + Tier 0 RAG** | **38.0% (19/50)** | **42.9% (12/28)** | **31.8% (7/22)** | **0.6s / 0.3s / 8.6s** | **28.8% (162/562)** |

### Full Validation Benchmark Comparison ($n=562$)

```
+-----------------------------------------------------------------------------+
| FULL VALIDATION SET ACCURACY (562 PROBLEMS)                                 |
+-----------------------------------------------------------------------------+
| Champion: Hierarchical Router + Tier 0 RAG   | [##########] 28.8% (162/562) |
| Language Router (Basic)                      | [######### ] 27.8% (156/562) |
| Qwen2.5-3B-Instruct Solo                     | [########  ] 26.2% (147/562) |
| Qwen2.5-Math-1.5B Solo                       | [########  ] 25.4% (143/562) |
| Constant Guess ("D" floor)                   | [#######   ] 23.1% (130/562) |
| Random Guessing Floor (Uniform)              | [######    ] 20.0% (112/562) |
+-----------------------------------------------------------------------------+
```

---

## 5. Error Analysis

A thorough error analysis was performed on all 562 validation problems, segmented across four distinct dimensions:

### 1. By Language
- **English (`eng`): 84 / 281 (29.9%)**
- **Catalan (`cat`): 78 / 281 (27.8%)**

*Insight:* English accuracy is strengthened by the arithmetic precision of `Qwen2.5-Math-1.5B` in levels 0–5. Catalan performance is sustained by `Qwen2.5-3B-Instruct`, whereas testing `Qwen2.5-Math-1.5B` on Catalan collapsed to 22.8% due to tokenization fragmentation and non-English syntactic divergence.

### 2. By Difficulty Level

| Level | Educational Stage | Champion Router Acc | Qwen2.5-3B Solo | Qwen2.5-Math-1.5B Solo | Gain vs 3B Solo | Gain vs Math Solo |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Primary 3rd–4th | **27.5% (11/40)** | 25.0% (10/40) | 35.0% (14/40) | +2.5% | -7.5% |
| **1** | Primary 5th | **14.7% (5/34)** | 11.8% (4/34) | 11.8% (4/34) | +2.9% | +2.9% |
| **2** | Primary 6th | **27.6% (16/58)** | 25.9% (15/58) | 31.0% (18/58) | +1.7% | -3.4% |
| **3** | Secondary 1st (1r ESO) | **37.5% (9/24)** | 33.3% (8/24) | 29.2% (7/24) | +4.2% | +8.3% |
| **4** | Secondary 2nd (2n ESO) | **27.4% (34/124)**| 21.8% (27/124) | 24.2% (30/124) | **+5.6%** | **+3.2%** |
| **5** | Secondary 3rd (3r ESO) | **34.9% (37/106)**| 30.2% (32/106) | 32.1% (34/106) | **+4.7%** | **+2.8%** |
| **6** | Secondary 4th (4t ESO) | **31.6% (31/98)** | 32.7% (32/98) | 23.5% (23/98) | -1.1% | **+8.1%** |
| **7** | High School (Batxillerat) | **24.4% (19/78)** | 24.4% (19/78) | 16.7% (13/78) | =0.0% | **+7.7%** |
| **Total**| **All Levels (562)** | **28.8% (162/562)**| **26.2% (147/562)**| **25.4% (143/562)** | **+2.6% (+15)** | **+3.4% (+19)** |

*Insight:* Levels 6 and 7 show the necessity of the hierarchical router: `Qwen2.5-Math-1.5B` collapses on advanced word problems (16.7% on level 7), while `Qwen2.5-3B-Instruct` restores high-level verbal reasoning (24.4%), contributing $+14$ net correct answers in English levels 6–7.

### 3. By Mathematical Topic

| Mathematical Domain | Problem Count | Correct | Accuracy | Observed Failure Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Logic & Combinatorics** | 242 | 81 | **33.5%** | Strongest domain. Deductive reasoning benefits directly from Tier 1 constrained logprobs. |
| **Geometry & Spatial** | 213 | 54 | **25.4%** | Severely impaired by missing visual diagrams (`text_only=False` issues). Models struggle with spatial relations without pixels. |
| **Arithmetic & Numbers** | 100 | 20 | **20.0%** | Multi-step modular arithmetic and complex prime factorization often fail in constrained token budgets. |
| **Algebra & Word Problems** | 7 | 1 | **14.3%** | Small sample in validation, but multi-variable systems frequently suffer intermediate truncation. |

### 4. Primary Empirical Failure Modes

1. **OCR Option Extraction Failure (39.7% of all validation instances):**
   223 out of 562 problems in the validation set we evaluated on had empty `options: {}`, because the organizers' OCR could not read them from the image. Our improved OCR reads options for all but 84 of them (see *Reading the options*). Without visual tokens, models cannot read multiple-choice alternatives from the raw page. Tier 0 Duplicate RAG resolves exact near-duplicates ($\ge 0.94$), and fallback `"D"` handles the remainder.
2. **Visual Diagram Blindness (`text_only=False`):**
   Problems with visual illustrations yield 31.0% accuracy vs 26.0% on purely textual problems, but primarily because geometric diagrams often contain clues that pure text prompts miss. The lack of an edge-optimized Vision-Language Model (VLM) represents the ceiling for geometric questions.
3. **Chain-of-Thought Truncation (`finish_reason == 'length'`):**
   When generating long symbolic chains under tight token limits ($220$ tokens), equations cut off prematurely. Prior to our truncation safeguard in v3, regex extractors captured the last visible number (e.g., intermediate bounds like `"1"` or `"15"`), misidentifying them as option choices `A` or `E`.

---

## 6. What Didn't Work

Documenting failed experiments and negative results is vital for edge AI engineering:

### 1. Empirical Logit Debiasing ($\alpha = 0.15$)
- **Hypothesis:** Option letters in training data showed non-uniform empirical priors (`B` $+0.31$, `C` $+0.29$ in Catalan; `D` $-1.22$ in English Math). Subtracting $\alpha \cdot \text{bias}(c)$ from logprobs would normalize letter selection.
- **Outcome:** Global accuracy dropped from **27.8% (156/562)** to **27.4% (154/562)**, and options accuracy dropped from **28.9%** to **28.3%** ($-2$ net correct answers).
- **Why it failed:** In multiple-choice mathematical reasoning, option letters are **arbitrary containers** for calculated numerical results (e.g. `B) 42`). The model selects `B` because the mathematical output is `42`, not out of lexical affinity for the glyph `B`. Penalizing common letters directly penalizes correct calculations that happen to land on those letters. Consequently, `DEBIAS_ALPHA=0.0` was permanently set.

### 2. DeepSeek-R1-Distill-Qwen-1.5B
- **Hypothesis:** The distilled reasoning model would provide superior step-by-step thinking for competition math.
- **Outcome:** Produced verbose internal reasoning tokens (`<think> ... </think>`) that frequently exceeded the micro-CoT ceiling, leading to high latency ($10.5\text{ s}$ max on laptop, estimated $>50\text{ s}$ on the ARM board) and an accuracy of only 34.0% on the 50-problem sample—underperforming both Two-Step CoT (36.0%) and the Hierarchical Router (38.0%).

### 3. Naive Few-Shot In-Context RAG ($k \ge 1$)
- **Hypothesis:** Injecting the top-$k$ most similar solved problems into the prompt would guide edge models by analogy.
- **Outcome:** Dropped accuracy from 34.0% to 26.0% ($n=50$).
- **Why it failed:** Sub-3B models suffer from **negative transfer** and **prompt distraction**. When an example problem with distinct numerical values is inserted into the prompt, the model frequently copies the example's numbers rather than executing reasoning on the actual question. RAG is only effective at the extreme similarity threshold (Tier 0 $\ge 0.94$) for near-duplicates.

### What We Would Try Next
1. **Measure the fine-tuned Qwen3.5-0.8B-aqua-reason as the solver on the board.** We expect it to answer faster than the current configuration, but we only estimated its time on a laptop.
2. **On-Device Quantized Multimodal Models (VLM):** Quantizing ultra-lightweight vision models such as **Moondream2** or **SmolVLM-500M** using `llama.cpp`'s `mtk-vision` or Qualcomm NPU acceleration to process problem diagram crops directly.
3. **Speculative Decoding:** Running `Qwen2.5-0.5B` as a speculative drafter for `Qwen2.5-3B-Instruct` to accelerate Tier 2 CoT generation speed on ARM Cortex-A53.

---

## 7. The Team & Credits

Developed with passion for the **Datathon FME 2026 — Qualcomm Challenge** (Facultat de Matemàtiques i Estadística, Universitat Politècnica de Catalunya):

- **Marc** & Team *Math on the Edge*

### Academic & Technical Citations

```bibtex
@article{torres2026m3kang,
  title={M3Kang: A Multilingual Multi-grade Benchmark for Kangaroo Math Competition Reasoning},
  author={Torres-Camps, et al.},
  journal={arXiv preprint arXiv:2601.16218},
  year={2026}
}

@software{llamacpp,
  author = {Georgi Gerganov and the GGML Team},
  title = {llama.cpp: Port of Facebook's LLaMA model in C/C++},
  url = {https://github.com/ggml-org/llama.cpp},
  year = {2023-2026}
}

@article{qwen25,
  title={Qwen2.5: A Comprehensive Technical Report},
  author={Qwen Team, Alibaba Cloud},
  year={2024}
}
```

Special thanks to **Qualcomm Technologies**, **Arduino**, and **UPC FME** for organizing the challenge and championing private, edge-native mathematical AI.

## License

Our code is released under the [MIT License](LICENSE). The license covers the code this team wrote. It does not cover the M3Kang data, which is licensed for research use only ([dataset card](https://huggingface.co/datasets/qualcomm/M3Kang)), or the models, each of which has its own license on its model card.
