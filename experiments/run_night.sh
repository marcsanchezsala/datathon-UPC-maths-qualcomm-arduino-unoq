#!/usr/bin/env bash
# Run experiments/rag_value.py for every model, on CPU, on ports 8095-8096 so
# nothing else on this laptop is disturbed. Logs to results/rag-exp/run.log.
#   bash experiments/run_night.sh
set -u
cd "$(dirname "$0")/.."
BIN=~/llama.cpp/build/bin/llama-server
OUT=results/rag-exp
export LLM_PORT=8095 EMBED_PORT=8096
mkdir -p "$OUT"

wait_up() { for _ in $(seq 120); do curl -s -m1 "127.0.0.1:$1/health" | grep -q ok && return 0; sleep 1; done; return 1; }

$BIN -m ~/models/bge-small-en-v1.5-q8_0.gguf --embedding --host 127.0.0.1 --port $EMBED_PORT --threads 2 \
  --ctx-size 512 --batch-size 512 --ubatch-size 512 --parallel 1 --cache-ram 0 > "$OUT/embed-server.log" 2>&1 &
EMBED_PID=$!
trap 'kill $EMBED_PID ${LLM_PID:-} 2>/dev/null' EXIT
wait_up $EMBED_PORT || { echo "embedding server did not start"; exit 1; }
[ -f data/index-v2.db ] || python3 -c "
import json,sys; sys.path.insert(0,'.')
from copilot import rag
P=[json.loads(l) for f in ('data/train.jsonl','data/val.jsonl') for l in open(f)]
for _ in rag.build('data/index-v2.db',[p for p in P if p['options']]): pass"

run() {  # label, gguf file, input, conditions
  echo "=== $(date +%H:%M) $1 on $3"
  $BIN -m "$2" --host 127.0.0.1 --port $LLM_PORT --threads 10 --ctx-size 4096 --parallel 1 --cache-ram 0 \
    --ctx-checkpoints 2 --jinja --reasoning off --reasoning-budget 0 > "$OUT/llm-server-$1.log" 2>&1 &
  LLM_PID=$!
  if wait_up $LLM_PORT; then
    python3 experiments/rag_value.py --model "$1" --input "$3" --conditions "$4" 2>&1 | grep -v "^\s*$"
  else
    echo "model server for $1 did not start"
  fi
  kill $LLM_PID 2>/dev/null; wait $LLM_PID 2>/dev/null; LLM_PID=
}

VAL=letter-k0,letter-k1v,value-k0,value-k1v,reason-k0,reason-ctxv
run 0.8b ~/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf data/demo/train.jsonl all
run 2b ~/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf data/demo/train.jsonl all
FT=~/models/Qwen3.5-0.8B-aqua-reason-Q8_0.gguf
[ -f $FT ] && run 0.8b-ft $FT data/demo/train.jsonl reason-k0,reason-ctxv,letter-k0
run 0.8b ~/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf data/val.jsonl $VAL
[ -f $FT ] && run 0.8b-ft $FT data/val.jsonl reason-k0,reason-ctxv
run 2b ~/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf data/val.jsonl $VAL
echo "=== $(date +%H:%M) done"
