#!/bin/bash

# GPU stats
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,power.limit --format=csv,noheader,nounits | awk -F", " '{printf "%-25s\n  Temp: %3s°C  GPU: %3s%%  VRAM: %3s%%\n  Used: %5sMB  Free: %5sMB  Total: %5sMB\n  Power: %6sW / %sW (%d%%)\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,($8/$9*100)}'

echo ""

# Active model + ctx
MODEL=$(curl -s --max-time 1 http://localhost:8082/model/active 2>/dev/null)
if [ -z "$MODEL" ]; then
  echo "  Model:  [model-api unreachable]"
else
  NAME=$(echo "$MODEL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','?'))" 2>/dev/null)
  CTX=$(echo "$MODEL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ctx','?'))" 2>/dev/null)
  echo "  Model:  $NAME  (ctx: $CTX)"
fi

echo ""

# llama-server stats (tokens/sec from last inference)
STATS=$(curl -s --max-time 1 http://localhost:8081/metrics 2>/dev/null)
if [ -z "$STATS" ]; then
  echo "  Inference: [no metrics endpoint]"
else
  echo "$STATS" | grep -E "tokens_per_second|prompt_tokens|generation_tokens" | awk '{printf "  %s\n", $0}'
fi
