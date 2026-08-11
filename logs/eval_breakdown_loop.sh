#!/bin/bash
# Runs a full by-rating/by-givens breakdown eval every 1000 training steps
# (matching --ckpt-every), independent of any Claude session.
cd /root/public_universal_transformer
RUN=logs/runs/extreme-full-3.8M-d96h4
CKPT="$RUN/latest.pt"
OUT="$RUN/eval_breakdown.log"
last=-1

while true; do
  if [ -f "$CKPT" ]; then
    step=$(python3 -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False).get('step', -1))" 2>/dev/null)
    if [ -n "$step" ] && [ "$step" -ge 0 ] && [ $((step % 1000)) -eq 0 ] && [ "$step" -ne "$last" ]; then
      { echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
        python3 evaluate.py --ckpt "$CKPT" --data data/sudoku-extreme-test.csv \
          --ratings data/sudoku-extreme-test.csv --budgets 32,64,96
        echo
      } >> "$OUT" 2>&1
      last=$step
    fi
  fi
  sleep 30
done
