#!/bin/bash
# For every run under logs/runs/, runs a full by-rating/by-givens breakdown eval
# each time its latest.pt's saved step crosses a new multiple of 1000 (matching
# --ckpt-every), independent of any Claude session.
cd /root/public_universal_transformer
declare -A last

while true; do
  for RUN in logs/runs/*/; do
    RUN=${RUN%/}
    CKPT="$RUN/latest.pt"
    [ -f "$CKPT" ] || continue
    step=$(python3 -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False).get('step', -1))" 2>/dev/null)
    prev=${last[$RUN]:--1}
    if [ -n "$step" ] && [ "$step" -ge 0 ] && [ $((step % 1000)) -eq 0 ] && [ "$step" -ne "$prev" ]; then
      { echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
        python3 evaluate.py --ckpt "$CKPT" --data data/sudoku-extreme-test.csv \
          --ratings data/sudoku-extreme-test.csv --budgets 32,64,96
        echo
      } >> "$RUN/eval_breakdown.log" 2>&1
      last[$RUN]=$step
    fi
  done
  sleep 30
done
