#!/bin/bash
# Runs independently of any Claude session — survives disconnects, dies only if the pod dies.
cd /root/public_universal_transformer
while true; do
  sleep 600
  git add logs/runs logs/extreme-full-1.6M-d256h8-v2
  if ! git diff --cached --quiet; then
    git commit -m "logs: progress $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/push_logs_loop.out 2>&1
    git push origin main >> logs/push_logs_loop.out 2>&1
  fi
done
