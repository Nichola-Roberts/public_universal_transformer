#!/bin/bash
# Runs independently of any Claude session — survives disconnects, dies only if the pod dies.
cd /root/public_universal_transformer
while true; do
  sleep 600
  git add logs/runs/extreme-full-3.8M-d96h4
  if ! git diff --cached --quiet; then
    git commit -m "logs: extreme-full-3.8M-d96h4 progress $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/push_logs_loop.out 2>&1
    git push origin main >> logs/push_logs_loop.out 2>&1
  fi
done
