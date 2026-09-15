#!/bin/zsh
# P1 full R=6 promotion runs for the screening winners (A5). Idempotent.
# Verdicts land in runs/p1_promote_verdicts.jsonl; G5 (different-day
# replication) remains PENDING and is run separately.
cd "$HOME/hone"
for cfg in spec-05b-k2 spec-05b-k3; do
  label="full-$cfg"
  if [ ! -f "runs/$label/results.json" ]; then
    .venv/bin/python scripts/measure.py --base configs/base-14b.json \
      --cand "configs/$cfg.json" --reps 6 --label "$label" \
      >> runs/p1_promote.log 2>&1 || { echo "$label FAILED" >> runs/p1_promote.log; continue; }
  fi
  .venv/bin/python scripts/verdict.py --results "runs/$label/results.json" \
    >> runs/p1_promote_verdicts.jsonl 2>> runs/p1_promote.log
  HF_HUB_OFFLINE=1 /opt/homebrew/bin/flock ~/.gpulock \
    .venv/bin/python scripts/divergence_probe.py \
    --results "runs/$label/results.json" \
    > "runs/$label/divergence_probe.json" 2>> runs/p1_promote.log
done
echo "P1 PROMOTE COMPLETE $(date '+%Y-%m-%d %H:%M:%S')" >> runs/p1_promote.log
