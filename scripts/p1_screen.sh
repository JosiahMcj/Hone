#!/bin/zsh
# P1 wave-1 screening (PROTOCOL A4 law: screening is NON-PROMOTABLE --
# it only selects candidates for full R=6 promotion runs). Sequential
# paired R=2 runs; every timed segment respects gpulock + guards via
# measure.py. Idempotent: completed labels are skipped on relaunch.
cd "$HOME/hone"
for cfg in spec-05b-k2 spec-05b-k3 spec-05b-k4 spec-05b-k6 spec-05b-k8 \
           spec-3b-k2 spec-3b-k4 spec-3b-k6; do
  label="screen-$cfg"
  if [ ! -f "runs/$label/results.json" ]; then
    .venv/bin/python scripts/measure.py --base configs/base-14b.json \
      --cand "configs/$cfg.json" --reps 2 --label "$label" \
      >> runs/p1_screen.log 2>&1 || { echo "$label FAILED" >> runs/p1_screen.log; continue; }
  fi
  .venv/bin/python scripts/verdict.py --results "runs/$label/results.json" \
    --screen >> runs/p1_screen_verdicts.jsonl 2>> runs/p1_screen.log
done
echo "P1 SCREEN COMPLETE $(date '+%Y-%m-%d %H:%M:%S')" >> runs/p1_screen.log
