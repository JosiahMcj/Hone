"""A/A' calibration: run measure.py with cand == base, then compute the
paired-noise floor, baseline byte-stability, and the suggested G1/G2
margins. Output is the text of PROTOCOL Amendment A1 -- appended by hand
after review, never automatically (amendments are written, dated acts).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone.stats import (contamination_check, paired_rel_diffs,  # noqa: E402
                        sign_test, summarize)

HONE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PYTHON = os.path.join(HONE_ROOT, ".venv", "bin", "python")


def byte_stability(rows_by_arm):
    """Across ALL scored segments of both identical arms: per prompt, how
    many distinct output hashes appeared? Perfectly stable = 1 each."""
    per_prompt = {}
    for arm_rows in rows_by_arm.values():
        for key, r in arm_rows.items():
            pid = key.split("|")[0]
            per_prompt.setdefault(pid, set()).add(r["text_sha256"])
    return {pid: len(hs) for pid, hs in sorted(per_prompt.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="baseline arm config")
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--skip-measure", action="store_true",
                    help="analyze an existing runs/<label>/results.json")
    args = ap.parse_args()

    out_dir = os.path.join(HONE_ROOT, "runs", args.label)
    results_path = os.path.join(out_dir, "results.json")

    if not args.skip_measure:
        rc = subprocess.run(
            [PYTHON, os.path.join(HONE_ROOT, "scripts", "measure.py"),
             "--base", args.config, "--cand", args.config,
             "--reps", str(args.reps), "--label", args.label]).returncode
        if rc != 0:
            raise SystemExit(f"measure.py failed rc={rc}")

    with open(results_path) as f:
        res = json.load(f)
    rows = res["rows"]

    diffs, missing = paired_rel_diffs(rows["base"], rows["cand"])
    noise = summarize(diffs)
    st = sign_test(diffs)
    stability = byte_stability(rows)
    unstable = {p: n for p, n in stability.items() if n > 1}

    mad = noise["mad_rel"]
    contaminated = contamination_check(rows)
    g1 = max(1.10, 1 + 3 * mad)
    g2_allow = max(0.05, 2 * mad)

    print("\n===== A/A' calibration --", args.label, "=====")
    print(json.dumps({"noise": noise, "sign_test": st,
                      "missing_pairs": missing,
                      "byte_stability_distinct_hashes": stability}, indent=1))
    print(f"""
----- PROTOCOL Amendment A1 draft ({args.label}) -----
- A/A' paired noise (decode tps, rel): median {noise['median_rel']:+.4f},
  MAD {mad:.4f}, extremes {noise['min_rel']:+.4f}/{noise['max_rel']:+.4f},
  n={noise['n']} pairs; sign test {st['verdict']} p={st['p']}.
- Baseline byte-stability: {'STABLE -- lossless gate applicable as byte-compare' if not unstable else f'UNSTABLE on {list(unstable)} -- lossless gate needs a registered fallback BEFORE use'}.
- G1 margin (this box, this model): >= {g1:.3f}x median paired decode
  speedup (= max(1.10, 1 + 3*MAD)).
- G2 allowance: prefill tps and suite wall-clock may not regress more
  than {g2_allow:.3f} (= max(0.05, 2*MAD)) relative.
- Contamination check (A3 rule, segment-median band 15%):
  {'CLEAN -- registrable' if not contaminated else f'CONTAMINATED on {contaminated} -- INVALID, rerun, do not register'}.
-------------------------------------------------""")


if __name__ == "__main__":
    main()
