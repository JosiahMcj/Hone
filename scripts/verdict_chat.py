"""Mechanical verdict for T3-A (prompt-cache reuse). PRIMARY metric =
prefill_s (time-to-first-token); LOWER is better, so a win is a NEGATIVE
paired relative diff. Decode tps is a side-check (caching must not move
it). Contamination sentinel = decode tps (homogeneous across items,
unlike prefill_s), reusing the A5 per-arm segment check.

Usage:
  verdict_chat.py --results runs/<label>/results.json [--aa]
--aa: frame the output as an A/A' noise floor + suggested margins.
Otherwise: apply G1 (prefill reduction), G3 (byte parity), G4
(contamination) against configs/margins.json.
"""
import argparse
import json
import os
import sys
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone.stats import (contamination_check, paired_rel_diffs,  # noqa: E402
                        sign_test, summarize)

HONE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MARGINS = os.path.join(HONE_ROOT, "configs", "margins.json")


def byte_parity(base_rows, cand_rows):
    mism = []
    for k in sorted(set(base_rows) & set(cand_rows)):
        if base_rows[k]["text_sha256"] != cand_rows[k]["text_sha256"]:
            mism.append(k)
    return mism


def byte_stability(rows_by_arm):
    per = {}
    for arm_rows in rows_by_arm.values():
        for key, r in arm_rows.items():
            pid = key.split("|")[0]
            per.setdefault(pid, set()).add(r["text_sha256"])
    return {p: n for p, n in ((p, len(s)) for p, s in per.items()) if n > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--aa", action="store_true")
    args = ap.parse_args()

    with open(args.results) as f:
        res = json.load(f)
    base, cand = res["rows"]["base"], res["rows"]["cand"]
    model = res["header"]["base_config"]["model"]

    pre_diffs, missing = paired_rel_diffs(base, cand, metric="prefill_s")
    pre = summarize(pre_diffs)
    st = sign_test(pre_diffs)
    dec_diffs, _ = paired_rel_diffs(base, cand, metric="generation_tps")
    dec = summarize(dec_diffs)
    contam = contamination_check(res["rows"])
    # absolute seconds saved per pair (base - cand)
    secs = []
    for k in sorted(set(base) & set(cand)):
        secs.append(base[k]["prefill_s"] - cand[k]["prefill_s"])
    med_saved = median(secs) if secs else 0.0

    print("===== T3-A prompt-cache verdict =====")
    print(f"model: {model}")
    print(f"pairs: {pre['n']}  missing: {missing}")
    print(f"prefill_s paired rel (cand/base-1; NEGATIVE=win): "
          f"median {pre.get('median_rel'):.4f}  MAD {pre.get('mad_rel'):.4f}  "
          f"min {pre.get('min_rel'):.4f}  max {pre.get('max_rel'):.4f}")
    print(f"prefill seconds saved per pair (base-cand): median {med_saved:.4f}s")
    print(f"sign test on prefill_s: {st}")
    print(f"decode tps paired rel (side-check, ~0 expected): "
          f"median {dec.get('median_rel'):.4f}  MAD {dec.get('mad_rel'):.4f}")
    print(f"contamination (decode-tps segment check): "
          f"{'CLEAN' if not contam else contam}")

    if args.aa:
        stab = byte_stability(res["rows"])
        mad = pre.get("mad_rel", 0.0)
        g1 = max(0.10, 3 * mad)
        print("\n----- A/A' noise floor (prefill_s) -----")
        print(f"byte-stability: {'STABLE' if not stab else stab}")
        print(f"suggested G1 prefill-reduction margin: >= {g1:.4f} "
              f"(= max(0.10, 3*MAD)); |median| bias {abs(pre.get('median_rel',0)):.4f}")
        print("register in margins.json under the model with a '_chat' suffix "
              "BEFORE judging the A/B run.")
        return

    # A/B verdict
    margins = {}
    if os.path.exists(MARGINS):
        with open(MARGINS) as f:
            margins = json.load(f)
    key = model + "_chat"
    m = margins.get(key, {})
    g1_min = m.get("g1_min_prefill_reduction", 0.10)
    mism = byte_parity(base, cand)
    g1_pass = (pre.get("median_rel", 0.0) <= -g1_min) and st["verdict"] == "POWERED" and st["p"] is not None and st["p"] < 0.05
    g3_pass = not mism
    g4_pass = not contam
    print(f"\nG1 (prefill reduction >= {g1_min:.2f}, powered): {g1_pass}")
    print(f"G3 (byte parity vs baseline): {g3_pass}  mismatches: {mism}")
    print(f"G4 (contamination-clean): {g4_pass}")
    print(f"PROMOTABLE (G1&G3&G4): {g1_pass and g3_pass and g4_pass}")
    if mism:
        print("NOTE: G3 mismatches -> run divergence_probe on those keys "
              "(near-tie-flip class check, A5 policy) before ruling.")


if __name__ == "__main__":
    main()
