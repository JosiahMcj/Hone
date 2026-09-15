"""Mechanical G1-G4 verdict on a measure.py results.json. No human
judgment inside (PROTOCOL promotion gate). G5 (different-day replication)
is procedural and reported as PENDING here.

--screen mode applies the same math but stamps the output NON-PROMOTABLE
(A4 screening law: screening selects candidates, never reports wins).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone.gates import finish_reason_shift  # noqa: E402
from hone.stats import (contamination_check, paired_rel_diffs,  # noqa: E402
                        sign_test, summarize)

HONE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def judge(results_path, margins_path, screen=False):
    with open(results_path) as f:
        res = json.load(f)
    with open(margins_path) as f:
        margins_all = json.load(f)
    model = res["header"]["base_config"]["model"]
    if model not in margins_all:
        raise SystemExit(f"no registered margins for {model} -- run A/A' "
                         "calibration and amend PROTOCOL first")
    m = margins_all[model]
    rows = res["rows"]

    verdict = {"model": model, "label": os.path.basename(
        os.path.dirname(results_path)), "screen": screen}

    # G4 first: a contaminated run judges nothing
    contaminated = contamination_check(rows)
    verdict["g4_contamination"] = contaminated
    verdict["g4_pass"] = not contaminated

    # G1: decode
    diffs, missing = paired_rel_diffs(rows["base"], rows["cand"],
                                      metric="generation_tps")
    st = sign_test(diffs)
    s = summarize(diffs)
    verdict["decode"] = {"summary": s, "sign_test": st,
                         "missing_pairs": missing}
    g1_effect = s["median_rel"] >= m["g1_min_median_rel"]
    g1_stat = (st["verdict"] == "POWERED" and st["p"] is not None
               and st["p"] < 0.05 and st["pos"] > st["neg"])
    verdict["g1_pass"] = bool(g1_effect and g1_stat)

    # G2: prefill + wall
    pdiffs, _ = paired_rel_diffs(rows["base"], rows["cand"],
                                 metric="prompt_tps")
    ps = summarize(pdiffs)
    walls = res.get("walls", {})
    wall_rel = (sum(walls.get("cand", [0])) / sum(walls.get("base", [1]))
                - 1.0) if walls.get("base") else None
    verdict["prefill_summary"] = ps
    verdict["wall_rel"] = wall_rel
    verdict["g2_pass"] = bool(
        ps["median_rel"] >= -m["g2_max_regression_rel"]
        and (wall_rel is None or wall_rel <= m["g2_max_regression_rel"]))

    # G3: lossless byte-parity (v1 candidates are all lossless-class)
    mismatch = [k for k in sorted(set(rows["base"]) & set(rows["cand"]))
                if rows["base"][k]["text_sha256"] != rows["cand"][k]["text_sha256"]]
    verdict["g3_mismatches"] = mismatch[:5]
    verdict["g3_pass"] = not mismatch

    verdict["finish_reason_shift"] = finish_reason_shift(
        rows["base"], rows["cand"])
    verdict["g5"] = "PENDING (different-day replication required)"
    verdict["promotable"] = (not screen and verdict["g1_pass"]
                             and verdict["g2_pass"] and verdict["g3_pass"]
                             and verdict["g4_pass"])
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--margins", default=os.path.join(
        HONE_ROOT, "configs", "margins.json"))
    ap.add_argument("--screen", action="store_true")
    args = ap.parse_args()
    v = judge(args.results, args.margins, screen=args.screen)
    print(json.dumps(v, indent=1))
    d = v["decode"]["summary"]
    tag = "SCREEN (non-promotable)" if v["screen"] else \
        ("PROMOTABLE pending G5" if v["promotable"] else "NOT PROMOTED")
    print(f"\n{v['label']}: median decode {d['median_rel']:+.1%} "
          f"(p={v['decode']['sign_test']['p']}), "
          f"prefill {v['prefill_summary']['median_rel']:+.1%}, "
          f"wall {v['wall_rel']:+.1%}, "
          f"G1={v['g1_pass']} G2={v['g2_pass']} G3={v['g3_pass']} "
          f"G4={v['g4_pass']} -> {tag}", file=sys.stderr)


if __name__ == "__main__":
    main()
