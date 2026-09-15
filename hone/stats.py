"""Paired statistics for timed comparisons. PROTOCOL laws:
- pair = (prompt, rep); analyze PAIRED relative differences only
- sign test over DISCORDANT pairs; < MIN_DISCORDANT => UNDERPOWERED,
  excluded from stop accounting, never counted as a negative
"""
import math
from statistics import median

MIN_DISCORDANT = 8
TIE_EPS = 1e-9  # relative-difference magnitude below this is a tie


def paired_rel_diffs(base_rows, cand_rows, metric="generation_tps"):
    """Rows are dicts keyed by (prompt_id, rep) -> row. Returns list of
    (key, rel_diff) where rel_diff = cand/base - 1 for every key present
    in BOTH arms. Keys missing from either arm are reported, never dropped
    silently."""
    common = sorted(set(base_rows) & set(cand_rows))
    missing = sorted((set(base_rows) ^ set(cand_rows)))
    diffs = []
    for k in common:
        b = base_rows[k][metric]
        c = cand_rows[k][metric]
        if b <= 0:
            raise ValueError(f"non-positive baseline metric at {k}")
        diffs.append((k, c / b - 1.0))
    return diffs, missing


def sign_test(diffs):
    """Exact two-sided binomial sign test over discordant pairs."""
    pos = sum(1 for _, d in diffs if d > TIE_EPS)
    neg = sum(1 for _, d in diffs if d < -TIE_EPS)
    n = pos + neg
    if n < MIN_DISCORDANT:
        return {"verdict": "UNDERPOWERED", "pos": pos, "neg": neg,
                "discordant": n, "p": None}
    k = max(pos, neg)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n * 2
    return {"verdict": "POWERED", "pos": pos, "neg": neg,
            "discordant": n, "p": min(1.0, p)}


CONTAMINATION_BAND = 0.15


def contamination_check(rows_by_arm, band=CONTAMINATION_BAND):
    """A3 sanity rule, PER-ARM per A5: a run is contaminated iff any
    SEGMENT's suite-median decode tps deviates more than `band` from its
    OWN ARM's grand median. Segment-scale, because the observed failure
    mode (r2: mid-segment contention) slows whole segments 40-60%, while
    clean-box per-pair tails reach +-14% without moving any median.
    Per-arm, because pooling flagged a legitimately +80%-faster candidate
    as "contamination" (A5 incident: all three screening winners).
    Returns list of offending (arm, rep, deviation)."""
    flagged = []
    for arm, arm_rows in rows_by_arm.items():
        segs = {}
        for key, r in arm_rows.items():
            pid, rep = key.split("|")
            segs.setdefault(rep, []).append(r["generation_tps"])
        meds = {rep: median(v) for rep, v in segs.items()}
        grand = median(meds.values())
        flagged.extend((arm, rep, m / grand - 1.0)
                       for rep, m in sorted(meds.items())
                       if abs(m / grand - 1.0) > band)
    return flagged


def summarize(diffs):
    vals = [d for _, d in diffs]
    if not vals:
        return {"n": 0}
    med = median(vals)
    mad = median([abs(v - med) for v in vals])
    return {"n": len(vals), "median_rel": med, "mad_rel": mad,
            "min_rel": min(vals), "max_rel": max(vals)}
