import math

import pytest

from hone.stats import (MIN_DISCORDANT, contamination_check,
                        paired_rel_diffs, sign_test, summarize)


def rows(vals, metric="generation_tps"):
    return {k: {metric: v} for k, v in vals.items()}


def test_paired_rel_diffs_pairs_by_key():
    base = rows({"p1|rep1": 10.0, "p2|rep1": 20.0})
    cand = rows({"p1|rep1": 11.0, "p2|rep1": 20.0})
    diffs, missing = paired_rel_diffs(base, cand)
    assert missing == []
    d = dict(diffs)
    assert d["p1|rep1"] == pytest.approx(0.10)
    assert d["p2|rep1"] == pytest.approx(0.0)


def test_missing_pairs_are_reported_never_silently_dropped():
    base = rows({"p1|rep1": 10.0, "p2|rep1": 20.0})
    cand = rows({"p1|rep1": 11.0})
    diffs, missing = paired_rel_diffs(base, cand)
    assert missing == ["p2|rep1"]
    assert len(diffs) == 1


def test_nonpositive_baseline_raises():
    with pytest.raises(ValueError):
        paired_rel_diffs(rows({"a": 0.0}), rows({"a": 1.0}))


def test_sign_test_exact_p_value():
    # 9 positive, 1 negative: p = 2 * (C(10,9)+C(10,10)) / 2^10 = 0.021484375
    diffs = [(f"k{i}", 0.1) for i in range(9)] + [("k9", -0.1)]
    st = sign_test(diffs)
    assert st["verdict"] == "POWERED"
    assert st["p"] == pytest.approx(2 * (math.comb(10, 9) + 1) / 1024)


def test_ties_are_not_discordant():
    # ties-destroy-power law: zeros must not count as evidence
    diffs = [(f"k{i}", 0.0) for i in range(20)] + \
            [(f"j{i}", 0.05) for i in range(MIN_DISCORDANT)]
    st = sign_test(diffs)
    assert st["discordant"] == MIN_DISCORDANT
    assert st["verdict"] == "POWERED"


def test_underpowered_never_a_negative():
    diffs = [("a", 0.2), ("b", -0.1), ("c", 0.3)]
    st = sign_test(diffs)
    assert st["verdict"] == "UNDERPOWERED"
    assert st["p"] is None


def _arm(tps_by_rep, prompts=3):
    return {f"p{p}|rep{r}": {"generation_tps": tps}
            for r, tps in tps_by_rep.items() for p in range(prompts)}


def test_canary_slow_segment_flagged_as_contamination():
    # r2 pattern: one whole segment at ~half speed must fire
    rows = {"base": _arm({1: 40.0, 2: 41.0, 3: 18.0}),
            "cand": _arm({1: 40.5, 2: 40.0, 3: 39.5})}
    hits = contamination_check(rows)
    assert hits == [("base", "rep3", pytest.approx(18.0 / 40.0 - 1.0))]


def test_canary_real_ab_effect_is_not_contamination():
    # A5: a candidate legitimately +80% faster must NOT be flagged --
    # deviations are judged against each ARM's own grand median, never
    # the pooled one (pooled flagged all three screening winners).
    rows = {"base": _arm({1: 12.6, 2: 12.7, 3: 12.5}),
            "cand": _arm({1: 22.6, 2: 22.8, 3: 22.5})}
    assert contamination_check(rows) == []


def test_slow_segment_within_one_arm_still_flags():
    rows = {"base": _arm({1: 12.6, 2: 12.7, 3: 12.5}),
            "cand": _arm({1: 22.6, 2: 22.8, 3: 9.0})}
    hits = contamination_check(rows)
    assert [(a, r) for a, r, _ in hits] == [("cand", "rep3")]


def test_clean_wobble_within_band_passes():
    # r6 pattern: +-8% segment wobble is ambient, not contamination
    rows = {"base": _arm({1: 37.5, 2: 41.0, 3: 40.0}),
            "cand": _arm({1: 38.2, 2: 40.3, 3: 41.5})}
    assert contamination_check(rows) == []


def test_summarize_median_and_mad():
    diffs = [("a", 0.10), ("b", 0.20), ("c", 0.30)]
    s = summarize(diffs)
    assert s["median_rel"] == pytest.approx(0.20)
    assert s["mad_rel"] == pytest.approx(0.10)
