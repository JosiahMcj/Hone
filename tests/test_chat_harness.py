"""Canary tests for the T3-A prompt-cache harness. Every new gate path
must PASS on clean input and FAIL on synthetic fraud (tautological-test
law); the verdict's win-direction is asserted BOTH ways (a detector needs
a known positive AND a known negative)."""
import json
import os

import pytest

from hone.chat_suite import SUITE_T3A, load_chat_suite
from hone.gates import GateError, validate_config
from hone.stats import paired_rel_diffs, sign_test, summarize

HONE_ROOT = os.path.dirname(os.path.dirname(__file__))


# --- config whitelist accepts the new knob, still rejects fraud ---

def test_reuse_flag_is_whitelisted():
    validate_config({"model": "m", "reuse_prompt_cache": True, "label": "x"})


def test_unknown_knob_still_hard_errors_with_chat_config():
    with pytest.raises(GateError, match="whitelist"):
        validate_config({"model": "m", "reuse_prompt_cache": True,
                         "secretly_patch_timer": 1})


# --- chat suite loads, is frozen-shaped, and rejects malformed items ---

def test_frozen_chat_suite_loads_and_hashes():
    items, sha = load_chat_suite()
    assert len(items) >= 4
    assert len(sha) == 64
    assert all({"id", "category", "prefix_messages", "final_user",
                "max_tokens"} <= set(it) for it in items)


def test_chat_suite_rejects_bad_message(tmp_path):
    bad = [{"id": "x", "category": "c", "max_tokens": 8,
            "final_user": "q",
            "prefix_messages": [{"role": "wizard", "content": "hi"}]}]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="bad message"):
        load_chat_suite(str(p))


def test_chat_suite_rejects_duplicate_ids(tmp_path):
    dup = [{"id": "x", "category": "c", "max_tokens": 8, "final_user": "q",
            "prefix_messages": [{"role": "user", "content": "a"}]},
           {"id": "x", "category": "c", "max_tokens": 8, "final_user": "q",
            "prefix_messages": [{"role": "user", "content": "b"}]}]
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(dup))
    with pytest.raises(ValueError, match="duplicate"):
        load_chat_suite(str(p))


# --- verdict win-direction: LOWER prefill_s is a win (negative rel diff) ---

def _rows(prefill_by_key):
    return {k: {"prefill_s": v, "generation_tps": 20.0, "text_sha256": "z"}
            for k, v in prefill_by_key.items()}


def test_g1_known_positive_cache_faster():
    # cand (cache) prefills in 1/3 the time on every pair -> clear win
    base = _rows({f"p{i}|rep{r}": 0.9 for i in range(6) for r in range(2)})
    cand = _rows({f"p{i}|rep{r}": 0.3 for i in range(6) for r in range(2)})
    diffs, missing = paired_rel_diffs(base, cand, metric="prefill_s")
    s = summarize(diffs)
    st = sign_test(diffs)
    assert not missing
    assert s["median_rel"] < -0.10          # >=10% reduction
    assert st["verdict"] == "POWERED" and st["p"] < 0.05


def test_g1_known_negative_no_improvement():
    # identical prefill_s -> no win; sign test must NOT declare an effect
    base = _rows({f"p{i}|rep{r}": 0.5 for i in range(6) for r in range(2)})
    cand = _rows({f"p{i}|rep{r}": 0.5 for i in range(6) for r in range(2)})
    diffs, _ = paired_rel_diffs(base, cand, metric="prefill_s")
    s = summarize(diffs)
    st = sign_test(diffs)
    assert abs(s["median_rel"]) < 1e-9
    assert not (s["median_rel"] <= -0.10)   # G1 would NOT pass
    assert st["verdict"] == "UNDERPOWERED"  # zero discordant pairs
