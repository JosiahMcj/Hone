"""Integrity + parity gates. Every gate here has a canary test in tests/
that proves it FAILS on synthetic fraud (tautological-test law)."""

# Whitelist of knobs a candidate config may set (PROTOCOL anti-gaming).
# Candidates are config JSON only; unknown keys are a hard error.
CONFIG_WHITELIST = {
    "model",              # required: HF repo or local path
    "draft_model",        # spec decode draft (path/repo)
    "num_draft_tokens",
    "kv_bits", "kv_group_size", "quantized_kv_start",
    "max_kv_size", "prefill_step_size",
    "wired_limit_gb",
    "reuse_prompt_cache",  # T3-A: reuse KV cache across turns (feed delta)
    "temp",               # T3-B: sampling temperature (0.0 = greedy)
    "seed",               # T3-B: base RNG seed (per-item = seed+index)
    "label",              # display only
}
REQUIRED_CONFIG_KEYS = {"model"}

RECOUNT_TOL_TOKENS = 2
RECOUNT_TOL_FRAC = 0.02
WALL_SANITY_FACTOR = 0.95


class GateError(Exception):
    """A gate failed. Segment/comparison is VOID, never silently patched."""


def validate_config(cfg):
    unknown = set(cfg) - CONFIG_WHITELIST
    if unknown:
        raise GateError(f"config knobs not in whitelist: {sorted(unknown)}")
    missing = REQUIRED_CONFIG_KEYS - set(cfg)
    if missing:
        raise GateError(f"config missing required keys: {sorted(missing)}")
    return True


def check_recount(claimed_tokens, recounted_tokens, prompt_id="?"):
    tol = max(RECOUNT_TOL_TOKENS, claimed_tokens * RECOUNT_TOL_FRAC)
    if abs(claimed_tokens - recounted_tokens) > tol:
        raise GateError(
            f"recount mismatch on {prompt_id}: runner claimed "
            f"{claimed_tokens}, harness recounted {recounted_tokens}")
    return True


def check_wall_sanity(wall_s, rows):
    """Subprocess wall-clock must cover the compute time the tps claims
    imply (forged-tps defense). rows: list of per-prompt result dicts."""
    implied = 0.0
    for r in rows:
        implied += r["prompt_tokens"] / max(r["prompt_tps"], 1e-9)
        implied += r["generation_tokens"] / max(r["generation_tps"], 1e-9)
    if wall_s < WALL_SANITY_FACTOR * implied:
        raise GateError(
            f"wall-clock {wall_s:.1f}s < {WALL_SANITY_FACTOR} x implied "
            f"compute {implied:.1f}s -- tps claims not physically possible")
    return True


def check_lossless_parity(base_rows, cand_rows):
    """Byte-identical output per (prompt, rep). Applicability gated on
    measured baseline byte-stability (calibration, PROTOCOL)."""
    mismatches = [k for k in sorted(set(base_rows) & set(cand_rows))
                  if base_rows[k]["text_sha256"] != cand_rows[k]["text_sha256"]]
    if mismatches:
        raise GateError(f"lossless parity failed on {len(mismatches)} "
                        f"pairs, first: {mismatches[0]}")
    return True


def check_frozen_hashes(header, expected):
    """expected: dict of name -> sha256 recorded at freeze time."""
    for name, want in expected.items():
        got = header.get(name)
        if got != want:
            raise GateError(f"frozen-artifact hash mismatch: {name}")
    return True


def finish_reason_shift(base_rows, cand_rows):
    """Report (not gate): distribution of finish_reason per arm. A speedup
    coinciding with a shift here is flagged per PROTOCOL EOS law."""
    def dist(rows):
        d = {}
        for r in rows.values():
            d[r["finish_reason"]] = d.get(r["finish_reason"], 0) + 1
        return d
    return {"base": dist(base_rows), "cand": dist(cand_rows)}
