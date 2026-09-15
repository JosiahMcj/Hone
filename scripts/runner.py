"""Harness-owned timed runner. ONE pass over the suite for ONE arm config.

This file's sha256 is recorded in every run header (PROTOCOL frozen
artifacts); candidates supply CONFIG JSON only, never code. Timing fields
come from mlx_lm's GenerationResponse; the orchestrator independently
recounts tokens and checks wall-clock sanity, so numbers emitted here are
claims, not verdicts.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone.gates import validate_config  # noqa: E402
from hone.suite import load_suite  # noqa: E402

STREAM_KWARG_KNOBS = ["num_draft_tokens", "kv_bits", "kv_group_size",
                      "quantized_kv_start", "max_kv_size",
                      "prefill_step_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-testing only; measured runs never set this")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    validate_config(cfg)

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    if cfg.get("wired_limit_gb"):
        mx.set_wired_limit(int(cfg["wired_limit_gb"] * (1 << 30)))

    model, tokenizer = load(cfg["model"])
    draft = None
    if cfg.get("draft_model"):
        draft, draft_tok = load(cfg["draft_model"])
        if draft_tok.vocab_size != tokenizer.vocab_size:
            raise SystemExit("draft/target vocab mismatch -- invalid pair")

    suite, suite_sha = load_suite(args.suite)
    if args.limit:
        suite = suite[: args.limit]

    kwargs = {k: cfg[k] for k in STREAM_KWARG_KNOBS if k in cfg}
    # temp>0 (T3-B): sampling. Default 0.0 = greedy (T1/T2 lossless class).
    # A base seed makes sampling reproducible per item (seed+index); both
    # arms sharing the seed are seed-MATCHED per prompt for a fair paired
    # comparison. At temp>0 the byte-compare quality gate does NOT apply
    # (spec sampling is distribution-preserving, not sample-path-identical
    # to no-draft even at a matched seed -- verified 2026-09-12).
    temp = float(cfg.get("temp", 0.0))
    seed = cfg.get("seed")
    sampler = make_sampler(temp=temp)

    out = open(args.out, "w")
    out.write(json.dumps({
        "type": "header", "config": cfg, "suite_sha256": suite_sha,
        "limited": bool(args.limit),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__ if hasattr(mlx_lm, "__version__")
        else "unknown",
    }) + "\n")

    for idx, item in enumerate(suite):
        if seed is not None:
            mx.random.seed(int(seed) + idx)
        messages = [{"role": "user", "content": item["prompt"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        text_parts, token_ids, last, draft_hits = [], [], None, 0
        for resp in stream_generate(model, tokenizer, prompt_ids,
                                    max_tokens=item["max_tokens"],
                                    draft_model=draft, sampler=sampler,
                                    **kwargs):
            text_parts.append(resp.text)
            token_ids.append(resp.token)
            draft_hits += 1 if resp.from_draft else 0
            last = resp
        text = "".join(text_parts)
        out.write(json.dumps({
            "type": "row", "prompt_id": item["id"],
            "category": item["category"],
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_tokens": last.prompt_tokens,
            "prompt_tps": last.prompt_tps,
            "generation_tokens": last.generation_tokens,
            "generation_tps": last.generation_tps,
            "peak_memory": last.peak_memory,
            "finish_reason": last.finish_reason,
            "from_draft_tokens": draft_hits,
            "token_ids": token_ids,
        }) + "\n")
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
