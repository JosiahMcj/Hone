"""T3-A timed runner: prompt-cache reuse on multi-turn conversations.

ONE pass over the chat suite for ONE arm config. Config JSON only (the
sha256 of THIS file is a frozen artifact in the run header). Two arms,
selected by cfg["reuse_prompt_cache"]:

  False (baseline): fresh cache, prefill the FULL templated conversation
         each turn, then generate the final assistant turn.
  True  (reuse):    a persistent cache already holds the conversation
         prefix (paid in prior turns); PRIME it here UNTIMED, then time
         only the delta (final user turn + gen prompt) + generation.

Semantics verified on this stack (probe 2026-09-11): the chat template
is prefix-stable and mlx-lm does NOT trim a common prefix, so reuse must
feed the delta, and reuse is token-identical to a fresh full prefill.

PRIMARY metric recorded DIRECTLY: prefill_s = time-to-first-token (the
prefill the arm actually paid). prompt_tps here is a DERIVED convenience
(prompt_tokens/prefill_s) so the orchestrator's wall-sanity/recount
gates still compose; never pair on it -- pair on prefill_s.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone.chat_suite import load_chat_suite  # noqa: E402
from hone.gates import GateError, validate_config  # noqa: E402


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
    reuse = bool(cfg.get("reuse_prompt_cache", False))

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(cfg["model"])
    # T3-A wave-2 (stacking): a draft may be present here, to ask whether
    # prompt-cache reuse's prefill win STACKS on spec-decode. The target
    # cache is what reuse persists; speculative_generate_step accepts a
    # prompt_cache and manages the draft's cache per call. Both arms carry
    # the same draft, so any spec-decode near-tie-flip cancels and the
    # cache-vs-nocache byte-parity gate still holds.
    draft = None
    draft_kw = {}
    if cfg.get("draft_model"):
        draft, draft_tok = load(cfg["draft_model"])
        if draft_tok.vocab_size != tokenizer.vocab_size:
            raise GateError("draft/target vocab mismatch -- invalid pair")
        draft_kw = {"draft_model": draft}
        if "num_draft_tokens" in cfg:
            draft_kw["num_draft_tokens"] = cfg["num_draft_tokens"]
    suite, suite_sha = load_chat_suite(args.suite)
    if args.limit:
        suite = suite[: args.limit]
    sampler = make_sampler(temp=0.0)  # greedy, lossless class

    out = open(args.out, "w")
    out.write(json.dumps({
        "type": "header", "config": cfg, "suite_sha256": suite_sha,
        "limited": bool(args.limit), "reuse_prompt_cache": reuse,
        "mlx_version": mx.__version__,
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
    }) + "\n")

    for item in suite:
        prefix = item["prefix_messages"]
        cached_ids = tokenizer.apply_chat_template(prefix, add_generation_prompt=False)
        full_ids = tokenizer.apply_chat_template(
            prefix + [{"role": "user", "content": item["final_user"]}],
            add_generation_prompt=True)
        if full_ids[:len(cached_ids)] != cached_ids:
            raise GateError(f"{item['id']}: chat template not prefix-stable; "
                            "delta boundary undefined")
        delta_ids = full_ids[len(cached_ids):]

        # With a draft, speculative_generate_step expects a COMBINED
        # prompt_cache = target-cache ++ draft-cache (it splits by the
        # target's layer count). Reuse must prime BOTH halves.
        model_cache = make_prompt_cache(model)
        if draft is not None:
            draft_cache = make_prompt_cache(draft)
            cache = model_cache + draft_cache
        else:
            draft_cache = None
            cache = model_cache
        prime_tokens = 0
        if reuse:
            # UNTIMED: prior turns' prefill, already paid in a real loop.
            prefix_arr = mx.array(cached_ids)[None]
            model(prefix_arr, cache=model_cache)
            if draft is not None:
                draft(prefix_arr, cache=draft_cache)
            mx.eval([c.state for c in cache])
            prime_tokens = len(cached_ids)
            fed = delta_ids
        else:
            fed = full_ids

        text_parts, token_ids, last, first_t, draft_hits = [], [], None, None, 0
        t0 = time.perf_counter()
        for resp in stream_generate(model, tokenizer, fed,
                                    max_tokens=item["max_tokens"],
                                    prompt_cache=cache, sampler=sampler,
                                    **draft_kw):
            if first_t is None:
                first_t = time.perf_counter()
            text_parts.append(resp.text)
            token_ids.append(resp.token)
            draft_hits += 1 if resp.from_draft else 0
            last = resp
        t_end = time.perf_counter()
        prefill_s = (first_t - t0) if first_t is not None else (t_end - t0)
        decode_s = (t_end - first_t) if first_t is not None else 0.0
        gen_tokens = len(token_ids)
        text = "".join(text_parts)

        out.write(json.dumps({
            "type": "row", "prompt_id": item["id"], "category": item["category"],
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "reuse": reuse,
            "prime_tokens": prime_tokens,
            "fed_tokens": len(fed),
            "prefill_s": prefill_s,               # PRIMARY metric (TTFT)
            "decode_s": decode_s,
            "prompt_tokens": last.prompt_tokens,
            "prompt_tps": (last.prompt_tokens / prefill_s) if prefill_s > 0 else 0.0,
            "generation_tokens": last.generation_tokens,
            "generation_tps": (gen_tokens / decode_s) if decode_s > 0 else 0.0,
            "from_draft_tokens": draft_hits,
            "peak_memory": last.peak_memory,
            "finish_reason": last.finish_reason,
            "token_ids": token_ids,
        }) + "\n")
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
