"""A5 evidence tool: at every spec-decode divergence point, teacher-force
the TARGET model over the common prefix and measure the top-2 logprob gap
at the position where the argmax flipped. Near-tie gaps (vs large control
gaps elsewhere) establish the batched-verification numerics mechanism.
One forward per diverged prompt; gaps at control positions come from the
same logits tensor.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone.suite import load_suite  # noqa: E402

HONE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--suite", default=os.path.join(
        HONE_ROOT, "suites", "t1", "prompts.json"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    res = json.load(open(args.results))
    rows = res["rows"]
    model_path = res["header"]["base_config"]["model"]
    suite, _ = load_suite(args.suite)
    prompts = {it["id"]: it["prompt"] for it in suite}

    seen, cases = set(), []
    for k in sorted(set(rows["base"]) & set(rows["cand"])):
        pid = k.split("|")[0]
        b, c = rows["base"][k], rows["cand"][k]
        if b["text_sha256"] != c["text_sha256"] and pid not in seen:
            seen.add(pid)
            bt, ct = b["text"], c["text"]
            div = next((i for i, (x, y) in enumerate(zip(bt, ct)) if x != y),
                       min(len(bt), len(ct)))
            # exact token ids (runs after A5) beat char-prefix re-tokenizing
            ids = b.get("token_ids"), c.get("token_ids")
            tok_div = None
            if all(ids):
                tok_div = next((i for i, (x, y) in enumerate(zip(*ids))
                                if x != y), min(map(len, ids)))
            cases.append((pid, bt, ct, div, ids[0], tok_div))

    if not cases:
        print(json.dumps({"model": model_path, "cases": [],
                          "note": "no divergences"}))
        return

    model, tokenizer = load(model_path)
    report = []
    for pid, bt, ct, div, base_ids, tok_div in cases:
        msg = [{"role": "user", "content": prompts[pid]}]
        pids = tokenizer.apply_chat_template(msg, add_generation_prompt=True)
        if tok_div is not None:
            prefix_ids = base_ids[:tok_div]  # EXACT generated prefix
        else:
            prefix_ids = tokenizer.encode(bt[:div], add_special_tokens=False)
        full = list(pids) + list(prefix_ids)
        logits = model(mx.array([full]))[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        def top2(pos):
            v = lp[pos]
            idx = mx.argpartition(-v, 2)[:2]
            pair = sorted(((float(v[i]), int(i)) for i in idx.tolist()),
                          reverse=True)
            return {"gap": pair[0][0] - pair[1][0],
                    "top": tokenizer.decode([pair[0][1]]),
                    "second": tokenizer.decode([pair[1][1]])}

        n = len(full)
        d = top2(n - 1)  # predicts the first diverged token
        controls = [dict(top2(len(pids) + int(len(prefix_ids) * f) - 1),
                         at=f"{int(f*100)}%")
                    for f in (0.25, 0.5, 0.75)]
        report.append({"prompt": pid, "div_char": div,
                       "div_token_index": tok_div,
                       "exact_token_prefix": tok_div is not None,
                       "divergence": d, "controls": controls,
                       "base_next": bt[div:div + 12],
                       "cand_next": ct[div:div + 12]})
    print(json.dumps({"model": model_path, "cases": report}, indent=1))


if __name__ == "__main__":
    main()
