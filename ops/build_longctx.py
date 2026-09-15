"""Build the frozen T3-C long-context suite: each item embeds a large
(~8-9k token) context assembled from THIS repo's own source files (self
-owned, deterministic, realistic code-reading), followed by a specific
question, then a modest decode budget. Output is suites/t3c/longctx.json
(the committed frozen artifact; this builder just makes it reproducible).
Same shape as the T1 suite {id, category, prompt, max_tokens} so the
existing runner.py + load_suite + verdict.py all work unchanged."""
import json
import os

from mlx_lm import load

HONE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_, tok = load("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")

# Source pool (repo-owned). Concatenated per item to reach the target.
POOL = [
    "scripts/measure.py", "scripts/runner.py", "scripts/runner_chat.py",
    "scripts/verdict.py", "scripts/verdict_chat.py", "hone/boxsafety.py",
    "hone/stats.py", "hone/gates.py", "hone/suite.py", "hone/chat_suite.py",
    "scripts/aa_calib.py", "scripts/divergence_probe.py",
]
TARGET_TOKENS = 8500

ITEMS = [
    ("longctx-guards", "What voids a timed segment, and how many times is it "
     "retried before the run aborts? Cite the function."),
    ("longctx-metric", "Which fields does the runner record per row, and which "
     "one is the primary speed metric? Answer from the code."),
    ("longctx-gates", "List every gate function and what fraud each one "
     "catches. Be specific to the code shown."),
    ("longctx-stats", "How is the sign test's 'discordant' count computed and "
     "what makes a run UNDERPOWERED? Quote the logic."),
]


def build_context(start):
    """Concatenate files starting at index `start` (wrapping) until the
    token budget is met; deterministic per start index."""
    parts, total, i = [], 0, start
    while total < TARGET_TOKENS and len(parts) < len(POOL):
        path = POOL[i % len(POOL)]
        with open(os.path.join(HONE, path)) as f:
            body = f.read()
        block = f"===== FILE: {path} =====\n{body}\n"
        parts.append(block)
        total += len(tok.encode(block))
        i += 1
    return "".join(parts)


def main():
    out = []
    for idx, (iid, question) in enumerate(ITEMS):
        ctx = build_context(idx * 2)
        prompt = ("You are reviewing a measurement harness. Read the source "
                  "files below, then answer the question.\n\n"
                  f"{ctx}\n\nQUESTION: {question}")
        ntok = len(tok.encode(prompt))
        out.append({"id": iid, "category": "long-context",
                    "prompt": prompt, "max_tokens": 256})
        print(f"{iid}: {ntok} context tokens")
    os.makedirs(os.path.join(HONE, "suites", "t3c"), exist_ok=True)
    dst = os.path.join(HONE, "suites", "t3c", "longctx.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {dst} ({len(out)} items)")


if __name__ == "__main__":
    main()
