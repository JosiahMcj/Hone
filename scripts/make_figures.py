"""Build every figure in figures/ from the raw run records in evidence/.

No plotted value is typed by hand. Each number is recomputed from
evidence/ with the same paired statistics the promotion gates use
(hone.stats.paired_rel_diffs / summarize), and every headline is checked
against the text of RESULTS.md before anything is drawn. If a recomputed
headline does not appear in RESULTS.md, the script stops (a figure that
disagrees with the write-up is a bug in one of them).

The only non-measured inputs are model parameter counts (read off the
model names) used for the draft/target ratio axis.

Usage:  python scripts/make_figures.py        (needs matplotlib + numpy)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from hone.stats import paired_rel_diffs, summarize  # noqa: E402

EV = os.path.join(ROOT, "evidence")
OUT = os.path.join(ROOT, "figures")
RESULTS = open(os.path.join(ROOT, "RESULTS.md")).read()

# Reference categorical palette, light mode (validated all-pairs for 3 slots).
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "svg.hashsalt": "hone", "lines.linewidth": 2,
})
RNG = np.random.default_rng(20260804)


def load(name):
    with open(os.path.join(EV, name, "results.json")) as f:
        return json.load(f)


def rel(name, metric):
    rows = load(name)["rows"]
    d, missing = paired_rel_diffs(rows["base"], rows["cand"], metric=metric)
    assert not missing, f"{name}: unpaired rows {missing}"
    return np.array([v for _, v in d])


def boot_ci(vals, n=10000):
    """95% bootstrap CI of the median, resampling paired differences."""
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return np.percentile(meds, [2.5, 97.5])


def check(label, value, fmt):
    s = fmt.format(value * 100)
    if s not in RESULTS:
        raise SystemExit(f"STOP: {label} recomputes to {s}, not found in RESULTS.md")
    print(f"  ok  {label:38s} {s}")
    return s


def cuda_rel(base, cand, key):
    def rows(n):
        with open(os.path.join(EV, "cuda", n + ".json")) as f:
            return {(r["id"], r["rep"]): r for r in json.load(f)["rows"]}
    b, c = rows(base), rows(cand)
    common = sorted(set(b) & set(c))
    return np.array([c[k][key] / b[k][key] - 1 for k in common
                     if b[k].get(key) and c[k].get(key)])


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=200, bbox_inches="tight",
                metadata={"Software": None})
    fig.savefig(os.path.join(OUT, stem + ".svg"), bbox_inches="tight",
                metadata={"Date": None, "Creator": None})
    plt.close(fig)
    print(f"  wrote figures/{stem}.png/.svg")


def per_prompt_medians(name, metric):
    rows = load(name)["rows"]
    out = {}
    for arm in ("base", "cand"):
        by = {}
        for r in rows[arm].values():
            by.setdefault(r["prompt_id"], []).append(r[metric])
        out[arm] = {p: float(np.median(v)) for p, v in by.items()}
    return out


def dumbbell(ax, name, metric, labels, xlabel):
    pm = per_prompt_medians(name, metric)
    prompts = sorted(pm["base"], key=lambda p: pm["base"][p])
    y = np.arange(len(prompts))
    for i, p in enumerate(prompts):
        ax.plot([pm["base"][p], pm["cand"][p]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([pm["base"][p] for p in prompts], y, s=40, color=C1, zorder=2,
               edgecolor=SURF, linewidth=1.5, label=labels[0])
    ax.scatter([pm["cand"][p] for p in prompts], y, s=40, color=C2, zorder=3,
               edgecolor=SURF, linewidth=1.5, label=labels[1])
    ax.set_yticks(y, prompts, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")


def fig_t1():
    d = rel("full-spec-05b-k2", "generation_tps")
    g5 = rel("g5-spec-05b-k2", "generation_tps")
    s = check("T1 14B decode (primary)", np.median(d), "{:+.1f}%")
    s5 = check("T1 14B decode (G5 replication)", np.median(g5), "{:+.1f}%")
    lo, hi = boot_ci(d)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    dumbbell(ax, "full-spec-05b-k2", "generation_tps",
             ("stock mlx_lm", "+ 0.5B draft, k=2"), "decode tokens/sec (median per prompt)")
    ax.set_title(f"T1: speculative decoding on Qwen2.5-Coder-14B (M4, 24GB)\n"
                 f"paired median {s} (95% CI {lo*100:+.1f}% to {hi*100:+.1f}%), "
                 f"n={len(d)} pairs; replication {s5}", loc="left", fontsize=10)
    save(fig, "fig1_t1_decode")


RATIO = [  # (label, evidence run, draft active B, target active B, platform, moe)
    ("0.5B / 14B", "full-spec-05b-k2", 0.5, 14, "mlx", False),
    ("0.5B / 32B", "full-spec-32b-05b-k2", 0.5, 32, "mlx", False),
    ("0.6B / 4B", "screen3-4b-spec-k2", 0.6, 4, "mlx", False),
    ("3B / 14B", "screen-spec-3b-k2", 3, 14, "mlx", False),
    ("0.6B / 30B-A3B", "screen-spec-30b-06b-k2", 0.6, 3, "mlx", True),
    ("1.7B / 30B-A3B", "screen-spec-30b-17b-k2", 1.7, 3, "mlx", True),
]
CUDA = [("7B / 32B", "32b-base", "32b-spec", 7, 32), ("7B / 72B", "72b-base", "72b-spec", 7, 72)]


def fig_ratio():
    check("T2 dense 32B decode", np.median(rel("full-spec-32b-05b-k2", "generation_tps")), "{:+.2f}%")
    check("4B k=2 prefill (guard fail)", np.median(rel("screen3-4b-spec-k2", "prompt_tps")), "{:+.1f}%")
    check("MoE 0.6B k=2 decode", np.median(rel("screen-spec-30b-06b-k2", "generation_tps")), "{:+.1f}%")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for lab, run, dr, tg, _, moe in RATIO:
        x = dr / tg
        dec = np.median(rel(run, "generation_tps")) * 100
        pre = np.median(rel(run, "prompt_tps")) * 100
        kw = dict(s=60, zorder=5, linewidth=1.8,
                  facecolor=SURF if moe else C1, edgecolor=C1)
        a1.scatter(x, dec, **kw)
        a2.scatter(x, pre, **kw)
        dy = 10 if lab == "3B / 14B" else 4
        a1.annotate(lab, (x, dec), xytext=(6, dy), textcoords="offset points", fontsize=8, color=INK2)
    for lab, b, c, dr, tg in CUDA:
        x = dr / tg
        dec = np.median(cuda_rel(b, c, "decode_tps")) * 100
        # speed change = base_time / cand_time - 1 (same direction as MLX prompt_tps)
        pre = np.median(cuda_rel(c, b, "prefill_s")) * 100
        check(f"CUDA {lab} decode", dec / 100, "{:+.0f}%")
        a1.scatter(x, dec, s=110, marker="s", color=C2, zorder=4, edgecolor=SURF, linewidth=1.5)
        a2.scatter(x, pre, s=110, marker="s", color=C2, zorder=4, edgecolor=SURF, linewidth=1.5)
        a1.annotate(lab, (x, dec), xytext=(-8, -4), textcoords="offset points",
                    fontsize=8, color=INK2, ha="right")
        a2.annotate(lab, (x, pre), xytext=(-8, -4), textcoords="offset points",
                    fontsize=8, color=INK2, ha="right")
    ticks = [0.015, 0.03, 0.1, 0.2, 0.5]
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks(ticks, [f"{t:.1%}".replace(".0%", "%") for t in ticks])
        a.minorticks_off()
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xlabel("draft / target ratio (active parameters, log scale)")
    a2.axhline(-5, color=C3, lw=1.5, ls="--")
    a2.annotate("MLX prefill guard (-5%)", (0.06, -5), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=INK2)
    a1.set_ylabel("decode change vs stock, % (paired median)")
    a2.set_ylabel("prefill-speed change vs stock, %")
    h = [plt.Line2D([], [], marker="o", ls="", color=C1, label="MLX, dense target"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURF, markeredgecolor=C1,
                    label="MLX, MoE target (~3B active)"),
         plt.Line2D([], [], marker="s", ls="", color=C2, label="CUDA A30 / vLLM")]
    a1.legend(handles=h, frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("The draft/target ratio law, and where its constants stop transferring",
                 x=0.06, ha="left", fontsize=11)
    fig.text(0.06, -0.04, "Screen-stage points (3B/14B, 0.6B/4B, MoE) are R=2 screens. "
             "MLX and CUDA are separate series: no single curve is fit across platforms.",
             fontsize=8, color=INK2)
    save(fig, "fig2_ratio_law")


def fig_cache():
    d = rel("full-chat-14b-mini", "prefill_s")
    s = check("T3-A prefill cut", np.median(d), "{:+.1f}%")
    lo, hi = boot_ci(d)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    dumbbell(ax, "full-chat-14b-mini", "prefill_s",
             ("full re-prefill each turn", "prompt-cache reuse"), "time to first token, s (median per conversation)")
    ax.set_title(f"T3-A: prompt-cache reuse on multi-turn chat (14B, M4)\n"
                 f"paired median prefill {s} (95% CI {lo*100:+.1f}% to {hi*100:+.1f}%), "
                 f"n={len(d)} pairs, byte-identical output", loc="left", fontsize=10)
    save(fig, "fig3_prompt_cache")


def bars(ax, items, ylabel):
    x = np.arange(len(items))
    for i, (lab, vals, color) in enumerate(items):
        m = np.median(vals) * 100
        lo, hi = boot_ci(vals) * 100
        ax.bar(i, m, width=0.55, color=color, edgecolor=SURF, linewidth=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=4, lw=1.2)
        ax.annotate(f"{m:+.1f}%", (i, hi if m >= 0 else lo), xytext=(0, 5 if m >= 0 else -13),
                    textcoords="offset points", ha="center", fontsize=9)
    lo_all = min(min(np.median(v), boot_ci(v)[0]) for _, v, _ in items) * 100
    if lo_all < 0:
        ax.set_ylim(bottom=lo_all * 1.12)
    ax.set_xticks(x, [i[0] for i in items], fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig_stack():
    alone = rel("full-chat-14b-mini", "prefill_s")
    sp = rel("full-stack-14b-mini", "prefill_s")
    sd = rel("full-stack-14b-mini", "generation_tps")
    check("stacked prefill cut", np.median(sp), "{:+.1f}%")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars(ax, [("prefill\n(cache reuse alone)", alone, C1),
              ("prefill\n(reuse + spec decode)", sp, C1),
              ("decode\n(reuse + spec decode)", sd, C3)],
         "change vs no reuse, % (paired median, 95% CI)")
    ax.set_title("Stacking: prompt-cache reuse keeps its prefill cut with speculative\n"
                 "decoding active, and leaves decode unchanged", loc="left", fontsize=10)
    save(fig, "fig4_stacking")


def fig_robust():
    items = [("greedy, short\n(T1)", rel("full-spec-05b-k2", "generation_tps"), C1),
             ("sampled temp 0.7\n(T3-B)", rel("full-temp07-14b-mini", "generation_tps"), C1),
             ("9k-token context\n(T3-C)", rel("full-longctx-14b-mini", "generation_tps"), C1)]
    check("T3-B sampled decode", np.median(items[1][1]), "{:+.1f}%")
    check("T3-C long-context decode", np.median(items[2][1]), "{:+.1f}%")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars(ax, items, "decode change vs stock, % (paired median, 95% CI)")
    ax.set_title("The 14B speculative-decoding win holds under sampling and long context",
                 loc="left", fontsize=10)
    save(fig, "fig5_robustness")


def fig_cuda():
    mlx_dec = [("14B", rel("full-spec-05b-k2", "generation_tps")),
               ("32B", rel("full-spec-32b-05b-k2", "generation_tps"))]
    cu_dec = [("32B", cuda_rel("32b-base", "32b-spec", "decode_tps")),
              ("72B", cuda_rel("72b-base", "72b-spec", "decode_tps"))]
    pc = cuda_rel("32b-pcoff", "32b-pcon", "turn2_prefill_s")
    check("CUDA prefix-cache prefill", np.median(pc), "{:+.0f}%")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [2, 1.1]})
    bars(a1, [(f"MLX {k}\n0.5B draft", v, C1) for k, v in mlx_dec] +
         [(f"CUDA {k}\n7B draft", v, C2) for k, v in cu_dec],
         "decode change vs stock, %")
    a1.set_title("Speculative decoding", loc="left", fontsize=10)
    bars(a2, [("MLX 14B", rel("full-chat-14b-mini", "prefill_s"), C1), ("CUDA 32B", pc, C2)],
         "prefill change vs no reuse, %")
    a2.set_title("Prompt / prefix-cache reuse", loc="left", fontsize=10)
    fig.suptitle("Cross-platform coda: both levers pay on NVIDIA A30 / vLLM, with different constants",
                 x=0.06, ha="left", fontsize=11)
    fig.text(0.06, -0.06, "CUDA drafts had to be 7B (vLLM rejects the small Qwen draft's vocab size), so "
             "CUDA ratios are 10-22% vs 1.6-3.6% on MLX. CUDA prefix-cache suite used short shared prefixes.",
             fontsize=8, color=INK2)
    save(fig, "fig7_cuda_coda")


def fig_headroom():
    sweeps = [("0.5B draft", "screen-spec-05b-k{}", (2, 3, 4, 6, 8), C1, "o"),
              ("3B draft", "screen-spec-{}k{}", (2, 4, 6), C2, "s")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True)
    for lab, pat, ks, color, mk in sweeps:
        dec, pre = [], []
        for k in ks:
            run = pat.format(k) if "05b" in pat else f"screen-spec-3b-k{k}"
            d = np.median(rel(run, "generation_tps"))
            check(f"headroom {lab} k={k} screen decode", d, "{:+.1f}%")
            dec.append(d * 100)
            pre.append(np.median(rel(run, "prompt_tps")) * 100)
        a1.plot(ks, dec, color=color, marker=mk, markersize=8, label=lab,
                markeredgecolor=SURF, markeredgewidth=1.5)
        a2.plot(ks, pre, color=color, marker=mk, markersize=8, label=lab,
                markeredgecolor=SURF, markeredgewidth=1.5)
    for a in (a1, a2):
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xticks([2, 3, 4, 6, 8])
        a.set_xlabel("draft tokens per step (k)")
    a2.axhline(-5, color=C3, lw=1.5, ls="--")
    a2.annotate("prefill guard (-5%)", (5, -5), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=INK2)
    a1.set_ylabel("decode change vs stock, % (R=2 screen)")
    a2.set_ylabel("prefill-speed change vs stock, %")
    a1.legend(frameon=False, loc="upper right")
    a1.set_title("Decode", loc="left", fontsize=10)
    a2.set_title("Prefill", loc="left", fontsize=10)
    fig.suptitle("T1 headroom map on the 14B: a sharp optimum at 0.5B / k=2, an over-draft "
                 "wall past k~4, and a 3B prefill tax", x=0.06, ha="left", fontsize=11)
    fig.text(0.06, -0.04, "Screen-stage R=2 values (screens rank, they never promote). "
             "The promoted 0.5B/k=2 full run measured +75.0%.", fontsize=8, color=INK2)
    save(fig, "fig6_headroom")


if __name__ == "__main__":
    for f in (fig_t1, fig_ratio, fig_cache, fig_stack, fig_robust, fig_cuda, fig_headroom):
        print(f.__name__)
        f()
