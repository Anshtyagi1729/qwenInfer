#!/usr/bin/env python
"""Render the README figures from results/*.json."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def line(ax, x, y, color, label, label_y=None):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.annotate(label, (x[-1], label_y if label_y is not None else y[-1]),
                xytext=(10, 0), textcoords="offset points", va="center", color=INK2)


def decode() -> None:
    runs = json.loads((R / "decode_compare.json").read_text())
    ctx = sorted({int(c) for r in runs for c in r["ms_per_tok"]})
    med = {i: [st.median(r["ms_per_tok"][str(c)]["median"] for r in runs if r["impl"] == i)
               for c in ctx] for i in ("ours", "hf-dynamic", "hf-static")}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    line(ax, ctx, med["hf-static"], AQUA, "HF StaticCache", med["hf-static"][-1] - 2.2)
    line(ax, ctx, med["hf-dynamic"], ORANGE, "HF DynamicCache", med["hf-dynamic"][-1] + 1.4)
    line(ax, ctx, med["ours"], BLUE, "QwenInfer")
    ax.axhline(20.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ctx[0], 19.2, "memory-bandwidth floor, 20.5 ms", color=INK2, fontsize=10, va="top")
    ax.set_xscale("log", base=2); ax.set_xticks(ctx); ax.set_xticklabels(ctx)
    ax.set_ylim(0, 62); ax.set_xlim(ctx[0] * 0.8, ctx[-1] * 3.2)
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("decode latency (ms / token)")
    ax.set_title("Batch-1 decode latency, Qwen2.5-0.5B on an MX450 (lower is better)", loc="left")
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout(); fig.savefig(R / "decode_latency.png", dpi=160); plt.close(fig)


def scaling() -> None:
    d = json.loads((R / "batch_scaling.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, color, label in (("float16", ORANGE, "fp16"), ("bfloat16", BLUE, "bf16")):
        bs = sorted(int(b) for b in d[key])
        line(ax, bs, [d[key][str(b)]["tok_per_s"] for b in bs], color, label)
    fp16_2 = d["float16"]["2"]["tok_per_s"]
    ax.annotate("fp16: batch 2 is slower than batch 1\n(cuBLAS kernel cliff)", (2, fp16_2),
                xytext=(9, 330), textcoords="data", color=INK2, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
    ax.set_xlabel("batch size (concurrent sequences)"); ax.set_ylabel("decode throughput (tokens / s)")
    ax.set_title("Decode throughput vs batch size (higher is better)", loc="left")
    ax.set_xlim(0, 72); ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(R / "batch_scaling.png", dpi=160); plt.close(fig)


def serving() -> None:
    p = R / "serving.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())["results"]
    names = [r["mode"] for r in res][::-1]
    vals = [r["tok_per_s"] for r in res][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(names, vals, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, y, f"{v:.0f} tok/s", va="center", color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.18); ax.set_xlabel("output throughput (tokens / s)")
    ax.set_title("96-request serving workload (higher is better)", loc="left")
    fig.tight_layout(); fig.savefig(R / "serving.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    decode(); scaling(); serving()
    print("figures:", ", ".join(sorted(f.name for f in R.glob("*.png"))))
