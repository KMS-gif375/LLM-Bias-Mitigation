#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
token_signal_plot.py  —  CPU step (no GPU / no model needed).

Renders the "annotated sentence" figure from the JSON produced by
scripts/token_signal_extract.py: for each token of the BBQ context+question it
shows where the signals react —
  * s5 (bias-head attention)  : heat row, how much the bias heads attend to the token
  * s7 (SAE bias feature)     : heat row, bias-feature activation at the token
  * s1 (evidence span)        : green underline on the quoted support span
  * s2 (group swap)           : purple box on the swapped demographic tokens
plus a header with the question, the model's raw vs. final answer, the per-token
peak SAE feature, and the 7 scalar signal values.

USAGE (any python with matplotlib; the repo venv has it):
  ./venv/bin/python scripts/token_signal_plot.py \
      --json results/.../token_signals/nationality_312.json \
      --out  results/.../token_signals/nationality_312.pdf
Writes both .pdf and .png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _style():
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.size": 11, "axes.titlesize": 12, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo, hi = float(np.min(a)), float(np.max(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _ans_letter(i):
    return {0: "A", 1: "B", 2: "C"}.get(i, "?") if isinstance(i, int) else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-label-fs", type=float, default=7.5)
    args = ap.parse_args()
    _style()

    d = json.loads(Path(args.json).read_text())
    meta = d["meta"]
    toks = d["tokens"]
    disp = [bool(x) for x in d["display_mask"]]
    # fall back to all tokens if no display region was found
    if not any(disp):
        disp = [True] * len(toks)
    idx = [i for i, m in enumerate(disp) if m]

    # RoBERTa-family tokenizers expose the newline marker as ``Ċ``. It is
    # useful internally but should not appear as a stray glyph in the figure.
    sub_tok = [toks[i].replace("Ċ", "") for i in idx]
    s5 = _norm([d["s5_attention"][i] for i in idx])
    s7 = _norm([d["s7_sae"][i] for i in idx])
    demo = [bool(d["demographic_mask"][i]) for i in idx]
    ev = [bool(d.get("s1_evidence", {}).get("token_mask", [False] * len(toks))[i]) for i in idx]
    n = len(idx)

    # ---- figure scaffold ---------------------------------------------------
    width = float(np.clip(0.34 * n, 9, 24))
    fig = plt.figure(figsize=(width, 4.7))
    gs = fig.add_gridspec(
        nrows=4, ncols=1, height_ratios=[1.45, 0.55, 0.55, 0.42], hspace=0.18,
        left=0.085, right=0.995, top=0.97, bottom=0.30)
    ax_hdr = fig.add_subplot(gs[0]); ax_hdr.axis("off")
    ax_s5 = fig.add_subplot(gs[1])
    ax_s7 = fig.add_subplot(gs[2])
    ax_mk = fig.add_subplot(gs[3])

    # ---- header text -------------------------------------------------------
    dec = d.get("decision", {})
    sig = d.get("scalar_signals", {})
    prim = d.get("primary_answer")
    prim = int(prim) if isinstance(prim, (int, float)) else None
    fin = dec.get("final_answer")
    opts = meta.get("options", ["", "", ""])
    peak = d.get("s7_peak", {})
    title = (f"{meta['category']} · example {meta['example_id']} · "
             f"context = {meta.get('context_condition','?')}")
    qline = f"Q:  {meta['question']}"
    aline = (f"raw answer:  ({_ans_letter(prim)}) {opts[prim] if prim in (0,1,2) else '?'}"
             f"    →    final:  ({_ans_letter(fin)}) {opts[fin] if fin in (0,1,2) else 'Unknown'}")
    if dec.get("p_score") is not None:
        aline += f"     [MoE p={dec['p_score']:.3f}, τ={dec.get('threshold_used')}]"
    sline = "   ".join(
        f"{k.split('_')[0]}={float(sig[k]):.2f}" for k in
        ["s1_evidence", "s2_counterfactual", "s3_confidence", "s4_consistency",
         "s5_bias_head", "s6_prompt_sensitivity", "s7_sae_feature"] if k in sig)
    pkline = (f"top SAE bias-feature #{peak.get('feature')} peaks at "
              f"“{peak.get('token','').strip()}” ({peak.get('value',0):.2f})") if peak else ""
    ax_hdr.text(0.0, 0.93, title, fontsize=12, fontweight="bold", va="top")
    ax_hdr.text(0.0, 0.66, qline, fontsize=10.5, va="top")
    ax_hdr.text(0.0, 0.42, aline, fontsize=10.5, va="top",
                color=("#B00020" if (prim != fin) else "#1A1A1A"))
    ax_hdr.text(0.0, 0.20, sig and ("signals:  " + sline) or "", fontsize=9.5,
                color="#444", va="top")
    if pkline:
        ax_hdr.text(0.0, 0.02, pkline, fontsize=9.5, color="#7A3E00", va="top")

    # ---- heat rows ---------------------------------------------------------
    for ax, vals, cmap, lab in [
        (ax_s5, s5, "Blues", "s5  bias-head\nattention"),
        (ax_s7, s7, "Oranges", "s7  SAE\nbias feature"),
    ]:
        ax.imshow(vals[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  extent=(-0.5, n - 0.5, 0, 1))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylabel(lab, fontsize=9, rotation=0, ha="right", va="center")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ---- marker row: s1 evidence (green), s2 swap / demographic (purple) ---
    ax_mk.set_xlim(-0.5, n - 0.5); ax_mk.set_ylim(0, 1)
    ax_mk.set_yticks([])
    for sp in ax_mk.spines.values():
        sp.set_visible(False)
    for j in range(n):
        if ev[j]:
            ax_mk.add_patch(Rectangle((j - 0.5, 0.55), 1.0, 0.4,
                                      facecolor="#2E7D32", edgecolor="none", alpha=0.85))
        if demo[j]:
            ax_mk.add_patch(Rectangle((j - 0.5, 0.05), 1.0, 0.4,
                                      facecolor="none", edgecolor="#6A1B9A", lw=1.6))
    ax_mk.set_ylabel("s1 · s2\nspans", fontsize=9, rotation=0, ha="right", va="center")

    # ---- token labels via xticks (robust spacing; demographic in red bold) -
    fs = float(np.clip(220.0 / max(n, 1), 5.0, args.max_label_fs))
    ax_mk.set_xticks(range(n))
    ax_mk.set_xticklabels([t if t.strip() else "·" for t in sub_tok],
                          rotation=90, fontsize=fs)
    ax_mk.tick_params(axis="x", length=0)
    for tk, isd in zip(ax_mk.get_xticklabels(), demo):
        if isd:
            tk.set_color("#B00020"); tk.set_fontweight("bold")

    # ---- legend (top-right, out of the token-label area) -------------------
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#2E7D32", edgecolor="none", label="s1 evidence span"),
           Patch(facecolor="white", edgecolor="#6A1B9A", lw=1.6, label="s2 swapped group")]
    fig.legend(handles=leg, loc="upper right", fontsize=8, frameon=False,
               bbox_to_anchor=(0.998, 0.998))

    fig.text(0.5, -0.02,
             "Heat = per-token signal intensity (min–max normalized).  "
             "Attention / SAE are diagnostic (descriptive), not causal.",
             ha="center", fontsize=8, color="#777")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[done] wrote {out} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
