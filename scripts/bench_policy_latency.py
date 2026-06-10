#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_policy_latency.py — measured wall-clock per example for (a) the deployable
condition-only policy and (b) the full 11-call signal pipeline, on the SAME
hardware as the reported baseline anchors (single H100 SXM, bfloat16, eager
attention, batch size 1, HF Transformers).

Condition-only  = 1 short-completion primary call (<=5 new tokens) + 1 MiniLM
                  sentence-embedding pass.
Full pipeline   = 4 prompt views (vanilla<=5, debias<=5, CoT<=100, cf-swap<=5)
                  + s1 evidence (<=96) + s2 counterfactual (<=64)
                  + 5 self-consistency samples (<=64, T=0.7) + MiniLM pass.
                  (s3/s5/s6/s7 reuse these passes; SAE encode is negligible.)

USAGE (pod, repo root): python3 scripts/bench_policy_latency.py --n 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except Exception:
    pass

from src.utils.llm_utils import LLMWrapper
from src.utils.data_loader import load_split, format_question

VAN = ("You are a helpful assistant. Answer the multiple-choice question by "
       "selecting (A), (B), or (C). Respond with only the letter.")
DEB = VAN + " Avoid stereotypes about any social group."
COT = ("Answer the multiple-choice question. Think step by step briefly, then "
       "finish with 'Answer: (X)'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", default="results/bench_latency.json")
    args = ap.parse_args()

    df = load_split(REPO / "data/sampled_v2", "test")
    items = [r.to_dict() for _, r in df.head(args.n).iterrows()]
    llm = LLMWrapper(model_name=args.model, dtype="bfloat16", device="auto",
                     hf_token=os.environ.get("HF_TOKEN"))
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # warmup
    q0 = format_question(items[0])
    llm.generate(user_message=q0, system_message=VAN, max_new_tokens=5)
    st.encode(["warmup"])

    # (a) condition-only
    t0 = time.time()
    for it in items:
        q = format_question(it)
        llm.generate(user_message=q, system_message=VAN, max_new_tokens=5)
        st.encode([f"{it['context']} {it['question']}"], show_progress_bar=False)
    t_cond = (time.time() - t0) / len(items)
    print(f"[bench] condition-only: {t_cond:.3f} s/example")

    # (b) full pipeline (11 generation calls + embedding)
    t0 = time.time()
    for it in items:
        q = format_question(it)
        llm.generate(user_message=q, system_message=VAN, max_new_tokens=5)        # vanilla
        llm.generate(user_message=q, system_message=DEB, max_new_tokens=5)        # debias view
        llm.generate(user_message=q, system_message=COT, max_new_tokens=100)      # CoT view
        llm.generate(user_message=q, system_message=VAN, max_new_tokens=5)        # cf-swap view (same len)
        llm.generate(user_message=q + "\nQuote the exact supporting span or NONE.",
                     system_message=VAN, max_new_tokens=96)                       # s1
        llm.generate(user_message=q, system_message=VAN, max_new_tokens=64)       # s2 swapped query
        for _ in range(5):                                                        # s4 samples
            llm.generate(user_message=q, system_message=VAN, max_new_tokens=64, temperature=0.7)
        st.encode([f"{it['context']} {it['question']}"], show_progress_bar=False)
    t_full = (time.time() - t0) / len(items)
    print(f"[bench] full 11-call pipeline: {t_full:.3f} s/example")

    out = {"n": len(items), "model": args.model, "cond_only_s": t_cond, "full_s": t_full,
           "hw": "1x H100 SXM, bfloat16, eager attention, batch size 1, HF Transformers"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("[done]", json.dumps(out))


if __name__ == "__main__":
    main()
