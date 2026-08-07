#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_policy_latency.py — generation-call micro-benchmark for (a) the deployable
condition-only policy and (b) an 11-call prompt schedule, on the same hardware
as the reported baseline anchors (single H100 SXM, bfloat16, eager attention,
batch size 1, HF Transformers).

This is not an end-to-end signal-pipeline benchmark: it does not time the actual
counterfactual transformation, attention-head extraction, SAE hooks, or MoE
forward pass. The reported full-pipeline value is therefore a lower-bound proxy.

Condition-only  = 1 short-completion primary call (<=5 new tokens) + 1 MiniLM
                  sentence-embedding pass.
Full pipeline   = 4 prompt views (vanilla<=5, debias<=5, CoT<=100,
                  counterfactual-instruction/unchanged-context<=5)
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
CF_INSTRUCTION = (
    "You are a helpful assistant evaluating whether your answer would change "
    "if the demographic groups in the context were swapped. Answer the "
    "multiple-choice question by selecting (A), (B), or (C). Respond with only "
    "the letter."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--schedule",
        choices=("legacy", "current"),
        default="current",
        help=(
            "legacy reproduces the call structure of the archived timing "
            "artifact; current uses the revised fourth-view instruction. "
            "Neither mode is an end-to-end signal-pipeline benchmark."
        ),
    )
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
        fourth_prompt = VAN if args.schedule == "legacy" else CF_INSTRUCTION
        llm.generate(user_message=q, system_message=fourth_prompt,
                     max_new_tokens=5)                                            # unchanged-context fourth view
        llm.generate(user_message=q + "\nQuote the exact supporting span or NONE.",
                     system_message=VAN, max_new_tokens=96)                       # s1
        llm.generate(user_message=q, system_message=VAN, max_new_tokens=64)       # s2 swapped query
        for _ in range(5):                                                        # s4 samples
            llm.generate(user_message=q, system_message=VAN, max_new_tokens=64, temperature=0.7)
        st.encode([f"{it['context']} {it['question']}"], show_progress_bar=False)
    t_full = (time.time() - t0) / len(items)
    print(f"[bench] 11-generation lower-bound proxy: {t_full:.3f} s/example")

    out = {"artifact_schema_version": 2,
           "n": len(items), "model": args.model, "schedule": args.schedule,
           "fourth_prompt": (
               "legacy short vanilla prompt" if args.schedule == "legacy"
               else "counterfactual instruction with unchanged context"
           ),
           "s2_query_uses_actual_swap": False,
           "cond_only_s": t_cond,
           "generation_schedule_proxy_s": t_full,
           "is_end_to_end_full_pipeline": False,
           "omitted_components": ["actual demographic swap", "s7 one-token generation",
                                  "four choice-log-probability forwards",
                                  "attention extraction", "SAE encoding", "MoE forward"],
           "hw": "1x H100 SXM, bfloat16, eager attention, batch size 1, HF Transformers"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("[done]", json.dumps(out))


if __name__ == "__main__":
    main()
