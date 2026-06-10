#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_corrected_s5s7.py — re-extract the two internal diagnostic signals with
the v1.0.1 corrected code (reviewer Q6):

  s5 (bias-head attention): offset-mapping token matching — fixes the BOS
      off-by-one and the single-token-only demographic matching.
  s7 (SAE bias features)  : hidden_states[L+1] — fixes the one-layer-early
      residual read for the Llama-Scope layer-15 SAE.

For every category file in results/v2/signals/main/*_signals.jsonl this writes
a corrected copy to results/v2/signals/corrected_s5s7/ with ONLY
signals.s5_bias_head and signals.s7_sae_feature replaced (all other fields,
ordering, and rows byte-identical), so downstream MoE/ablation reruns can
point at the corrected directory while sharing the original embeddings.

GPU REQUIRED (one forward pass with attentions + one 1-token generate per
example). Resumable: completed category files are skipped.

USAGE (pod, repo root, after `scp .env`):
  python3 scripts/extract_corrected_s5s7.py            # all categories
  python3 scripts/extract_corrected_s5s7.py --limit 8  # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
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

import numpy as np  # noqa: E402

from src.utils.data_loader import load_split  # noqa: E402
from src.utils.llm_utils import LLMWrapper  # noqa: E402
from src.signals.bias_head import compute_bias_head_activation, load_bias_heads  # noqa: E402
from src.signals.sae_feature import SAEWrapper, compute_sae_signal  # noqa: E402
from src.signals.prompts import PROMPT_BUILDERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("corrected_s5s7")

SRC = REPO / "results/v2/signals/main"
DST = REPO / "results/v2/signals/corrected_s5s7"
FEATURES_JSON = REPO / "results/v2_runpod/sae_layers/features_layer15.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="per-category row limit (0 = all)")
    args = ap.parse_args()

    DST.mkdir(parents=True, exist_ok=True)

    pool = {}
    for s in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = row.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d
    logger.info(f"pool: {len(pool)} instances")

    heads = load_bias_heads()[:20]
    feats = json.loads(FEATURES_JSON.read_text())
    feat_idx = list(feats["bias_features"]) if isinstance(feats, dict) else list(feats)
    logger.info(f"bias heads: {len(heads)}, SAE features: {len(feat_idx)}")
    assert len(feat_idx) == 56, f"expected the 56 released bias features, got {len(feat_idx)}"

    llm = LLMWrapper(model_name=args.model, dtype="bfloat16", device="auto")
    sae = SAEWrapper(release="llama_scope_lxr_8x", sae_id="l15r_8x", layer=15)
    builder = PROMPT_BUILDERS["vanilla"]

    for f in sorted(SRC.glob("*_signals.jsonl")):
        if f.name.startswith("._"):
            continue  # macOS AppleDouble 메타파일
        cat = f.name.replace("_signals.jsonl", "")
        out_f = DST / f.name
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        if out_f.exists():
            done = sum(1 for l in out_f.read_text().splitlines() if l.strip())
            if done >= len(rows):
                logger.info(f"[{cat}] skip (already {done} rows)")
                continue
        t0 = time.time()
        old5, new5, old7, new7 = [], [], [], []
        with out_f.open("w") as fh:
            for i, r in enumerate(rows):
                item = pool.get(f"{r['category']}::{int(r['example_id'])}")
                if item is None:
                    fh.write(json.dumps(r) + "\n")
                    continue
                s5 = compute_bias_head_activation(item=item, llm=llm, prompt_builder=builder,
                                                  head_indices=heads)
                s7 = compute_sae_signal(item=item, llm=llm, sae=sae, prompt_builder=builder,
                                        bias_feature_indices=feat_idx)
                sig = dict(r.get("signals", {}))
                old5.append(float(sig.get("s5_bias_head", 0.0)))
                old7.append(float(sig.get("s7_sae_feature", 0.0)))
                sig["s5_bias_head"] = float(s5)
                sig["s7_sae_feature"] = float(s7 if s7 is not None else 0.0)
                new5.append(sig["s5_bias_head"]); new7.append(sig["s7_sae_feature"])
                r = dict(r); r["signals"] = sig
                fh.write(json.dumps(r) + "\n")
                if (i + 1) % 200 == 0:
                    logger.info(f"[{cat}] {i+1}/{len(rows)} ({(time.time()-t0)/(i+1):.2f}s/ex)")
        if new5:
            c5 = float(np.corrcoef(old5, new5)[0, 1]) if np.std(old5) > 0 and np.std(new5) > 0 else float("nan")
            c7 = float(np.corrcoef(old7, new7)[0, 1]) if np.std(old7) > 0 and np.std(new7) > 0 else float("nan")
            z5o = float(np.mean(np.array(old5) == 0)); z5n = float(np.mean(np.array(new5) == 0))
            logger.info(f"[{cat}] done {len(rows)} rows in {time.time()-t0:.0f}s | "
                        f"s5 corr={c5:.3f} zeros {z5o:.2%}->{z5n:.2%} | s7 corr={c7:.3f}")
    logger.info(f"[done] corrected signals in {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
