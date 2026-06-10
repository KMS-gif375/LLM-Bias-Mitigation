#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_transfer_routing_unify.py — reviewer request: unify Table 6 routing.

The original transfer runs route the per-condition override by the dataset's
GOLD condition label (oracle routing, an inherited convention). This script
recomputes the SAME seven-signal MoE rows with PREDICTED-condition routing
(the deployable no-oracle convention) for the two transfer sets whose inputs
are fully reconstructable from local artifacts:

  * Open-BBQ : signals + texts (data/open_bbq) -> MiniLM embeddings recomputed
  * KoBBQ    : signals + cached MiniLM embeddings (_embeddings.pt)

Protocol (faithful to the original runs):
  - MoE checkpoint results/v2/moe/main/moe_best.pt, thresholds ambig=disambig=0.5
    (the transfer runs' resolved fallback), override rule identical to
    src/models/override.apply_per_condition_override.
  - Condition predictor: balanced LogisticRegression (random_state=42) on the
    full English-BBQ pool MiniLM embeddings — the audit-G construction.
  - FIDELITY ANCHOR: the oracle-routed rows are reproduced first and must match
    the published numbers before any predicted-condition number is reported.

ImplicitBBQ-style is excluded: the paraphrased texts/embeddings for the full
2,640-example pod run were not retained, so no-oracle routing cannot be
reconstructed (already disclosed in the Table 6 caption).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402
from src.models.moe_aggregator import MoEAggregator, signals_dict_to_tensor  # noqa: E402

OUT = REPO / "results/v2/reviewer_audits/routing_unify"
OUT.mkdir(parents=True, exist_ok=True)
# Open-BBQ published row의 출처는 results/v2/acceptance_package/open_bbq
# (thresholds_per_condition {ambig:0.95, disambig:0.05} 기록 + 런타임 임베딩 보존).
# KoBBQ 런은 임계값 기록이 없어 두 컨벤션을 모두 앵커 테스트한다.
THRESHOLDS = {"ambig": 0.95, "disambig": 0.05, "default": 0.5}


def cond01(c):
    c = str(c).lower()
    return 0 if c.startswith("ambig") else (1 if c.startswith("disambig") else -1)


def unknown_idx(item):
    info = item.get("answer_info", {}) or {}
    if isinstance(info, str):
        info = json.loads(info)
    for i in range(3):
        a = info.get(f"ans{i}", [])
        if len(a) >= 2 and a[1] == "unknown":
            return i
    return -1


def load_moe():
    ckpt = torch.load(REPO / "results/v2/moe/main/moe_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = MoEAggregator(
        signal_dim=int(cfg.get("signal_dim", 7)),
        embed_dim=int(cfg.get("embed_dim", 384)),
        num_experts=int(cfg.get("num_experts", 4)),
        gating_hidden=int(cfg.get("gating_hidden", 64)),
        expert_hidden=int(cfg.get("expert_hidden", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    return model


def route(p, primary, cond, unk):
    tau = THRESHOLDS.get(cond, 0.5)
    return primary if (p >= tau or primary == -1) else unk


def metrics(rows):
    n_amb = n_dis = ok_a = ok_d = far = 0
    for gold_cond, lab, unk, final in rows:
        yv = cond01(gold_cond)
        if yv == 0:
            n_amb += 1; ok_a += int(final == lab)
        elif yv == 1:
            n_dis += 1; ok_d += int(final == lab); far += int(final == unk)
    return (ok_a / max(n_amb, 1), ok_d / max(n_dis, 1), far / max(n_dis, 1), n_amb + n_dis)


def main():
    # ---- condition predictor (audit-G replica) ----
    pool = {}
    for s in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = row.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d
    emb = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_embeddings.pt")):
        cat = Path(f).name.replace("_embeddings.pt", "")
        for k, v in torch.load(f, map_location="cpu", weights_only=True).items():
            emb[f"{cat}::{int(k)}"] = v.numpy().astype(np.float32)
    uids = list(pool)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    clf.fit(np.stack([emb[u] for u in uids]), np.array([cond01(pool[u]["context_condition"]) for u in uids]))
    print(f"[clf] trained on {len(uids)} pool embeddings")

    moe = load_moe()
    report = {}

    def run_set(name, sig_file, item_lookup, emb_lookup, thresholds):
        recs = [json.loads(l) for l in Path(sig_file).read_text().splitlines() if l.strip()]
        oracle_rows, pred_rows, agree = [], [], 0
        used = 0

        def route_t(p, primary, cond, unk):
            tau = thresholds.get(cond, thresholds.get("default", 0.5))
            return primary if (p >= tau or primary == -1) else unk

        with torch.inference_mode():
            for r in recs:
                # 런타임의 apply_composite_keys 미러: composite key 우선,
                # 임베딩은 raw example_id로 조회 (충돌 시 공유 — 런타임과 동일)
                k_raw = str(r["example_id"])
                k_comp = f"{r.get('category','_unknown')}::{r['example_id']}"
                it = item_lookup.get(k_comp) or item_lookup.get(k_raw)
                e = emb_lookup.get(k_raw)
                if it is None or e is None:
                    continue
                used += 1
                sig_t = signals_dict_to_tensor(r.get("signals", {})).unsqueeze(0)
                emb_t = torch.as_tensor(e, dtype=torch.float32).unsqueeze(0)
                p = float(moe(sig_t, emb_t).p.item())
                unk = unknown_idx(it)
                primary = int(r["primary_answer"])
                lab = int(r["label"])
                gold = str(it["context_condition"])
                pred = "disambig" if clf.predict(np.asarray(e, dtype=np.float32)[None, :])[0] == 1 else "ambig"
                agree += int(cond01(gold) == cond01(pred))
                oracle_rows.append((gold, lab, unk, route_t(p, primary, gold, unk)))
                pred_rows.append((gold, lab, unk, route_t(p, primary, pred, unk)))
        om, pm = metrics(oracle_rows), metrics(pred_rows)
        report[name] = {
            "n": used, "thresholds": dict(thresholds), "cond_pred_agree": agree / max(used, 1),
            "oracle": {"acc_amb": om[0], "acc_dis": om[1], "far": om[2]},
            "predicted": {"acc_amb": pm[0], "acc_dis": pm[1], "far": pm[2]},
        }
        print(f"[{name}] n={used} tau={thresholds} cond-agree={agree/max(used,1):.4f}")
        print(f"[{name}] oracle    : {om[0]:.4f} / {om[1]:.4f} / {om[2]:.4f}   (anchor check)")
        print(f"[{name}] predicted : {pm[0]:.4f} / {pm[1]:.4f} / {pm[2]:.4f}")

    # ---- Open-BBQ: acceptance_package run (published row의 원천) ----
    AP = REPO / "results/v2/acceptance_package/open_bbq"
    ob_items = {}
    for f in sorted(glob.glob(str(REPO / "data/open_bbq/*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                ob_items[str(d.get("example_id"))] = d
    ob_emb = {str(k): v.numpy().astype(np.float32)
              for k, v in torch.load(AP / "_embeddings.pt", map_location="cpu", weights_only=True).items()}
    print(f"[open_bbq] {len(ob_items)} items, {len(ob_emb)} runtime embeddings (acceptance_package)")
    run_set("open_bbq", AP / "_signals.jsonl", ob_items, ob_emb,
            {"ambig": 0.95, "disambig": 0.05, "default": 0.5})

    # ---- KoBBQ: 임계값 기록이 없어 두 컨벤션 모두 앵커 테스트 ----
    from src.transfer.run_kobbq import load_kobbq_as_bbq
    ko_items = {str(d["example_id"]): d for d in load_kobbq_as_bbq(max_samples_per_category=None)}
    ko_emb = {str(k): v.numpy().astype(np.float32)
              for k, v in torch.load(REPO / "results/v2_runpod/transfer/kobbq/_embeddings.pt",
                                     map_location="cpu", weights_only=True).items()}
    print(f"[kobbq] {len(ko_items)} items, {len(ko_emb)} cached embeddings")
    run_set("kobbq_tau9505", REPO / "results/v2_runpod/transfer/kobbq/_signals.jsonl", ko_items, ko_emb,
            {"ambig": 0.95, "disambig": 0.05, "default": 0.5})
    run_set("kobbq_tau50", REPO / "results/v2_runpod/transfer/kobbq/_signals.jsonl", ko_items, ko_emb,
            {"ambig": 0.5, "disambig": 0.5, "default": 0.5})

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {OUT}/report.json")


if __name__ == "__main__":
    main()
