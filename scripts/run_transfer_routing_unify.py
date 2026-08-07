#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_transfer_routing_unify.py — reviewer request: unify Table 6 routing.

The original transfer runs route the per-condition override by the dataset's
GOLD condition label (oracle routing, an inherited convention). This script
recomputes the SAME seven-signal MoE rows with PREDICTED-condition routing
(the deployable no-oracle convention) for the two transfer sets whose archived
model inputs can be reconstructed without new LLM inference:

  * Open-BBQ : archived signals and MiniLM embeddings from the acceptance
    package, with item metadata in data/open_bbq.
  * KoBBQ    : archived signals and cached MiniLM embeddings; item metadata is
    loaded from the pinned KoBBQ revision used by src/transfer/run_kobbq.py.

Protocol (faithful to the finalized Open-BBQ run):
  - public MoE checkpoint results/v2_runpod/moe/main/moe_best.pt, thresholds ambig=0.95 and
    disambig=0.05 (default fallback 0.5), override rule identical to
    src/models/override.apply_per_condition_override.
  - Condition predictor: balanced LogisticRegression (random_state=42) on the
    full English-BBQ pool MiniLM embeddings — the audit-G construction.
  - ORACLE PROVENANCE DIAGNOSTIC: the script emits gold-condition rows beside
    predicted-condition rows for manual comparison. It does not enforce a
    numerical fidelity assertion.

ImplicitBBQ-style is excluded: the paraphrased texts/embeddings for the full
2,640-example pod run were not retained, so no-oracle routing cannot be
reconstructed (already disclosed in the Table 6 caption).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from collections import defaultdict
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

OUT = REPO / "results/v2/reviewer_audits/routing_unify_published"
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_moe(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
    parser = argparse.ArgumentParser(
        description="Replay Open-BBQ/KoBBQ routing with an explicit MoE checkpoint."
    )
    parser.add_argument(
        "--moe-checkpoint",
        type=Path,
        default=REPO / "results/v2_runpod/moe/main/moe_best.pt",
        help="Checkpoint used for every replayed retention score.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    checkpoint = args.moe_checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

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
    source_text_pairs = {
        (str(item.get("context", "")), str(item.get("question", "")))
        for item in pool.values()
    }
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    clf.fit(np.stack([emb[u] for u in uids]), np.array([cond01(pool[u]["context_condition"]) for u in uids]))
    print(f"[clf] trained on {len(uids)} pool embeddings")

    moe = load_moe(checkpoint)
    report = {
        "_provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "thresholds": dict(THRESHOLDS),
        }
    }

    def run_set(name, sig_file, item_lookup, emb_lookup, thresholds):
        recs = [json.loads(l) for l in Path(sig_file).read_text().splitlines() if l.strip()]
        oracle_rows, pred_rows, condition_rows, agree = [], [], [], 0
        oracle_nonoverlap, pred_nonoverlap, condition_nonoverlap = [], [], []
        overlap_amb = overlap_dis = 0
        gate_sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(moe.num_experts, dtype=float))
        gate_counts: dict[str, int] = defaultdict(int)
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
                moe_out = moe(sig_t, emb_t)
                p = float(moe_out.p.item())
                category = str(r.get("category", it.get("category", "_unknown")))
                gate_sums[category] += moe_out.gate_w.squeeze(0).cpu().numpy().astype(float)
                gate_counts[category] += 1
                unk = unknown_idx(it)
                primary = int(r["primary_answer"])
                lab = int(r["label"])
                gold = str(it["context_condition"])
                pred = "disambig" if clf.predict(np.asarray(e, dtype=np.float32)[None, :])[0] == 1 else "ambig"
                agree += int(cond01(gold) == cond01(pred))
                oracle_rows.append((gold, lab, unk, route_t(p, primary, gold, unk)))
                pred_rows.append((gold, lab, unk, route_t(p, primary, pred, unk)))
                condition_rows.append((gold, lab, unk, unk if pred == "ambig" else primary))
                if name == "open_bbq":
                    overlaps_source = (
                        str(it.get("context", "")), str(it.get("question", ""))
                    ) in source_text_pairs
                    if overlaps_source:
                        if cond01(gold) == 0:
                            overlap_amb += 1
                        else:
                            overlap_dis += 1
                    else:
                        oracle_nonoverlap.append(oracle_rows[-1])
                        pred_nonoverlap.append(pred_rows[-1])
                        condition_nonoverlap.append(condition_rows[-1])
        om, pm = metrics(oracle_rows), metrics(pred_rows)
        cm = metrics(condition_rows)
        report[name] = {
            "n": used, "thresholds": dict(thresholds), "cond_pred_agree": agree / max(used, 1),
            "oracle": {"acc_amb": om[0], "acc_dis": om[1], "far": om[2]},
            "predicted": {"acc_amb": pm[0], "acc_dis": pm[1], "far": pm[2]},
            "condition_only_predicted": {
                "acc_amb": cm[0], "acc_dis": cm[1], "far": cm[2]
            },
            "mean_gate_weights_by_category": {
                cat: (gate_sums[cat] / gate_counts[cat]).tolist()
                for cat in sorted(gate_sums)
            },
        }
        if name == "open_bbq":
            onm, pnm, cnm = (
                metrics(oracle_nonoverlap), metrics(pred_nonoverlap),
                metrics(condition_nonoverlap),
            )
            report[name]["source_text_overlap_audit"] = {
                "matching_rule": "exact (context, question) string pair",
                "source_item_n": len(pool),
                "source_unique_text_pair_n": len(source_text_pairs),
                "overlap_n": overlap_amb + overlap_dis,
                "overlap_ambiguous_n": overlap_amb,
                "overlap_disambiguated_n": overlap_dis,
                "nonoverlap_n": len(pred_nonoverlap),
                "nonoverlap_ambiguous_n": sum(
                    cond01(row[0]) == 0 for row in pred_nonoverlap
                ),
                "nonoverlap_disambiguated_n": sum(
                    cond01(row[0]) == 1 for row in pred_nonoverlap
                ),
                "nonoverlap_oracle": {
                    "acc_amb": onm[0], "acc_dis": onm[1], "far": onm[2]
                },
                "nonoverlap_predicted_moe": {
                    "acc_amb": pnm[0], "acc_dis": pnm[1], "far": pnm[2]
                },
                "nonoverlap_condition_only_predicted": {
                    "acc_amb": cnm[0], "acc_dis": cnm[1], "far": cnm[2]
                },
            }
            with (out_dir / "open_bbq_cluster_routing.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["category"] + [f"Expert {k + 1}" for k in range(moe.num_experts)])
                for cat in sorted(gate_sums):
                    writer.writerow([cat] + list(gate_sums[cat] / gate_counts[cat]))
        print(f"[{name}] n={used} tau={thresholds} cond-agree={agree/max(used,1):.4f}")
        print(f"[{name}] oracle    : {om[0]:.4f} / {om[1]:.4f} / {om[2]:.4f}   (anchor check)")
        print(f"[{name}] predicted : {pm[0]:.4f} / {pm[1]:.4f} / {pm[2]:.4f}")
        print(f"[{name}] cond-only : {cm[0]:.4f} / {cm[1]:.4f} / {cm[2]:.4f}")
        if name == "open_bbq":
            print(
                f"[{name}] exact source-text overlap: "
                f"{overlap_amb + overlap_dis}/{used} "
                f"(amb={overlap_amb}, dis={overlap_dis}); "
                f"non-overlap predicted MoE: "
                f"{pnm[0]:.4f} / {pnm[1]:.4f} / {pnm[2]:.4f}; "
                f"condition-only: {cnm[0]:.4f} / {cnm[1]:.4f} / {cnm[2]:.4f}"
            )

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

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {out_dir}/report.json")


if __name__ == "__main__":
    main()
