#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
recompute_bias_direction.py — re-derive every published stereotype-DIRECTION
number from saved predictions using the FIXED is_stereotyped_answer matcher
(group-tag normalization + answer-text fallback; see bbq_evaluator._group_matches).

Outputs (results/v2/reviewer_audits/bias_direction_fix/):
  corrected_taxonomy.json   — blocked / keeps / slips splits + per-seed residual
                              stereo/anti counts + residual score, with old values
  corrected_composite.json  — composite baseline re-parsed with the fixed
                              extract_letter, per-seed test metrics + paired
                              bootstrap vs the MoE predicted-condition variant
  corrected_kobbq_bias.json — KoBBQ transfer bias scores under the fixed converter
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402
from src.evaluation.bbq_evaluator import is_stereotyped_answer  # noqa: E402  (FIXED version)
from src.baselines.composite_prompting import extract_letter  # noqa: E402  (FIXED version)
from src.evaluation.bbq_evaluator import parse_prediction  # noqa: E402

OUT = REPO / "results/v2/reviewer_audits/bias_direction_fix"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 999]


def load_pool():
    pool = {}
    for s in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = row.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d
    return pool


def main():
    pool = load_pool()
    report = {}

    # ============ 1. taxonomy / residuals from explanations.jsonl ============
    ex = [json.loads(l) for l in (REPO / "results/v2/rule_explanations/ours_predicted_condition/explanations.jsonl").read_text().splitlines() if l.strip()]
    print(f"[tax] {len(ex)} decisions")

    def direction(e, ans_idx):
        it = pool.get(e["uid"])
        if it is None or ans_idx not in (0, 1, 2):
            return None
        return is_stereotyped_answer(it, int(ans_idx))

    old = Counter(e["decision_type"] for e in ex)
    new = Counter()
    resid = defaultdict(lambda: [0, 0])  # seed -> [stereo, anti]
    keeps_cat = defaultdict(lambda: [0, 0])
    def bucket(d, stereo_key, anti_key):
        # None/'unknown' 방향(판단 불가)은 anti로 떨어뜨리지 않고 별도 집계
        if d == "stereotyped":
            new[stereo_key] += 1
        elif d == "anti_stereotyped":
            new[anti_key] += 1
        else:
            new["undetermined_direction"] += 1
        return d

    for e in ex:
        dt = e["decision_type"]
        if dt in ("stereotyped_answer_blocked", "anti_stereotyped_answer_blocked"):
            bucket(direction(e, e["primary_answer"]),
                   "stereotyped_answer_blocked", "anti_stereotyped_answer_blocked")
        elif dt in ("wrong_stereotyped_keep", "wrong_anti_stereotyped_keep"):
            d = bucket(direction(e, e["final_answer"]),
                       "wrong_stereotyped_keep", "wrong_anti_stereotyped_keep")
            if d in ("stereotyped", "anti_stereotyped"):
                keeps_cat[e["category"]][0 if d == "stereotyped" else 1] += 1
        elif dt in ("bias_slip", "anti_stereotype_slip"):
            bucket(direction(e, e["final_answer"]), "bias_slip", "anti_stereotype_slip")
        else:
            new[dt] += 1
        # residual: ambiguous rows whose FINAL answer is non-unknown
        if str(e.get("context_condition", "")).startswith("ambig") and e.get("final_answer") not in (None, e.get("unknown_index")):
            try:
                fa = int(e["final_answer"])
            except (TypeError, ValueError):
                fa = -1
            if fa in (0, 1, 2) and fa != int(e.get("unknown_index", -9)):
                d = direction(e, fa)
                if d in ("stereotyped", "anti_stereotyped"):
                    resid[e["seed"]][0 if d == "stereotyped" else 1] += 1

    st, an = new["wrong_stereotyped_keep"], new["wrong_anti_stereotyped_keep"]
    p_keeps = binomtest(st, st + an, 0.5).pvalue if (st + an) else None
    scores = []
    for s in SEEDS:
        a, b = resid[s]
        if a + b > 0:
            scores.append(2 * a / (a + b) - 1)
    report["taxonomy"] = {
        "old": dict(old), "new": dict(new),
        "keeps_binom_p": float(p_keeps) if p_keeps is not None else None,
        "keeps_by_category": {c: {"stereo": v[0], "anti": v[1]} for c, v in sorted(keeps_cat.items())},
        "residual_per_seed": {s: {"stereo": resid[s][0], "anti": resid[s][1]} for s in SEEDS},
        "residual_score_mean": float(np.mean(scores)) if scores else None,
        "residual_score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
        "residual_n_seeds_nonzero": len(scores),
    }
    print("[tax] old:", {k: old[k] for k in sorted(old)})
    print("[tax] new:", {k: new[k] for k in sorted(new)})
    print(f"[tax] keeps {st}:{an} p={p_keeps:.3f} | residual score {np.mean(scores):.4f}±{np.std(scores, ddof=1):.4f} ({len(scores)} seeds)")

    # ============ 2. composite re-parse + per-seed metrics + paired bootstrap ============
    comp = [json.loads(l) for l in (REPO / "results/v2_runpod/baselines/composite/predictions.jsonl").read_text().splitlines() if l.strip()]
    cmap = {}
    for r in comp:
        uid = f"{r['category']}::{int(r['example_id'])}"
        cmap[uid] = parse_prediction(extract_letter(r["raw_response"]))
    n_unparsed = sum(1 for v in cmap.values() if v not in (0, 1, 2))
    print(f"[comp] reparsed {len(cmap)} composite predictions "
          f"({n_unparsed} unparseable -> scored wrong, standard convention)")

    rows_m, deltas_all = [], {m: [] for m in ("amb", "dis", "far")}
    rng_p = {m: [] for m in ("amb", "dis", "far")}
    for seed in SEEDS:
        tp = [json.loads(l) for l in (REPO / f"results/v2/clean_experiments/seed_{seed}/test_predictions.jsonl").read_text().splitlines() if l.strip()]
        uids = [r["uid"] for r in tp if r["uid"] in cmap]
        ours = {r["uid"]: int(r["ours_predicted_condition"]) for r in tp}
        y = np.array([0 if str(pool[u]["context_condition"]).startswith("ambig") else 1 for u in uids])
        lab = np.array([int(pool[u]["label"]) for u in uids])
        unk = np.array([next(i for i in range(3) if pool[u]["answer_info"][f"ans{i}"][1] == "unknown") for u in uids])
        cp = np.array([cmap[u] for u in uids])
        op = np.array([ours[u] for u in uids])

        def metr(pred):
            amb = y == 0; dis = y == 1
            return (float((pred[amb] == lab[amb]).mean()), float((pred[dis] == lab[dis]).mean()),
                    float((pred[dis] == unk[dis]).mean()))

        mc, mo = metr(cp), metr(op)
        rows_m.append({"seed": seed, "comp_amb": mc[0], "comp_dis": mc[1], "comp_far": mc[2], "n": len(uids)})
        print(f"[comp] seed {seed}: composite {mc[0]:.4f}/{mc[1]:.4f}/{mc[2]:.4f} (n={len(uids)})  ours {mo[0]:.4f}/{mo[1]:.4f}/{mo[2]:.4f}")

        rng = np.random.default_rng(seed)
        n = len(uids); B = 10000
        dl = np.zeros((B, 3))
        for b in range(B):
            idx = rng.integers(0, n, n)
            yb, lb, ub, cb, ob = y[idx], lab[idx], unk[idx], cp[idx], op[idx]
            a, d2 = yb == 0, yb == 1
            dl[b, 0] = (ob[a] == lb[a]).mean() - (cb[a] == lb[a]).mean()
            dl[b, 1] = (ob[d2] == lb[d2]).mean() - (cb[d2] == lb[d2]).mean()
            dl[b, 2] = (ob[d2] == ub[d2]).mean() - (cb[d2] == ub[d2]).mean()
        for mi, m in enumerate(("amb", "dis", "far")):
            obs = (mo[mi] - mc[mi])
            deltas_all[m].append(obs)
            p = 2 * min((dl[:, mi] <= 0).mean(), (dl[:, mi] >= 0).mean())
            # Monte Carlo p는 0이 될 수 없음 — 1/B로 하한
            rng_p[m].append(min(1.0, max(float(p), 1.0 / B)))

    agg = lambda k: (float(np.mean([r[k] for r in rows_m])), float(np.std([r[k] for r in rows_m], ddof=1)))
    report["composite"] = {
        "n_unparseable_of_8864": n_unparsed,
        "per_seed": rows_m,
        "row_mean_std": {"acc_amb": agg("comp_amb"), "acc_dis": agg("comp_dis"), "far": agg("comp_far")},
        "paired_deltas_mean": {m: float(np.mean(v)) for m, v in deltas_all.items()},
        "paired_p_max": {m: max(v) for m, v in rng_p.items()},
    }
    print(f"[comp] NEW row: {agg('comp_amb')[0]:.4f}±{agg('comp_amb')[1]:.4f} / "
          f"{agg('comp_dis')[0]:.4f}±{agg('comp_dis')[1]:.4f} / {agg('comp_far')[0]:.4f}±{agg('comp_far')[1]:.4f}")
    print(f"[comp] paired deltas (ours-composite): {report['composite']['paired_deltas_mean']}  p_max={report['composite']['paired_p_max']}")

    # ============ 3. KoBBQ corrected bias scores (fixed converter) ============
    try:
        from src.transfer.run_kobbq import load_kobbq_as_bbq
        krecs = {str(d["example_id"]): d for d in load_kobbq_as_bbq(max_samples_per_category=None)}
        ksig = [json.loads(l) for l in (REPO / "results/v2_runpod/transfer/kobbq/_signals.jsonl").read_text().splitlines() if l.strip()]
        cnt = {"ambig": [0, 0], "disambig": [0, 0]}
        for r in ksig:
            it = krecs.get(str(r["example_id"]))
            if it is None:
                continue
            p = int(r["primary_answer"])
            d = is_stereotyped_answer(it, p)
            if d in ("stereotyped", "anti_stereotyped"):
                cnt[it["context_condition"]][0 if d == "stereotyped" else 1] += 1
        kb = {c: (2 * v[0] / (v[0] + v[1]) - 1 if (v[0] + v[1]) else None, v) for c, v in cnt.items()}
        report["kobbq_bias_corrected"] = {c: {"score": kb[c][0], "stereo_anti": kb[c][1]} for c in kb}
        print(f"[kobbq] corrected bias: ambig={kb['ambig'][0]:.3f} {kb['ambig'][1]}, disambig={kb['disambig'][0]:.3f} {kb['disambig'][1]}")
    except Exception as e:
        print(f"[kobbq] skipped: {e}")

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {OUT}/report.json")


if __name__ == "__main__":
    main()
