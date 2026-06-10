#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_r2_audits.py — CPU audits for review round-2 hardening.

(A) TEMPLATE-DISJOINT split: condition accuracy + condition-only retention when
    train/test share NO (category, question_index) template.
(B) Acc_amb=1.0000 rescue accounting: per-seed ambig->disambig classifier errors
    and what the retention rule then did (from test_predictions.jsonl).
(C) Seed test-set overlap fractions (pairwise).
(D) s2 default-to-1.0 quantification: instances without exactly two non-unknown
    groups, and swap no-ops where neither group string matches the context.
(E) Retained-side keeps: wrong stereotyped (76) vs anti-stereotyped (65) by
    category + binomial test.
(F) Implicit-cue per-category FAR (transfer MoE variant, from predictions.csv).
(G) Condition-only retention rows for transfer sets (EN classifier on MiniLM
    embeddings; KoBBQ uses cached embeddings).
(H) High-risk label precision against audit outcomes (explanations.jsonl).
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402

SEEDS = [42, 123, 456, 789, 999]
OUT = REPO / "results/v2/reviewer_audits/r2_audits"
OUT.mkdir(parents=True, exist_ok=True)


def cond01(c):
    c = str(c).lower()
    return 0 if c.startswith("ambig") else (1 if c.startswith("disambig") else -1)


def unknown_idx(item):
    info = item.get("answer_info", {}) or {}
    for i in range(3):
        a = info.get(f"ans{i}", [])
        if len(a) >= 2 and a[1] == "unknown":
            return i
    return -1


def load_pool():
    pool = {}
    for s in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = row.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d
    return pool


def load_primary():
    out = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_signals.jsonl")):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                try:
                    out[f"{r['category']}::{int(r['example_id'])}"] = int(r["primary_answer"])
                except (TypeError, ValueError, KeyError):
                    pass
    return out


def load_emb():
    emb = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_embeddings.pt")):
        cat = Path(f).name.replace("_embeddings.pt", "")
        for k, v in torch.load(f, map_location="cpu", weights_only=True).items():
            emb[f"{cat}::{int(k)}"] = v.numpy().astype(np.float32)
    return emb


def retention_metrics(uids, pool, prim, pred_dis):
    """Condition-only rule: predicted ambig -> unknown; else keep primary."""
    n_amb = n_dis = ok_amb = ok_dis = far = 0
    for u, pd_ in zip(uids, pred_dis):
        it = pool[u]; y = cond01(it["context_condition"]); lab = int(it["label"])
        unk = unknown_idx(it); p = prim.get(u, -1)
        final = p if pd_ else unk
        if y == 0:
            n_amb += 1; ok_amb += int(final == lab)
        else:
            n_dis += 1; ok_dis += int(final == lab); far += int(final == unk)
    return (ok_amb / max(n_amb, 1), ok_dis / max(n_dis, 1), far / max(n_dis, 1), n_amb, n_dis)


def main():
    pool = load_pool(); prim = load_primary(); emb = load_emb()
    print(f"[load] pool={len(pool)} prim={len(prim)} emb={len(emb)}")
    report = {}

    # ---------------- (A) template-disjoint ----------------
    uids = list(pool)
    groups = np.array([f"{pool[u]['category']}::{pool[u].get('question_index')}" for u in uids])
    X = np.stack([emb[u] for u in uids]); y = np.array([cond01(pool[u]["context_condition"]) for u in uids])
    print(f"[A] templates: {len(set(groups))} groups over {len(uids)} instances")
    accs, rets = [], []
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te]); accs.append(float((pred == y[te]).mean()))
        te_uids = [uids[i] for i in te]
        a, d, f, na, nd = retention_metrics(te_uids, pool, prim, pred.astype(bool))
        rets.append((a, d, f))
        print(f"  seed {seed}: cond_acc={accs[-1]:.4f}  amb={a:.4f} dis={d:.4f} far={f:.4f} (n={na}/{nd})")
    A = {"cond_acc_mean": float(np.mean(accs)), "cond_acc_std": float(np.std(accs)),
         "acc_amb_mean": float(np.mean([r[0] for r in rets])), "acc_amb_std": float(np.std([r[0] for r in rets])),
         "acc_dis_mean": float(np.mean([r[1] for r in rets])), "acc_dis_std": float(np.std([r[1] for r in rets])),
         "far_mean": float(np.mean([r[2] for r in rets])), "far_std": float(np.std([r[2] for r in rets])),
         "n_templates": int(len(set(groups)))}
    report["A_template_disjoint"] = A
    print(f"[A] DISJOINT: cond {A['cond_acc_mean']:.4f}±{A['cond_acc_std']:.4f} | "
          f"amb {A['acc_amb_mean']:.4f} dis {A['acc_dis_mean']:.4f} far {A['far_mean']:.4f}")

    # ---------------- (B) rescue accounting ----------------
    B = []
    for seed in SEEDS:
        rows = [json.loads(l) for l in (REPO / f"results/v2/clean_experiments/seed_{seed}/test_predictions.jsonl").read_text().splitlines() if l.strip()]
        mis = [r for r in rows if str(r["context_condition"]).startswith("ambig")
               and str(r.get("predicted_condition", "")).startswith("disambig")]
        resc = sum(1 for r in mis if int(r.get("ours_predicted_condition", -9)) == int(r["label"]))
        B.append({"seed": seed, "amb_misclassified": len(mis), "kept_but_correct": resc})
        print(f"[B] seed {seed}: ambig->disambig errors={len(mis)}, of which final answer still correct={resc}")
    report["B_rescue"] = B

    # ---------------- (C) seed overlap ----------------
    tsets = {s: set(json.loads(l)["uid"] for l in (REPO / f"results/v2/clean_experiments/seed_{s}/test_predictions.jsonl").read_text().splitlines() if l.strip()) for s in SEEDS}
    ovl = [len(tsets[a] & tsets[b]) / 1328 for a, b in combinations(SEEDS, 2)]
    report["C_overlap"] = {"mean_pairwise_frac": float(np.mean(ovl)), "min": float(min(ovl)), "max": float(max(ovl))}
    print(f"[C] pairwise test overlap: mean {np.mean(ovl):.3f} (min {min(ovl):.3f}, max {max(ovl):.3f})")

    # ---------------- (D) s2 defaults ----------------
    def bounded(text, ctx):
        esc = re.escape(text)
        pre = r"\b" if text and text[0].isalnum() else ""
        suf = r"\b" if text and text[-1].isalnum() else ""
        return re.search(f"{pre}{esc}{suf}", ctx, re.IGNORECASE) is not None
    n_no2, n_noop = 0, 0
    per_cat = defaultdict(lambda: [0, 0, 0])
    for u, it in pool.items():
        info = it.get("answer_info", {}) or {}
        grp = [(i, str(it.get(f"ans{i}", ""))) for i in range(3)
               if len(info.get(f"ans{i}", [])) >= 2 and info[f"ans{i}"][1] != "unknown"]
        c = it["category"]; per_cat[c][2] += 1
        if len(grp) != 2:
            n_no2 += 1; per_cat[c][0] += 1; continue
        ctx = str(it.get("context", ""))
        if not (bounded(grp[0][1], ctx) or bounded(grp[1][1], ctx)):
            n_noop += 1; per_cat[c][1] += 1
    report["D_s2"] = {"not_two_groups": n_no2, "swap_noop_no_match": n_noop, "n_pool": len(pool),
                      "frac_default_or_noop": (n_no2 + n_noop) / len(pool),
                      "per_category": {c: {"not2": v[0], "noop": v[1], "n": v[2]} for c, v in sorted(per_cat.items())}}
    print(f"[D] s2: not-two-groups={n_no2} ({n_no2/len(pool):.2%}), swap-noop={n_noop} ({n_noop/len(pool):.2%})")

    # ---------------- (E) retained keeps by category ----------------
    ex = [json.loads(l) for l in (REPO / "results/v2/rule_explanations/ours_predicted_condition/explanations.jsonl").read_text().splitlines() if l.strip()]
    st = [e for e in ex if e["decision_type"] == "wrong_stereotyped_keep"]
    an = [e for e in ex if e["decision_type"] == "wrong_anti_stereotyped_keep"]
    bc = Counter(e["category"] for e in st); ba = Counter(e["category"] for e in an)
    p_all = binomtest(len(st), len(st) + len(an), 0.5).pvalue
    worst = sorted(set(bc) | set(ba), key=lambda c: -(bc[c] + ba[c]))[:3]
    E = {"stereo": len(st), "anti": len(an), "binom_p_two_sided": float(p_all),
         "by_category": {c: {"stereo": bc.get(c, 0), "anti": ba.get(c, 0)} for c in sorted(set(bc) | set(ba))},
         "worst3_p": {c: float(binomtest(bc.get(c, 0), bc.get(c, 0) + ba.get(c, 0), 0.5).pvalue) if bc.get(c,0)+ba.get(c,0) else None for c in worst}}
    report["E_keeps"] = E
    print(f"[E] keeps: stereo {len(st)} vs anti {len(an)}  binom p={p_all:.3f}  worst3={ {c:(bc.get(c,0),ba.get(c,0)) for c in worst} }")

    # ---------------- (F) implicit per-category FAR (MoE transfer variant) ----------------
    impl = pd.read_csv(REPO / "results/v2_runpod/transfer/implicit_bbq/predictions.csv")
    # unknown index from the implicit source data
    src_files = []
    for cand in ("data/implicit_bbq_generated_v2", "data/implicit_bbq_v2_300", "data/implicit_bbq"):
        src_files = sorted(glob.glob(str(REPO / cand / "*.jsonl")))
        if src_files:
            print(f"[F] implicit source dir: {cand} ({len(src_files)} files)")
            break
    unk_map = {}
    for f in src_files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                ai = d.get("answer_info", {})
                if isinstance(ai, str):
                    ai = json.loads(ai)
                d2 = {"answer_info": ai}
                unk_map[str(d.get("example_id"))] = unknown_idx(d2)
    fcat = {}
    for c, sub in impl[impl.context_condition == "disambig"].groupby("category"):
        unks = sub["example_id"].astype(str).map(unk_map)
        far = float((sub["final_answer"].astype(float) == unks.astype(float)).mean()) if len(sub) else float("nan")
        fcat[c] = {"far": far, "n": int(len(sub))}
    spread = max(v["far"] for v in fcat.values()) - min(v["far"] for v in fcat.values())
    report["F_implicit_far"] = {"per_category": fcat, "spread": float(spread), "n_src": len(unk_map)}
    print("[F] implicit per-cat FAR:", {c: round(v['far'], 3) for c, v in sorted(fcat.items())}, f"spread={spread:.3f}")

    # ---------------- (G) condition-only transfer rows ----------------
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    clf_en = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    clf_en.fit(np.stack([emb[u] for u in uids]), y)

    def transfer_condonly(name, sig_file, text_lookup=None, emb_pt=None):
        recs = [json.loads(l) for l in Path(sig_file).read_text().splitlines() if l.strip()]
        ok = []
        embs = {}
        if emb_pt and Path(emb_pt).exists():
            embs = {str(k): v.numpy().astype(np.float32) for k, v in torch.load(emb_pt, map_location="cpu", weights_only=True).items()}
        texts, keys = [], []
        for r in recs:
            k = str(r["example_id"])
            if k in embs:
                continue
            t = text_lookup.get(k) if text_lookup else None
            if t:
                texts.append(t); keys.append(k)
        if texts:
            V = st_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False)
            for k, v in zip(keys, V):
                embs[k] = np.asarray(v, dtype=np.float32)
        n_amb = n_dis = ok_amb = ok_dis = far = 0
        for r in recs:
            k = str(r["example_id"])
            if k not in embs:
                continue
            pdis = bool(clf_en.predict(embs[k][None, :])[0])
            yv = cond01(r["context_condition"]); lab = int(r["label"]); p = int(r["primary_answer"])
            unk = transfer_unks[name].get(k, -1)
            final = p if pdis else unk
            if yv == 0:
                n_amb += 1; ok_amb += int(final == lab)
            elif yv == 1:
                n_dis += 1; ok_dis += int(final == lab); far += int(final == unk)
        return {"acc_amb": ok_amb / max(n_amb, 1), "acc_dis": ok_dis / max(n_dis, 1),
                "far": far / max(n_dis, 1), "n": n_amb + n_dis}

    # text + unknown lookups for transfer sets
    transfer_unks = defaultdict(dict)
    text_lk = defaultdict(dict)
    for f in sorted(glob.glob(str(REPO / "data/open_bbq/*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                ai = d.get("answer_info", {}); ai = json.loads(ai) if isinstance(ai, str) else ai
                k = str(d.get("example_id"))
                transfer_unks["open_bbq"][k] = unknown_idx({"answer_info": ai})
                text_lk["open_bbq"][k] = f"{d.get('context','')} {d.get('question','')}"
    for f in src_files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                ai = d.get("answer_info", {}); ai = json.loads(ai) if isinstance(ai, str) else ai
                k = str(d.get("example_id"))
                transfer_unks["implicit"][k] = unknown_idx({"answer_info": ai})
                text_lk["implicit"][k] = f"{d.get('context','')} {d.get('question','')}"
    # kobbq: embeddings cached; unknown idx from its signals? need answer_info -> reload via loader
    try:
        from src.transfer.run_kobbq import load_kobbq_as_bbq
        for d in load_kobbq_as_bbq(max_samples_per_category=None):
            transfer_unks["kobbq"][str(d.get("example_id"))] = unknown_idx(d)
    except Exception as e:
        print(f"[G] kobbq unknown-idx load failed: {e}")

    G = {}
    G["open_bbq"] = transfer_condonly("open_bbq", REPO / "results/v2_runpod/transfer/open_bbq/_signals.jsonl", text_lk["open_bbq"])
    G["implicit"] = transfer_condonly("implicit", REPO / "results/v2_runpod/transfer/implicit_bbq/_signals.jsonl", text_lk["implicit"])
    G["kobbq"] = transfer_condonly("kobbq", REPO / "results/v2_runpod/transfer/kobbq/_signals.jsonl", None, REPO / "results/v2_runpod/transfer/kobbq/_embeddings.pt")
    report["G_condonly_transfer"] = G
    for k, v in G.items():
        print(f"[G] cond-only {k}: amb={v['acc_amb']:.4f} dis={v['acc_dis']:.4f} far={v['far']:.4f} (n={v['n']})")

    # ---------------- (H) high-risk precision ----------------
    hi = [e for e in ex if e.get("bias_risk_level") == "high"]
    block_types = {"stereotyped_raw_answer_blocked", "anti_stereotyped_unsupported_answer_blocked",
                   "ambiguous_abstention"}
    bad_kept = {"stereotype_bias_slip", "anti_stereotype_slip", "wrong_stereotyped_keep", "wrong_anti_stereotyped_keep"}
    tp = sum(1 for e in hi if e["decision_type"] in block_types | bad_kept)
    dist = Counter(e["decision_type"] for e in hi)
    report["H_highrisk"] = {"n_high": len(hi), "aligned_with_block_or_slip": tp,
                            "precision_proxy": tp / max(len(hi), 1), "decision_dist": dict(dist)}
    print(f"[H] high-risk n={len(hi)}, aligned={tp} ({tp/max(len(hi),1):.3f})  dist={dict(dist)}")

    (OUT / "r2_audit_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {OUT}/r2_audit_report.json")


if __name__ == "__main__":
    main()
