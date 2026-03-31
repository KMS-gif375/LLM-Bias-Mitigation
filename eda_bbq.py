"""
BBQ 데이터셋 탐색적 데이터 분석 (EDA)
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("eda_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 법적 보호 특성 기반 선정 7개 카테고리
SELECTED_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Race_ethnicity", "Religion", "Sexual_orientation",
]

# =============================================================
# 데이터 로드
# =============================================================
def load_all_data():
    all_data = {}
    for cat in SELECTED_CATEGORIES:
        f = DATA_DIR / f"{cat}.jsonl"
        if not f.exists():
            print(f"  [경고] {f} 파일 없음, 건너뜀")
            continue
        with open(f, "r", encoding="utf-8") as fp:
            items = [json.loads(line) for line in fp if line.strip()]
        all_data[cat] = items
    return all_data

all_data = load_all_data()
all_items = []
for cat, items in all_data.items():
    for item in items:
        item["_category"] = cat
    all_items.extend(items)

df = pd.DataFrame(all_items)
print(f"총 {len(df)}개 항목 로드 완료")
print(f"카테고리: {list(all_data.keys())}")

# =============================================================
# 1. 카테고리별 데이터 수 분포
# =============================================================
fig, ax = plt.subplots(figsize=(14, 6))
cat_counts = df["_category"].value_counts().sort_values(ascending=True)
colors = sns.color_palette("viridis", len(cat_counts))
bars = ax.barh(cat_counts.index, cat_counts.values, color=colors)
for bar, val in zip(bars, cat_counts.values):
    ax.text(val + 100, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=10)
ax.set_xlabel("질문 수")
ax.set_title("1. 카테고리별 데이터 수 분포", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_category_distribution.png", dpi=150)
plt.close()
print("  [1/12] 카테고리별 데이터 수 분포")

# =============================================================
# 2. 맥락 조건 (ambig/disambig) 분포
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 전체
ctx_counts = df["context_condition"].value_counts()
axes[0].pie(ctx_counts.values, labels=["모호 (ambig)", "비모호 (disambig)"],
            autopct='%1.1f%%', colors=["#FF9999", "#66B2FF"], startangle=90,
            textprops={'fontsize': 12})
axes[0].set_title("전체 맥락 조건 비율", fontsize=12)

# 카테고리별
ctx_by_cat = df.groupby(["_category", "context_condition"]).size().unstack(fill_value=0)
ctx_by_cat.plot(kind="bar", ax=axes[1], color=["#FF9999", "#66B2FF"])
axes[1].set_xlabel("")
axes[1].set_ylabel("질문 수")
axes[1].set_title("카테고리별 맥락 조건 분포", fontsize=12)
axes[1].legend(["모호 (ambig)", "비모호 (disambig)"])
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle("2. 맥락 조건 (Ambig / Disambig) 분포", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_context_condition.png", dpi=150)
plt.close()
print("  [2/12] 맥락 조건 분포")

# =============================================================
# 3. 질문 극성 (neg/nonneg) 분포
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pol_counts = df["question_polarity"].value_counts()
axes[0].pie(pol_counts.values, labels=["부정적 (neg)", "비부정적 (nonneg)"],
            autopct='%1.1f%%', colors=["#FF6B6B", "#4ECDC4"], startangle=90,
            textprops={'fontsize': 12})
axes[0].set_title("전체 극성 비율", fontsize=12)

pol_by_cat = df.groupby(["_category", "question_polarity"]).size().unstack(fill_value=0)
pol_by_cat.plot(kind="bar", ax=axes[1], color=["#FF6B6B", "#4ECDC4"])
axes[1].set_xlabel("")
axes[1].set_ylabel("질문 수")
axes[1].set_title("카테고리별 극성 분포", fontsize=12)
axes[1].legend(["neg", "nonneg"])
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle("3. 질문 극성 (neg / nonneg) 분포", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_question_polarity.png", dpi=150)
plt.close()
print("  [3/12] 질문 극성 분포")

# =============================================================
# 4. 정답 위치 (label) 분포 - 위치 편향 확인용
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 전체 label 분포
label_counts = df["label"].value_counts().sort_index()
axes[0].bar(["(A) ans0", "(B) ans1", "(C) ans2"], label_counts.values,
            color=["#E74C3C", "#3498DB", "#2ECC71"])
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10)
axes[0].set_ylabel("질문 수")
axes[0].set_title("전체 정답 위치 분포", fontsize=12)

# 맥락별 label 분포
label_by_ctx = df.groupby(["context_condition", "label"]).size().unstack(fill_value=0)
label_by_ctx.plot(kind="bar", ax=axes[1], color=["#E74C3C", "#3498DB", "#2ECC71"])
axes[1].set_xlabel("")
axes[1].set_ylabel("질문 수")
axes[1].set_title("맥락별 정답 위치 분포", fontsize=12)
axes[1].legend(["(A) ans0", "(B) ans1", "(C) ans2"])
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle("4. 정답 위치 (Label) 분포 — 순환 순열 필요성 확인", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_label_distribution.png", dpi=150)
plt.close()
print("  [4/12] 정답 위치 분포")

# =============================================================
# 5. 맥락별 × 카테고리별 정답 위치 히트맵
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, ctx in enumerate(["ambig", "disambig"]):
    sub = df[df["context_condition"] == ctx]
    pivot = sub.groupby(["_category", "label"]).size().unstack(fill_value=0)
    # 비율로 변환
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.columns = ["(A)", "(B)", "(C)"]
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[idx],
                vmin=0, vmax=100, cbar_kws={'label': '%'})
    ctx_label = "모호" if ctx == "ambig" else "비모호"
    axes[idx].set_title(f"{ctx_label} 맥락", fontsize=12)
    axes[idx].set_ylabel("")

plt.suptitle("5. 카테고리 × 맥락별 정답 위치 비율 (%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_label_heatmap.png", dpi=150)
plt.close()
print("  [5/12] 정답 위치 히트맵")

# =============================================================
# 6. ans2 값 분석 (Unknown 고정 여부 확인)
# =============================================================
ans2_values = df["ans2"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
top_ans2 = ans2_values.head(10)
ax.barh(top_ans2.index[::-1], top_ans2.values[::-1], color=sns.color_palette("Set2", 10))
for i, (val, name) in enumerate(zip(top_ans2.values[::-1], top_ans2.index[::-1])):
    ax.text(val + 100, i, f'{val:,} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
ax.set_xlabel("질문 수")
ax.set_title("6. ans2 (선택지 C) 값 분포 — Unknown 고정 확인", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_ans2_values.png", dpi=150)
plt.close()
print("  [6/12] ans2 값 분석")

# =============================================================
# 7. 맥락 조건 × 극성 교차 분포
# =============================================================
fig, ax = plt.subplots(figsize=(12, 6))
cross = df.groupby(["_category", "context_condition", "question_polarity"]).size().reset_index(name="count")
cross["group"] = cross["context_condition"] + " / " + cross["question_polarity"]
pivot_cross = cross.pivot_table(index="_category", columns="group", values="count", fill_value=0)
pivot_cross.plot(kind="bar", ax=ax, color=["#FF6B6B", "#FFB347", "#4ECDC4", "#45B7D1"])
ax.set_xlabel("")
ax.set_ylabel("질문 수")
ax.set_title("7. 카테고리별 맥락 × 극성 교차 분포", fontsize=14, fontweight='bold')
ax.legend(title="맥락 / 극성", bbox_to_anchor=(1.02, 1))
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_context_polarity_cross.png", dpi=150)
plt.close()
print("  [7/12] 맥락×극성 교차 분포")

# =============================================================
# 8. 맥락 길이 분석
# =============================================================
df["context_len"] = df["context"].str.len()
df["question_len"] = df["question"].str.len()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 전체 context 길이 분포
axes[0, 0].hist(df["context_len"], bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(df["context_len"].mean(), color="red", linestyle="--", label=f'평균: {df["context_len"].mean():.0f}')
axes[0, 0].set_xlabel("문자 수")
axes[0, 0].set_ylabel("빈도")
axes[0, 0].set_title("맥락(Context) 길이 분포")
axes[0, 0].legend()

# 전체 question 길이 분포
axes[0, 1].hist(df["question_len"], bins=50, color="#2ECC71", edgecolor="white", alpha=0.8)
axes[0, 1].axvline(df["question_len"].mean(), color="red", linestyle="--", label=f'평균: {df["question_len"].mean():.0f}')
axes[0, 1].set_xlabel("문자 수")
axes[0, 1].set_ylabel("빈도")
axes[0, 1].set_title("질문(Question) 길이 분포")
axes[0, 1].legend()

# 카테고리별 context 길이 boxplot
cat_order = df.groupby("_category")["context_len"].median().sort_values().index
sns.boxplot(data=df, x="_category", y="context_len", order=cat_order, ax=axes[1, 0],
            palette="viridis", fliersize=2)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("문자 수")
axes[1, 0].set_title("카테고리별 맥락 길이")

# 맥락 조건별 context 길이
sns.boxplot(data=df, x="context_condition", y="context_len", ax=axes[1, 1],
            palette=["#FF9999", "#66B2FF"])
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("문자 수")
axes[1, 1].set_title("맥락 조건별 맥락 길이")
axes[1, 1].set_xticklabels(["모호 (ambig)", "비모호 (disambig)"])

plt.suptitle("8. 텍스트 길이 분석", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_text_length.png", dpi=150)
plt.close()
print("  [8/12] 텍스트 길이 분석")

# =============================================================
# 9. 고정관념 대상 그룹 분석
# =============================================================
stereo_groups = []
for _, row in df.iterrows():
    meta = row.get("additional_metadata", {})
    if isinstance(meta, dict):
        groups = meta.get("stereotyped_groups", [])
        if isinstance(groups, list):
            for g in groups:
                stereo_groups.append({"category": row["_category"], "group": g})

sg_df = pd.DataFrame(stereo_groups)
if len(sg_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 전체 top 20 고정관념 그룹
    top_groups = sg_df["group"].value_counts().head(20)
    axes[0].barh(top_groups.index[::-1], top_groups.values[::-1],
                 color=sns.color_palette("coolwarm", 20))
    axes[0].set_xlabel("빈도")
    axes[0].set_title("상위 20개 고정관념 대상 그룹", fontsize=12)

    # 카테고리별 고유 그룹 수
    unique_groups = sg_df.groupby("category")["group"].nunique().sort_values()
    axes[1].barh(unique_groups.index, unique_groups.values, color=sns.color_palette("viridis", len(unique_groups)))
    for i, v in enumerate(unique_groups.values):
        axes[1].text(v + 0.2, i, str(v), va='center', fontsize=10)
    axes[1].set_xlabel("고유 그룹 수")
    axes[1].set_title("카테고리별 고유 고정관념 그룹 수", fontsize=12)

    plt.suptitle("9. 고정관념 대상 그룹 분석", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_stereotyped_groups.png", dpi=150)
    plt.close()
    print("  [9/12] 고정관념 대상 그룹 분석")

# =============================================================
# 10. 카테고리별 고정관념 그룹 상세 히트맵
# =============================================================
if len(sg_df) > 0:
    # 카테고리별 상위 그룹
    top_cats = ["Age", "Gender_identity", "Race_ethnicity", "Religion", "Nationality"]
    fig, axes = plt.subplots(1, len(top_cats), figsize=(24, 6))
    for i, cat in enumerate(top_cats):
        cat_sg = sg_df[sg_df["category"] == cat]["group"].value_counts().head(8)
        axes[i].barh(cat_sg.index[::-1], cat_sg.values[::-1],
                     color=sns.color_palette("Set2", 8))
        axes[i].set_title(cat, fontsize=11, fontweight='bold')
        axes[i].set_xlabel("빈도")
    plt.suptitle("10. 주요 카테고리별 고정관념 대상 그룹 TOP 8", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_stereo_by_category.png", dpi=150)
    plt.close()
    print("  [10/12] 카테고리별 고정관념 그룹 상세")

# =============================================================
# 11. 질문 유형 분석 (question_index 분포)
# =============================================================
fig, ax = plt.subplots(figsize=(12, 6))
qi_by_cat = df.groupby("_category")["question_index"].nunique().sort_values()
ax.barh(qi_by_cat.index, qi_by_cat.values, color=sns.color_palette("magma", len(qi_by_cat)))
for i, v in enumerate(qi_by_cat.values):
    ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
ax.set_xlabel("고유 질문 템플릿 수")
ax.set_title("11. 카테고리별 고유 질문 템플릿 수 (question_index)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_question_templates.png", dpi=150)
plt.close()
print("  [11/12] 질문 템플릿 분석")

# =============================================================
# 12. 종합 요약 테이블
# =============================================================
summary_rows = []
for cat in sorted(all_data.keys()):
    items = all_data[cat]
    sub = df[df["_category"] == cat]
    ambig = sub[sub["context_condition"] == "ambig"]
    disambig = sub[sub["context_condition"] == "disambig"]

    # label 분포 (비모호에서)
    disambig_labels = disambig["label"].value_counts()

    summary_rows.append({
        "카테고리": cat,
        "전체": len(items),
        "모호": len(ambig),
        "비모호": len(disambig),
        "모호/비모호 비율": f"{len(ambig)/len(disambig):.2f}" if len(disambig) > 0 else "N/A",
        "neg": len(sub[sub["question_polarity"] == "neg"]),
        "nonneg": len(sub[sub["question_polarity"] == "nonneg"]),
        "neg 비율": f"{len(sub[sub['question_polarity']=='neg'])/len(sub)*100:.1f}%",
        "평균 맥락 길이": f"{sub['context_len'].mean():.0f}",
        "고유 템플릿 수": sub["question_index"].nunique(),
    })

summary_df = pd.DataFrame(summary_rows)

# 총합 행 추가
total_row = {
    "카테고리": "=== 총합 ===",
    "전체": len(df),
    "모호": len(df[df["context_condition"] == "ambig"]),
    "비모호": len(df[df["context_condition"] == "disambig"]),
    "모호/비모호 비율": f'{len(df[df["context_condition"]=="ambig"])/len(df[df["context_condition"]=="disambig"]):.2f}',
    "neg": len(df[df["question_polarity"] == "neg"]),
    "nonneg": len(df[df["question_polarity"] == "nonneg"]),
    "neg 비율": f"{len(df[df['question_polarity']=='neg'])/len(df)*100:.1f}%",
    "평균 맥락 길이": f"{df['context_len'].mean():.0f}",
    "고유 템플릿 수": df["question_index"].nunique(),
}
summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

# 테이블 시각화
fig, ax = plt.subplots(figsize=(20, 6))
ax.axis('off')
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# 헤더 스타일
for j in range(len(summary_df.columns)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
# 총합 행 스타일
last_row = len(summary_df)
for j in range(len(summary_df.columns)):
    table[last_row, j].set_facecolor('#D6E4F0')
    table[last_row, j].set_text_props(fontweight='bold')

plt.suptitle("12. BBQ 데이터셋 종합 요약", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_summary_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [12/12] 종합 요약 테이블")

# 콘솔에도 출력
print("\n" + "=" * 80)
print("BBQ 데이터셋 종합 요약")
print("=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)

# 인사이트 정리
print("\n" + "=" * 80)
print("EDA 인사이트 정리")
print("=" * 80)

# ans2 고정 여부
unknown_count = len(df[df["ans2"].str.lower().str.contains("can't be determined|cannot be determined|unknown|undetermined", na=False)])
print(f"\n[1] 선택지 (C) 고정 문제:")
print(f"    - ans2가 Unknown 계열인 비율: {unknown_count/len(df)*100:.1f}%")
print(f"    - 고유 ans2 값 수: {df['ans2'].nunique()}")
print(f"    → 순환 순열 적용이 필수적")

# 맥락 조건 균형
ambig_ratio = len(df[df["context_condition"] == "ambig"]) / len(df) * 100
print(f"\n[2] 맥락 조건 균형:")
print(f"    - 모호: {ambig_ratio:.1f}%, 비모호: {100-ambig_ratio:.1f}%")
print(f"    → {'균형 잡힘' if abs(ambig_ratio - 50) < 2 else '불균형 주의'}")

# 극성 균형
neg_ratio = len(df[df["question_polarity"] == "neg"]) / len(df) * 100
print(f"\n[3] 극성 균형:")
print(f"    - neg: {neg_ratio:.1f}%, nonneg: {100-neg_ratio:.1f}%")
print(f"    → {'균형 잡힘' if abs(neg_ratio - 50) < 2 else '불균형 주의'}")

# 카테고리 불균형
max_cat = cat_counts.max()
min_cat = cat_counts.min()
print(f"\n[4] 카테고리 간 데이터 불균형:")
print(f"    - 최대: {cat_counts.idxmax()} ({max_cat:,}개)")
print(f"    - 최소: {cat_counts.idxmin()} ({min_cat:,}개)")
print(f"    - 비율: {max_cat/min_cat:.1f}배 차이")
print(f"    → 카테고리별 성능 비교 시 정규화 필요")

# 맥락 길이
print(f"\n[5] 텍스트 길이:")
print(f"    - 맥락 평균: {df['context_len'].mean():.0f}자, 중앙값: {df['context_len'].median():.0f}자")
print(f"    - 질문 평균: {df['question_len'].mean():.0f}자")
ambig_len = df[df["context_condition"] == "ambig"]["context_len"].mean()
disambig_len = df[df["context_condition"] == "disambig"]["context_len"].mean()
print(f"    - 모호 맥락 평균: {ambig_len:.0f}자, 비모호 맥락 평균: {disambig_len:.0f}자")
print(f"    → 비모호 맥락이 {disambig_len/ambig_len:.1f}배 더 길음 (추가 정보 포함)")

# 정답 위치 분포
print(f"\n[6] 정답 위치 분포 (label):")
for label in [0, 1, 2]:
    cnt = len(df[df["label"] == label])
    print(f"    - label={label}: {cnt:,}개 ({cnt/len(df)*100:.1f}%)")
ambig_label2 = len(df[(df["context_condition"] == "ambig") & (df["label"] == 2)])
ambig_total = len(df[df["context_condition"] == "ambig"])
print(f"    - 모호 맥락에서 label=2 (Unknown 정답): {ambig_label2/ambig_total*100:.1f}%")
print(f"    → 모호 맥락에서 정답이 항상 (C)에 집중 → 순환 순열 필수")

print(f"\n시각화 결과: {OUTPUT_DIR.absolute()}/")
print("완료!")
