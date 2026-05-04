"""
Phase 5: Ablation 실험 모듈.

서브모듈:
    - signal_ablation:       7개 신호별 leave-one-out
    - sae_ablation:          SAE Top-K / Layer / 식별 방법 비교
    - cluster_ablation:      MoE expert 수(K)와 cluster taxonomy 비교
    - loco_ablation:         Leave-One-Category-Out 7-fold CV
    - visualization:         논문용 figure 5종 (PDF)
    - qualitative_analysis:  Top SAE feature / bias-head / failure case
"""

from src.ablation.signal_ablation import (
    SIGNAL_NAMES,
    SignalAblationResult,
    SignalAblationSummary,
    run_signal_ablation,
)
from src.ablation.sae_ablation import (
    IDENTIFICATION_FNS,
    SAEAblationConfig,
    SAEAblationResult,
    SAEAblationSummary,
    run_sae_ablation,
)
from src.ablation.cluster_ablation import (
    DEFAULT_TAXONOMY,
    TAXONOMIES,
    ClusterAblationConfig,
    ClusterAblationResult,
    ClusterAblationSummary,
    num_experts_in_taxonomy,
    run_cluster_ablation,
)
from src.ablation.loco_ablation import (
    LOCOAblationSummary,
    LOCOFoldResult,
    run_loco_ablation,
)
from src.ablation.visualization import (
    plot_baseline_comparison_bar,
    plot_bias_head_heatmap,
    plot_cluster_routing_heatmap,
    plot_risk_coverage_curve,
    plot_sae_feature_activation,
    render_all,
    save_pdf,
    set_paper_style,
)
from src.ablation.qualitative_analysis import (
    BiasHeadExample,
    FailureAnalysis,
    FailureCase,
    FeatureExample,
    failure_cases,
    save_qualitative_analysis,
    top_bias_head_attention_examples,
    top_sae_max_activating_examples,
)

__all__ = [
    # signal
    "SIGNAL_NAMES",
    "SignalAblationResult",
    "SignalAblationSummary",
    "run_signal_ablation",
    # sae
    "IDENTIFICATION_FNS",
    "SAEAblationConfig",
    "SAEAblationResult",
    "SAEAblationSummary",
    "run_sae_ablation",
    # cluster
    "DEFAULT_TAXONOMY",
    "TAXONOMIES",
    "ClusterAblationConfig",
    "ClusterAblationResult",
    "ClusterAblationSummary",
    "num_experts_in_taxonomy",
    "run_cluster_ablation",
    # loco
    "LOCOAblationSummary",
    "LOCOFoldResult",
    "run_loco_ablation",
    # viz
    "plot_baseline_comparison_bar",
    "plot_bias_head_heatmap",
    "plot_cluster_routing_heatmap",
    "plot_risk_coverage_curve",
    "plot_sae_feature_activation",
    "render_all",
    "save_pdf",
    "set_paper_style",
    # qualitative
    "BiasHeadExample",
    "FailureAnalysis",
    "FailureCase",
    "FeatureExample",
    "failure_cases",
    "save_qualitative_analysis",
    "top_bias_head_attention_examples",
    "top_sae_max_activating_examples",
]
