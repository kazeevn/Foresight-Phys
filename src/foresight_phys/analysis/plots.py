"""Generate analysis figures, one PDF per figure under ``docs/analysis/plots/``.

After the move to forecasts-with-uncertainty, numeric figures focus on
calibration (CDF of |z|, coverage bars) rather than raw point-prediction
log-accuracy.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
DISC_TYPES = {"bool", "categorical"}

DIFFICULTY_ORDER = (
    "intractable (none correct)",
    "hard (1 right)",
    "discriminative",
    "easy (1 wrong)",
    "trivial (all correct)",
)
DIFFICULTY_COLORS = ("#b3261e", "#e8842a", "#e6c200", "#83b14f", "#0b7d2b")

COVERAGE_LABELS = ("|z| < 1σ", "|z| < 2σ", "|z| < 3σ")
COVERAGE_CUTOFFS = (1.0, 2.0, 3.0)


def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def _paper_macro_score(scored: pd.DataFrame, value_col: str) -> pd.Series:
    valid = scored[scored[value_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    per_exp = valid.groupby(["model", "file_id", "experiment"])[value_col].mean()
    per_file = per_exp.groupby(["model", "file_id"]).mean()
    return per_file.groupby("model").mean()


def _model_order(scored: pd.DataFrame) -> list[str]:
    macro = _paper_macro_score(scored, "quality")
    if macro.empty:
        return sorted(scored["model"].unique())
    return macro.sort_values().index.tolist()


def _model_palette(models: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("Blues")
    n = len(models)
    if n <= 1:
        return {models[0]: cmap(0.7)} if models else {}
    return {m: cmap(0.3 + 0.6 * i / (n - 1)) for i, m in enumerate(models)}


def _nice_label(model_name: str) -> str:
    return model_name.replace("gpt-", "")


def _largest_paper(scored: pd.DataFrame) -> str | None:
    counts = scored.groupby("file_id").size()
    if counts.empty:
        return None
    return counts.idxmax()


def _aggregate_bars(
    scored: pd.DataFrame,
    *,
    title: str,
    weighting: str,
) -> plt.Figure:
    models = _model_order(scored)
    palette = _model_palette(models)

    if weighting == "paper-macro":
        quality_macro = _paper_macro_score(scored, "quality")
        numeric_quality_macro = _paper_macro_score(
            scored.assign(
                _nq=np.where(scored["type"].isin(NUMERIC_TYPES), scored["quality"], np.nan)
            ),
            "_nq",
        )
        disc_quality_macro = _paper_macro_score(
            scored.assign(
                _dq=np.where(scored["type"].isin(DISC_TYPES), scored["quality"], np.nan)
            ),
            "_dq",
        )
        formula_quality_macro = _paper_macro_score(
            scored.assign(
                _fq=np.where(scored["type"] == "formula", scored["quality"], np.nan)
            ),
            "_fq",
        )

        def get(series: pd.Series, m: str) -> float:
            return float(series.get(m, float("nan"))) if m in series.index else float("nan")

        metrics = {
            "Overall\nquality": [get(quality_macro, m) for m in models],
            "Numeric\nquality": [get(numeric_quality_macro, m) for m in models],
            "Bool/cat.\nquality": [get(disc_quality_macro, m) for m in models],
            "Formula\nquality": [get(formula_quality_macro, m) for m in models],
        }
    else:  # per-field micro
        metrics = {
            "Overall\nquality": [
                scored.loc[scored.model == m, "quality"].mean() for m in models
            ],
            "Numeric\nquality": [
                scored.loc[
                    (scored.model == m) & scored["type"].isin(NUMERIC_TYPES), "quality"
                ].mean()
                for m in models
            ],
            "Bool/cat.\nquality": [
                scored.loc[
                    (scored.model == m) & scored["type"].isin(DISC_TYPES), "quality"
                ].mean()
                for m in models
            ],
            "Formula\nquality": [
                scored.loc[
                    (scored.model == m) & (scored["type"] == "formula"), "quality"
                ].mean()
                for m in models
            ],
        }

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [metrics[k][i] for k in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
               label=_nice_label(m), color=palette[m])
    ax.set_xticks(x, list(metrics.keys()))
    ax.set_ylim(0, 1)
    ax.set_ylabel("quality (proper-scoring-rule)")
    ax.set_title(title)
    ax.legend(title="model", loc="upper right")
    fig.tight_layout()
    return fig


def fig_aggregate_paper_macro(scored: pd.DataFrame) -> plt.Figure:
    return _aggregate_bars(
        scored,
        title="Aggregate quality by model — paper-macro\n"
              "(one paper = one vote; matches per-run aggregate_*)",
        weighting="paper-macro",
    )


def fig_aggregate_per_field(scored: pd.DataFrame) -> plt.Figure:
    return _aggregate_bars(
        scored,
        title="Aggregate quality by model — per-field micro\n"
              "(one field = one vote; over-weights papers with many redundant fields)",
        weighting="per-field",
    )


def fig_coverage_bars(scored: pd.DataFrame) -> plt.Figure:
    """Per-model fraction of numeric predictions with |z| inside each σ bucket."""
    num = scored[scored["type"].isin(NUMERIC_TYPES) & scored["z"].notna()].copy()
    num["abs_z"] = num["z"].abs()
    models = _model_order(scored)
    palette = _model_palette(models)

    bars: dict[str, list[float]] = {}
    for label, cutoff in zip(COVERAGE_LABELS, COVERAGE_CUTOFFS):
        col = f"_within_{label}"
        num[col] = (num["abs_z"] < cutoff).astype(float)
    for m in models:
        bars[m] = [
            float(_paper_macro_score(num[num.model == m], f"_within_{label}").get(m, np.nan))
            for label in COVERAGE_LABELS
        ]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(COVERAGE_LABELS))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, bars[m], width,
               label=_nice_label(m), color=palette[m])
    # Ideal calibration reference: Gaussian fractions inside 1, 2, 3 σ.
    ideal = [math.erf(c / math.sqrt(2)) for c in COVERAGE_CUTOFFS]
    for cutoff, value in zip(x, ideal):
        ax.hlines(value, cutoff - 0.4, cutoff + 0.4, colors="k", linestyles="--",
                  linewidth=1.0, label="_nolegend_")
    ax.set_xticks(x, list(COVERAGE_LABELS))
    ax.set_ylim(0, 1)
    ax.set_ylabel("paper-macro fraction inside σ bucket")
    ax.set_title("Numeric calibration: coverage vs. ideal Gaussian (dashed)")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def fig_abs_z_cdf(scored: pd.DataFrame) -> plt.Figure:
    """CDF of |z| per model vs. the ideal half-normal CDF."""
    num = scored[scored["type"].isin(NUMERIC_TYPES) & scored["z"].notna()]
    models = _model_order(scored)
    palette = _model_palette(models)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        sub = num[num.model == m]
        r = np.sort(np.abs(sub.z.to_numpy()))
        if len(r) == 0:
            continue
        cdf = np.arange(1, len(r) + 1) / len(r)
        ax.plot(r, cdf, label=_nice_label(m), color=palette[m], lw=2)
    # Ideal: half-normal CDF = erf(z / sqrt(2))
    zs = np.linspace(0, 4, 200)
    ax.plot(zs, np.array([math.erf(z / math.sqrt(2)) for z in zs]), "--",
            color="grey", lw=1.5, label="ideal (Gaussian)")
    for x_, label in (
        (1.0, "1σ"),
        (2.0, "2σ"),
        (3.0, "3σ"),
    ):
        ax.axvline(x_, color="k", lw=0.6, ls=":")
        ax.text(x_ + 0.02, 0.04, label, rotation=90)
    ax.set_xlabel("|z| = |gt − pred| / σ (linear or dex, per chosen distribution)")
    ax.set_ylabel("cumulative fraction of numeric predictions")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration: CDF of |z| vs. ideal Gaussian (dashed)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_reliability(scored: pd.DataFrame) -> plt.Figure:
    """Reliability diagram for bool predictions (and categorical top-class)."""
    df = scored[scored["type"].isin(DISC_TYPES) & scored["prob_true"].notna()].copy()
    if df.empty:
        # Fall back to using probabilities for categorical via parquet json.
        df = scored[scored["type"].isin(DISC_TYPES) & scored["quality"].notna()].copy()
        # Approximate p as max(probabilities) for categorical; for bool use 1-quality...
        # Keep it simple: only plot when prob_true present.
    models = _model_order(scored)
    palette = _model_palette(models)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.0, label="perfect")
    bins = np.linspace(0, 1, 11)
    for m in models:
        sub = df[df.model == m]
        if sub.empty:
            continue
        # Group bool predictions: prob_true bucket -> mean correctness.
        sub = sub.copy()
        sub["bin"] = np.clip(np.digitize(sub["prob_true"], bins) - 1, 0, len(bins) - 2)
        agg = sub.groupby("bin").agg(
            mean_p=("prob_true", "mean"),
            mean_correct=("correct", "mean"),
            n=("correct", "size"),
        )
        ax.plot(agg["mean_p"], agg["mean_correct"], "o-",
                color=palette[m], label=f"{_nice_label(m)} (n={int(sub.shape[0])})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical fraction correct")
    ax.set_title("Reliability diagram (boolean predictions)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_crps_bars(scored: pd.DataFrame) -> plt.Figure:
    """Per-model mean CRPS / Brier across numeric / discrete / formula fields."""
    models = _model_order(scored)
    palette = _model_palette(models)

    def macro(mask: pd.Series, value_col: str) -> dict[str, float]:
        sub = scored[mask].copy()
        if sub.empty:
            return {m: float("nan") for m in models}
        macro_series = _paper_macro_score(sub, value_col)
        return {
            m: (
                float(macro_series.get(m, float("nan")))
                if m in macro_series.index
                else float("nan")
            )
            for m in models
        }

    crps_col = "crps_scaled" if "crps_scaled" in scored.columns else "crps"
    crps_label = "Numeric\nrel. CRPS" if crps_col == "crps_scaled" else "Numeric\nCRPS"
    crps = macro(scored["type"].isin(NUMERIC_TYPES), crps_col)
    bool_br = macro(scored["type"] == "bool", "brier")
    cat_br = macro(scored["type"] == "categorical", "brier")
    form_br = macro(scored["type"] == "formula", "brier")

    metrics = {
        crps_label: [crps[m] for m in models],
        "Bool\nBrier": [bool_br[m] for m in models],
        "Categorical\nBrier": [cat_br[m] for m in models],
        "Formula\nBrier": [form_br[m] for m in models],
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [metrics[k][i] for k in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
               label=_nice_label(m), color=palette[m])
    ax.set_xticks(x, list(metrics.keys()))
    ax.set_ylabel("mean (lower is better)")
    ax.set_title("Per-model proper scoring rules — paper-macro of experiment-macro")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_difficulty(wide: pd.DataFrame) -> plt.Figure:
    counts = wide["difficulty"].value_counts().reindex(DIFFICULTY_ORDER).fillna(0)
    by_type = (
        wide.groupby(["type", "difficulty"]).size().unstack(fill_value=0)
        .reindex(columns=DIFFICULTY_ORDER, fill_value=0)
    )
    by_type_pct = by_type.div(by_type.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 2]})

    axes[0].barh(DIFFICULTY_ORDER, counts.values, color=DIFFICULTY_COLORS)
    total = max(int(counts.sum()), 1)
    for i, v in enumerate(counts.values):
        axes[0].text(v + max(total * 0.01, 1), i,
                     f"{int(v)} ({v / total * 100:.0f}%)", va="center", fontsize=9)
    axes[0].set_title(f"All {total} fields (per-field counts)")
    axes[0].set_xlabel("# fields")
    axes[0].invert_yaxis()

    by_type_pct.plot.barh(stacked=True, color=DIFFICULTY_COLORS,
                          ax=axes[1], legend=False)
    axes[1].set_xlabel("share of fields of that type")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Difficulty composition by result type")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Per-field difficulty across the models (per-field micro view)", y=1.02)
    fig.tight_layout()
    return fig


def fig_numeric_scatter(scored: pd.DataFrame) -> plt.Figure:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["numeric_gt"].notna()
        & scored["numeric_pred"].notna()
        & (scored["numeric_gt"] != 0)
        & (scored["numeric_pred"] != 0)
    ].assign(
        abs_gt=lambda d: d.numeric_gt.abs(),
        abs_pred=lambda d: d.numeric_pred.abs(),
    )
    models = _model_order(scored)
    palette = _model_palette(models)
    big = _largest_paper(scored)

    n = len(models)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, m in zip(flat, models):
        sub = num[num.model == m]
        if big is not None:
            cms_mask = sub.file_id == big
            ax.scatter(sub.loc[~cms_mask, "abs_gt"], sub.loc[~cms_mask, "abs_pred"],
                       s=12, alpha=0.55, color=palette[m], label="other papers")
            ax.scatter(sub.loc[cms_mask, "abs_gt"], sub.loc[cms_mask, "abs_pred"],
                       s=10, alpha=0.5, color="#c0392b",
                       label=f"{big} (largest)")
        else:
            ax.scatter(sub["abs_gt"], sub["abs_pred"], s=12, alpha=0.55,
                       color=palette[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        lo, hi = 1e-4, 1e8
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
        ax.plot([lo, hi], [lo * 10, hi * 10], "k:", lw=0.6, alpha=0.5)
        ax.plot([lo, hi], [lo / 10, hi / 10], "k:", lw=0.6, alpha=0.5)
        ax.set_title(f"{_nice_label(m)} (n = {len(sub)})")
        ax.legend(loc="lower right", fontsize=8)
    for ax in flat[len(models):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("|ground truth|")
    for ax in axes[:, 0]:
        ax.set_ylabel("|prediction|")
    fig.suptitle("Numeric predictions vs. ground truth (log–log; dashed = perfect, dotted = ±1 decade)")
    fig.tight_layout()
    return fig


def fig_per_paper(scored: pd.DataFrame) -> plt.Figure:
    models = _model_order(scored)
    palette = _model_palette(models)

    valid = scored[scored["quality"].notna()]
    per_exp = valid.groupby(["model", "file_id", "experiment"])["quality"].mean()
    per_file = per_exp.groupby(["model", "file_id"]).mean().unstack("model")[models]
    per_file["_mean"] = per_file.mean(axis=1)
    per_file = per_file.sort_values("_mean").drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(per_file)), 5))
    x = np.arange(len(per_file))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, per_file[m].values, width,
               label=_nice_label(m), color=palette[m])
    ax.set_xticks(x, per_file.index, rotation=60, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("paper-level mean quality (experiment-macro within paper)")
    ax.set_title("Per-paper quality, sorted by cross-model mean")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_model_agreement(wide: pd.DataFrame) -> plt.Figure:
    models = sorted(
        c.removeprefix("correct__")
        for c in wide.columns if c.startswith("correct__")
    )
    n = len(models)
    correct = {m: wide[f"correct__{m}"].astype(bool).to_numpy() for m in models}
    mat = np.zeros((n, n))
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            a, b = correct[mi], correct[mj]
            mat[i, j] = (a & b).sum() / max((a | b).sum(), 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.55 else "black", fontsize=10)
    ax.set_xticks(range(n), [_nice_label(m) for m in models])
    ax.set_yticks(range(n), [_nice_label(m) for m in models])
    ax.set_title("Jaccard agreement on correctly-answered fields\n"
                 "(per-field micro; |A∩B| / |A∪B|)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _decision_per_model(decisions: pd.DataFrame, scored: pd.DataFrame) -> tuple[list[str], dict[str, dict]]:
    from .decisions import summarize_decisions

    summary = summarize_decisions(decisions)
    by_model = {row["model"]: row for row in summary["per_model"]}
    models = [m for m in _model_order(scored) if m in by_model]
    return models, by_model


def fig_decision_quality(decisions: pd.DataFrame, scored: pd.DataFrame) -> plt.Figure:
    """Per-model decision quality: did the forecaster make the right call?"""
    models, by_model = _decision_per_model(decisions, scored)
    palette = _model_palette(models)

    def get(model: str, key: str) -> float:
        v = by_model.get(model, {}).get(key)
        return float(v) if v is not None else float("nan")

    metrics = {
        "Decision\naccuracy": [get(m, "decision_accuracy_paper_macro") for m in models],
        "Selection\naccuracy": [get(m, "selection_accuracy") for m in models],
        "Direction\naccuracy": [get(m, "direction_accuracy") for m in models],
        "OOM\nwithin ×3": [get(m, "oom_set_factor3") for m in models],
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [metrics[k][i] for k in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
               label=_nice_label(m), color=palette[m])
    ax.set_xticks(x, list(metrics.keys()))
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of decisions correct")
    ax.set_title("Decision usefulness: choosing / direction / order-of-magnitude\n"
                 "(from annotated comparison sets; paper-macro for the headline)")
    ax.legend(title="model", loc="upper right")
    fig.tight_layout()
    return fig


def fig_decision_regret(decisions: pd.DataFrame, scored: pd.DataFrame) -> plt.Figure:
    """Selection regret of the forecast-driven policy vs. a no-model baseline.

    Normalised regret in [0, 1]; 0 is the oracle pick. The gap between the
    no-model (random-pick) baseline and the model is the decision value added.
    """
    models, by_model = _decision_per_model(decisions, scored)
    palette = _model_palette(models)

    model_regret = [by_model.get(m, {}).get("mean_regret") for m in models]
    base_regret = [by_model.get(m, {}).get("baseline_regret") for m in models]
    model_regret = [float(v) if v is not None else float("nan") for v in model_regret]
    base_regret = [float(v) if v is not None else float("nan") for v in base_regret]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(models))
    width = 0.38
    ax.bar(x - width / 2, base_regret, width, label="no-model baseline (random pick)",
           color="#b0b0b0")
    for i, m in enumerate(models):
        ax.bar(x[i] + width / 2, model_regret[i], width,
               color=palette[m], label="forecast-driven" if i == 0 else "_nolegend_")
    ax.set_xticks(x, [_nice_label(m) for m in models])
    ax.set_ylabel("normalised selection regret (lower is better)")
    ax.set_title("Experiment-selection regret: forecast-driven vs. no-model baseline\n"
                 "(gap = decision value added by the forecaster)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_foresight_lift(full_scored: pd.DataFrame) -> plt.Figure:
    """Full-context vs. name-only quality: how much the experiment description adds."""
    macro = _paper_macro_score(full_scored, "quality")
    base_models = [m for m in macro.index if not m.endswith("[name-only]")]
    base_models = [m for m in base_models if f"{m} [name-only]" in macro.index]
    base_models = sorted(base_models, key=lambda m: macro.get(m, 0.0))

    full_vals = [float(macro.get(m, float("nan"))) for m in base_models]
    name_vals = [float(macro.get(f"{m} [name-only]", float("nan"))) for m in base_models]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(base_models))
    width = 0.38
    ax.bar(x - width / 2, name_vals, width, label="name-only (typed key prior)", color="#b0b0b0")
    ax.bar(x + width / 2, full_vals, width, label="full experiment context", color="#0b7d2b")
    for xi, lo, hi in zip(x, name_vals, full_vals):
        if math.isfinite(lo) and math.isfinite(hi):
            ax.annotate(f"+{hi - lo:.3f}", (xi, max(hi, lo) + 0.01), ha="center", fontsize=9)
    ax.set_xticks(x, [_nice_label(m) for m in base_models])
    ax.set_ylim(0, 1)
    ax.set_ylabel("paper-macro quality")
    ax.set_title("Foresight lift: predictive value of the experiment description\n"
                 "(gap over the name-only prior = real physical foresight)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


PlotFactory = Callable[..., plt.Figure]


def _figure_specs() -> list[tuple[str, PlotFactory, tuple[str, ...]]]:
    return [
        ("aggregate_paper_macro", fig_aggregate_paper_macro, ("scored",)),
        ("per_paper", fig_per_paper, ("scored",)),
        ("coverage_bars", fig_coverage_bars, ("scored",)),
        ("abs_z_cdf", fig_abs_z_cdf, ("scored",)),
        ("reliability", fig_reliability, ("scored",)),
        ("crps_bars", fig_crps_bars, ("scored",)),
        ("aggregate_per_field", fig_aggregate_per_field, ("scored",)),
        ("difficulty_distribution", fig_difficulty, ("wide",)),
        ("numeric_scatter", fig_numeric_scatter, ("scored",)),
        ("model_agreement", fig_model_agreement, ("wide",)),
        ("decision_quality", fig_decision_quality, ("decisions", "scored")),
        ("decision_regret", fig_decision_regret, ("decisions", "scored")),
        ("foresight_lift", fig_foresight_lift, ("ablation",)),
    ]


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    return False


def write_all_plots(paths: AnalysisPaths | None = None) -> list[Path]:
    paths = paths or default_paths()
    paths.ensure_dirs()

    _setup_matplotlib()
    full_scored = pd.read_parquet(paths.scored_parquet)
    if "ablation" in full_scored.columns:
        scored = full_scored[full_scored["ablation"] == "none"].copy()
        ablation = full_scored if (full_scored["ablation"] == "name-only").any() else pd.DataFrame()
    else:
        scored = full_scored
        ablation = pd.DataFrame()
    wide = pd.read_parquet(paths.per_field_parquet)
    decisions = (
        pd.read_parquet(paths.decisions_parquet)
        if paths.decisions_parquet.exists()
        else pd.DataFrame()
    )
    context = {"scored": scored, "wide": wide, "decisions": decisions, "ablation": ablation}

    written: list[Path] = []
    for stem, builder, arg_names in _figure_specs():
        if any(_is_empty(context[name]) for name in arg_names):
            continue
        fig = builder(*(context[name] for name in arg_names))
        out = paths.plots_dir / f"{stem}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written
