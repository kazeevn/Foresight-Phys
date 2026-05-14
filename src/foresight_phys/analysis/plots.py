"""Generate analysis figures, one PDF per figure under ``docs/analysis/plots/``.

The figures are split into a per-paper-macro view (``aggregate_paper_macro``,
``per_paper``, ``paper_dominance``) and a per-field-micro view
(``aggregate_per_field``, ``difficulty_distribution``, ``model_agreement``,
the numeric-threshold / scatter / CDF plots). The per-paper view is the one
that matches the per-run ``aggregate_*`` numbers and the public report.
"""
from __future__ import annotations

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

# Multi-threshold cutoffs (must stay in lockstep with analyze.NUMERIC_THRESHOLDS).
THRESHOLD_LABELS = ("within 0.1 dex", "within ×2", "within ×3", "within ×10")
THRESHOLD_CUTOFFS = (0.1, float(np.log10(2)), float(np.log10(3)), 1.0)


# ----------------------------------------------------------------------- helpers


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
    """Replicates ``analyze._paper_macro`` so the plots match the JSON summary."""
    valid = scored[scored[value_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    per_exp = valid.groupby(["model", "file_id", "experiment"])[value_col].mean()
    per_file = per_exp.groupby(["model", "file_id"]).mean()
    return per_file.groupby("model").mean()


def _model_order(scored: pd.DataFrame) -> list[str]:
    """Order models by paper-macro score ascending — strongest model rendered last."""
    macro = _paper_macro_score(scored, "score")
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
    """File with the most fields — useful for the 'one paper dominates' panel."""
    counts = scored.groupby("file_id").size()
    if counts.empty:
        return None
    return counts.idxmax()


# ------------------------------------------------------------------------ plots


def _aggregate_bars(
    scored: pd.DataFrame,
    *,
    title: str,
    weighting: str,
) -> plt.Figure:
    """Shared bar layout for the paper-macro and per-field-micro headline figures."""
    models = _model_order(scored)
    palette = _model_palette(models)

    if weighting == "paper-macro":
        score_macro = _paper_macro_score(scored, "score")
        norm_log = _paper_macro_score(
            scored.assign(
                _nls=np.where(scored["type"].isin(NUMERIC_TYPES),
                              scored["normalized_log_accuracy_score"], np.nan)
            ),
            "_nls",
        )
        disc = scored.assign(
            _correct=np.where(scored["type"].isin(DISC_TYPES),
                              scored["correct"].astype(float), np.nan)
        )
        disc_macro = _paper_macro_score(disc, "_correct")
        formula = scored.assign(
            _correct=np.where(scored["type"] == "formula",
                              scored["correct"].astype(float), np.nan)
        )
        formula_macro = _paper_macro_score(formula, "_correct")

        def get(series: pd.Series, m: str) -> float:
            return float(series.get(m, float("nan"))) if m in series.index else float("nan")

        metrics = {
            "Overall score\n(prediction_quality)": [get(score_macro, m) for m in models],
            "Numeric normalized\nlog-accuracy score": [get(norm_log, m) for m in models],
            "Bool / categorical\naccuracy": [get(disc_macro, m) for m in models],
            "Formula accuracy\n(judged)": [get(formula_macro, m) for m in models],
        }
    else:  # per-field micro
        metrics = {
            "Overall score\n(prediction_quality)": [
                scored.loc[scored.model == m, "score"].mean() for m in models
            ],
            "Numeric normalized\nlog-accuracy score": [
                scored.loc[
                    (scored.model == m) & scored["type"].isin(NUMERIC_TYPES),
                    "normalized_log_accuracy_score",
                ].mean()
                for m in models
            ],
            "Bool / categorical\naccuracy": [
                scored.loc[
                    (scored.model == m) & scored["type"].isin(DISC_TYPES),
                    "correct",
                ].mean()
                for m in models
            ],
            "Formula accuracy\n(judged)": [
                scored.loc[
                    (scored.model == m) & (scored["type"] == "formula"),
                    "correct",
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
    ax.set_ylabel("score / accuracy")
    ax.set_title(title)
    ax.legend(title="model", loc="upper right")
    fig.tight_layout()
    return fig


def fig_aggregate_paper_macro(scored: pd.DataFrame) -> plt.Figure:
    return _aggregate_bars(
        scored,
        title="Aggregate benchmark performance by model — paper-macro\n"
              "(one paper = one vote; matches per-run aggregate_*)",
        weighting="paper-macro",
    )


def fig_aggregate_per_field(scored: pd.DataFrame) -> plt.Figure:
    return _aggregate_bars(
        scored,
        title="Aggregate benchmark performance by model — per-field micro\n"
              "(one field = one vote; over-weights papers with many redundant fields)",
        weighting="per-field",
    )


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


def fig_thresholds(scored: pd.DataFrame) -> plt.Figure:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["numeric_gt"].notna()
        & scored["numeric_pred"].notna()
        & (scored["numeric_gt"] != 0)
        & (scored["numeric_pred"] != 0)
    ].copy()
    models = _model_order(scored)
    palette = _model_palette(models)
    gt = num.drop_duplicates(["file_id", "experiment", "key"])["numeric_gt"].to_numpy()
    baseline = float(np.exp(np.median(np.log(np.abs(gt))))) if len(gt) else 1.0

    # Add per-row indicator columns and paper-macro them, so the bars match
    # the JSON summary.
    bars: dict[str, list[float]] = {}
    for label, cutoff in zip(THRESHOLD_LABELS, THRESHOLD_CUTOFFS):
        col = f"_within_{label}"
        num[col] = (num["log_accuracy"] < cutoff).astype(float)
    for m in models:
        bars[m] = [
            float(_paper_macro_score(num[num.model == m], f"_within_{label}").get(m, np.nan))
            for label in THRESHOLD_LABELS
        ]

    base_df = num.drop_duplicates(["file_id", "experiment", "key"]).copy()
    base_df["model"] = "_baseline_"
    base_df["log_accuracy"] = np.abs(np.log10(np.abs(baseline / base_df["numeric_gt"])))
    for label, cutoff in zip(THRESHOLD_LABELS, THRESHOLD_CUTOFFS):
        base_df[f"_within_{label}"] = (base_df["log_accuracy"] < cutoff).astype(float)
    bars["const baseline"] = [
        float(_paper_macro_score(base_df, f"_within_{label}").iloc[0])
        for label in THRESHOLD_LABELS
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    keys = models + ["const baseline"]
    palette = {**palette, "const baseline": "grey"}
    x = np.arange(len(THRESHOLD_LABELS))
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, bars[k], width,
               label=_nice_label(k), color=palette[k])
    ax.set_xticks(x, list(THRESHOLD_LABELS))
    ax.set_ylim(0, 1)
    ax.set_ylabel("paper-macro fraction of numeric predictions")
    ax.set_title("Numeric prediction accuracy at common physics-intuition thresholds\n"
                 "(paper-macro of experiment-macro)")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def fig_log_ratio_cdf(scored: pd.DataFrame) -> plt.Figure:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["numeric_gt"].notna()
        & scored["numeric_pred"].notna()
        & (scored["numeric_gt"] != 0)
        & (scored["numeric_pred"] != 0)
    ]
    models = _model_order(scored)
    palette = _model_palette(models)
    gt = num.drop_duplicates(["file_id", "experiment", "key"])["numeric_gt"].to_numpy()
    baseline = float(np.exp(np.median(np.log(np.abs(gt))))) if len(gt) else 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        sub = num[num.model == m]
        r = np.sort(sub.log_accuracy.to_numpy())
        cdf = np.arange(1, len(r) + 1) / len(r)
        ax.plot(r, cdf, label=_nice_label(m), color=palette[m], lw=2)
    r = np.sort(np.abs(np.log10(np.abs(baseline / gt))))
    cdf = np.arange(1, len(r) + 1) / len(r)
    ax.plot(r, cdf, "--", color="grey", lw=1.5,
            label=f"constant baseline ({baseline:.2f})")
    for x_, label in (
        (np.log10(2), "factor of 2"),
        (np.log10(3), "factor of 3"),
        (1.0, "1 decade"),
    ):
        ax.axvline(x_, color="k", lw=0.6, ls=":")
        ax.text(x_ + 0.02, 0.04, label, rotation=90)
    ax.set_xlabel("|log10(prediction / ground truth)|")
    ax.set_ylabel("cumulative fraction of numeric predictions")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_title("How close are numeric predictions, on log scale?\n"
                 "(per-field micro CDF)")
    ax.legend(loc="lower right")
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


def fig_paper_dominance(scored: pd.DataFrame) -> plt.Figure:
    big = _largest_paper(scored)
    models = _model_order(scored)
    palette = _model_palette(models)
    if big is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no paper data", ha="center", va="center")
        ax.axis("off")
        return fig
    n_models = max(scored["model"].nunique(), 1)
    n_total = int(scored.groupby("file_id").size().sum() / n_models)
    n_big = int((scored.file_id == big).sum() / n_models)

    macro_all = _paper_macro_score(scored, "score")
    macro_wo = _paper_macro_score(scored[scored.file_id != big], "score")
    micro_all = scored.groupby("model")["score"].mean()
    micro_wo = scored[scored.file_id != big].groupby("model")["score"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(models))
    width = 0.4
    for ax, (label, all_series, wo_series) in zip(
        axes,
        [
            ("paper-macro (one paper = one vote)", macro_all, macro_wo),
            ("per-field micro (one field = one vote)", micro_all, micro_wo),
        ],
    ):
        ax.bar(x - width / 2,
               [all_series.get(m, np.nan) for m in models],
               width, color=[palette[m] for m in models],
               label=f"all {n_total} fields / 20 papers")
        ax.bar(x + width / 2,
               [wo_series.get(m, np.nan) for m in models],
               width, color=[palette[m] for m in models],
               hatch="//", edgecolor="white",
               label=f"excluding {big}")
        for i, m in enumerate(models):
            for off, val in ((-width / 2, all_series.get(m, np.nan)),
                             (width / 2, wo_series.get(m, np.nan))):
                if np.isfinite(val):
                    ax.text(i + off, val + 0.01, f"{val:.2f}",
                            ha="center", fontsize=8)
        ax.set_xticks(x, [_nice_label(m) for m in models])
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("mean field score")
    fig.suptitle(f"How much does {big} ({n_big} fields) move the headline?")
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


def fig_per_paper(scored: pd.DataFrame) -> plt.Figure:
    """Per-paper (file_id) view of paper-macro scores per model."""
    models = _model_order(scored)
    palette = _model_palette(models)

    # Per (model, file): experiment-mean of score, then file-level.
    valid = scored[scored["score"].notna()]
    per_exp = valid.groupby(["model", "file_id", "experiment"])["score"].mean()
    per_file = per_exp.groupby(["model", "file_id"]).mean().unstack("model")[models]
    per_file["_mean"] = per_file.mean(axis=1)
    per_file = per_file.sort_values("_mean").drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(per_file)), 5))
    x = np.arange(len(per_file))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, per_file[m].values, width,
               label=_nice_label(m), color=palette[m])
    ax.set_xticks(x, per_file.index, rotation=60, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("paper-level mean score (experiment-macro within paper)")
    ax.set_title("Per-paper performance, sorted by cross-model mean\n"
                 "(each bar = one paper's experiment-macro score for one model)")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- driver

PlotFactory = Callable[..., plt.Figure]


def _figure_specs() -> list[tuple[str, PlotFactory, tuple[str, ...]]]:
    """(filename stem, builder, required-arg names)."""
    return [
        # Per-paper-macro view (matches summary.json / per-run aggregates).
        ("aggregate_paper_macro", fig_aggregate_paper_macro, ("scored",)),
        ("per_paper", fig_per_paper, ("scored",)),
        ("paper_dominance", fig_paper_dominance, ("scored",)),
        ("numeric_thresholds", fig_thresholds, ("scored",)),
        # Per-field-micro view (one field = one vote; useful for difficulty
        # bucketing and sanity-check CDFs but should not be the headline).
        ("aggregate_per_field", fig_aggregate_per_field, ("scored",)),
        ("difficulty_distribution", fig_difficulty, ("wide",)),
        ("numeric_log_ratio_cdf", fig_log_ratio_cdf, ("scored",)),
        ("numeric_scatter", fig_numeric_scatter, ("scored",)),
        ("model_agreement", fig_model_agreement, ("wide",)),
    ]


def write_all_plots(paths: AnalysisPaths | None = None) -> list[Path]:
    paths = paths or default_paths()
    paths.ensure_dirs()

    _setup_matplotlib()
    scored = pd.read_parquet(paths.scored_parquet)
    wide = pd.read_parquet(paths.per_field_parquet)
    context = {"scored": scored, "wide": wide}

    written: list[Path] = []
    for stem, builder, arg_names in _figure_specs():
        fig = builder(*(context[name] for name in arg_names))
        out = paths.plots_dir / f"{stem}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written
