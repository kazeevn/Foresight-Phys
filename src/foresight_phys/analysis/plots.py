"""Generate analysis figures, one PDF per figure under ``docs/analysis/plots/``."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

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


def _model_order(scored: pd.DataFrame) -> list[str]:
    """Stable ordering: by mean per-field score ascending, so the strongest is last."""
    return (
        scored.groupby("model")["score"].mean()
        .sort_values().index.tolist()
    )


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


def fig_aggregate(scored: pd.DataFrame) -> plt.Figure:
    models = _model_order(scored)
    palette = _model_palette(models)
    metrics = {
        "Overall score\n(prediction_quality)": [
            scored.loc[scored.model == m, "score"].mean() for m in models
        ],
        "Numeric normalized\nsMAPE score": [
            scored.loc[
                (scored.model == m) & scored["type"].isin(NUMERIC_TYPES),
                "normalized_smape_score",
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
    ax.set_title("Aggregate benchmark performance by model")
    ax.legend(title="model", loc="upper right")
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
    axes[0].set_title(f"All {total} fields")
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
    fig.suptitle("Per-field difficulty across the models", y=1.02)
    fig.tight_layout()
    return fig


def fig_thresholds(scored: pd.DataFrame) -> plt.Figure:
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

    metric_labels = ["within ±30%", "within ×2", "within decade"]
    bars = {}
    for m in models:
        sub = num[num.model == m]
        lr = np.abs(np.log10(np.abs(sub.numeric_pred / sub.numeric_gt)))
        bars[m] = [
            float((sub.smape < 0.3).mean()),
            float((lr < np.log10(2)).mean()),
            float((lr < 1).mean()),
        ]
    lr = np.abs(np.log10(np.abs(baseline / gt)))
    smapes = 2 * np.abs(baseline - gt) / (np.abs(baseline) + np.abs(gt))
    bars["const baseline"] = [
        float((smapes < 0.3).mean()),
        float((lr < np.log10(2)).mean()),
        float((lr < 1).mean()),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = models + ["const baseline"]
    palette = {**palette, "const baseline": "grey"}
    x = np.arange(len(metric_labels))
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, bars[k], width,
               label=_nice_label(k), color=palette[k])
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of numeric predictions")
    ax.set_title("Numeric prediction accuracy at common physics-intuition thresholds")
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
        r = np.sort(np.abs(np.log10(np.abs(sub.numeric_pred / sub.numeric_gt))).to_numpy())
        cdf = np.arange(1, len(r) + 1) / len(r)
        ax.plot(r, cdf, label=_nice_label(m), color=palette[m], lw=2)
    r = np.sort(np.abs(np.log10(np.abs(baseline / gt))))
    cdf = np.arange(1, len(r) + 1) / len(r)
    ax.plot(r, cdf, "--", color="grey", lw=1.5,
            label=f"constant baseline ({baseline:.2f})")
    for x_, label in ((np.log10(2), "factor of 2"), (1.0, "1 decade")):
        ax.axvline(x_, color="k", lw=0.6, ls=":")
        ax.text(x_ + 0.02, 0.04, label, rotation=90)
    ax.set_xlabel("|log10(prediction / ground truth)|")
    ax.set_ylabel("cumulative fraction of numeric predictions")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_title("How close are numeric predictions, on log scale?")
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
    n_total = int(scored.groupby("file_id").size().sum() / max(scored["model"].nunique(), 1))
    n_big = int((scored.file_id == big).sum() / max(scored["model"].nunique(), 1))

    overall = [scored[scored.model == m]["score"].mean() for m in models]
    without = [scored[(scored.model == m) & (scored.file_id != big)]["score"].mean()
               for m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(models))
    width = 0.4
    bars_all = ax.bar(x - width / 2, overall, width,
                      label=f"all {n_total} fields",
                      color=[palette[m] for m in models])
    bars_wo = ax.bar(x + width / 2, without, width,
                     label=f"excluding {big} ({n_total - n_big} fields)",
                     color=[palette[m] for m in models],
                     hatch="//", edgecolor="white")
    for bars in (bars_all, bars_wo):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [_nice_label(m) for m in models])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean field score")
    ax.set_title(f"One paper ({big}, {n_big} fields) skews the aggregate")
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
    ax.set_title("Jaccard agreement on correctly-answered fields\n(|A∩B| / |A∪B|)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def fig_per_paper(scored: pd.DataFrame) -> plt.Figure:
    models = _model_order(scored)
    palette = _model_palette(models)
    g = (scored.groupby(["file_id", "model"])["score"].mean()
              .unstack("model"))[models]
    g["_mean"] = g.mean(axis=1)
    g = g.sort_values("_mean").drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(g)), 5))
    x = np.arange(len(g))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, g[m].values, width,
               label=_nice_label(m), color=palette[m])
    ax.set_xticks(x, g.index, rotation=60, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean field score on that paper")
    ax.set_title("Per-paper performance (sorted by cross-model mean)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_leakage(scored: pd.DataFrame) -> plt.Figure:
    models = _model_order(scored)
    rows = []
    for m in models:
        sub = scored[scored.model == m]
        for leak, label in [(False, "value NOT in description"),
                            (True, "value literally in description")]:
            s = sub[sub.leak == leak]
            rows.append({
                "model": _nice_label(m),
                "leak": label,
                "mean_score": s["score"].mean(),
                "n": len(s),
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="model", columns="leak", values="mean_score").reindex(
        [_nice_label(m) for m in models]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot.bar(ax=ax, color=["#7fa9d8", "#0b3d91"])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean field score")
    ax.set_xticklabels([_nice_label(m) for m in models], rotation=0)
    leak_fraction = float((scored.drop_duplicates(
        ["file_id", "experiment", "key"])["leak"]).mean())
    ax.set_title(
        f"Effect of literal-number leakage from description "
        f"({leak_fraction * 100:.0f}% of fields)"
    )
    ax.legend(title=None, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- driver

PlotFactory = Callable[..., plt.Figure]


def _figure_specs() -> list[tuple[str, PlotFactory, tuple[str, ...]]]:
    """(filename stem, builder, required-arg names)."""
    return [
        ("aggregate_by_model", fig_aggregate, ("scored",)),
        ("difficulty_distribution", fig_difficulty, ("wide",)),
        ("numeric_thresholds", fig_thresholds, ("scored",)),
        ("numeric_log_ratio_cdf", fig_log_ratio_cdf, ("scored",)),
        ("numeric_scatter", fig_numeric_scatter, ("scored",)),
        ("paper_dominance", fig_paper_dominance, ("scored",)),
        ("model_agreement", fig_model_agreement, ("wide",)),
        ("per_paper", fig_per_paper, ("scored",)),
        ("leakage_effect", fig_leakage, ("scored",)),
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
