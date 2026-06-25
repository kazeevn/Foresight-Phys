"""Cross-model analysis: pivot per-field, classify difficulty, write summary JSON.

All per-model and per-(model, ...) aggregates use a paper-macro of an
experiment-macro (one experiment = one vote within its paper, one paper = one
vote within a model). This matches the per-run ``aggregate_*`` numbers written
by ``benchmark.run_benchmark`` so the cross-model summary is not dominated by
papers that contribute many redundant fields.

Numeric fields are summarised through proper-scoring-rule quantities: mean
normalized CRPS (the primary metric, lower is better), the derived quality
``1 - crps_scaled / CRPS_SCALED_CAP`` mapped onto [0, 1], and coverage at
1σ / 2σ as calibration diagnostics.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .decisions import load_comparison_sets, write_decisions
from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
DISC_TYPES = {"bool", "categorical"}

COVERAGE_BUCKETS = (
    ("frac_within_1sigma", 1.0),
    ("frac_within_2sigma", 2.0),
    ("frac_within_3sigma", 3.0),
)


def _paper_macro(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Per (model, file, exp) row-mean, then per (model, file), then per model.

    Mirrors ``metrics.build_file_report_data`` /
    ``benchmark.average_metric``.
    """
    valid = df[df[value_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    per_exp = valid.groupby(["model", "file_id", "experiment"])[value_col].mean()
    per_file = per_exp.groupby(["model", "file_id"]).mean()
    return per_file.groupby("model").mean()


def _difficulty(n_correct: int, n_models: int) -> str:
    if n_correct == n_models:
        return "trivial (all correct)"
    if n_correct == n_models - 1:
        return "easy (1 wrong)"
    if n_correct == 0:
        return "intractable (none correct)"
    if n_correct == 1:
        return "hard (1 right)"
    return "discriminative"


def _build_wide(scored: pd.DataFrame) -> pd.DataFrame:
    models = sorted(scored["model"].unique())
    wide = scored.pivot_table(
        index=[
            "file_id",
            "experiment",
            "key",
            "type",
            "gt_value",
            "result_description",
            "experiment_description",
        ],
        columns="model",
        values=["correct", "quality", "crps", "numeric_pred", "z"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [f"{a}__{b}" if b else a for a, b in wide.columns]

    correct_cols = [f"correct__{m}" for m in models]
    wide["n_correct"] = wide[correct_cols].sum(axis=1).astype(int)
    wide["n_models"] = len(models)
    wide["difficulty"] = wide.apply(
        lambda r: _difficulty(r["n_correct"], r["n_models"]), axis=1
    )
    return wide


def _per_model_summary(scored: pd.DataFrame) -> list[dict]:
    quality_macro = _paper_macro(scored, "quality")
    correct_macro = _paper_macro(
        scored.assign(correct_f=scored["correct"].astype(float)),
        "correct_f",
    )

    numeric_df = scored[scored["type"].isin(NUMERIC_TYPES)].copy()
    numeric_df["abs_z"] = numeric_df["z"].abs()
    for col, cutoff in COVERAGE_BUCKETS:
        numeric_df[col] = (numeric_df["abs_z"] < cutoff).astype(float)
    crps_macro = _paper_macro(numeric_df, "crps")
    crps_scaled_macro = (
        _paper_macro(numeric_df, "crps_scaled")
        if "crps_scaled" in numeric_df.columns
        else pd.Series(dtype=float)
    )
    numeric_quality_macro = _paper_macro(numeric_df, "quality")
    coverage_macros = {col: _paper_macro(numeric_df, col) for col, _ in COVERAGE_BUCKETS}

    disc_df = (
        scored[scored["type"].isin(DISC_TYPES)]
        .assign(correct_f=lambda d: d["correct"].astype(float))
    )
    disc_macro = _paper_macro(disc_df, "correct_f")
    disc_brier_macro = _paper_macro(disc_df, "brier")

    rows: list[dict] = []
    for m, sub in scored.groupby("model"):
        row = {
            "model": m,
            "n_fields": int(len(sub)),
            "n_papers": int(sub["file_id"].nunique()),
            "mean_quality": float(quality_macro.get(m, float("nan"))) if m in quality_macro.index else None,
            "mean_correct": float(correct_macro.get(m, float("nan"))) if m in correct_macro.index else None,
            "numeric_quality": (
                float(numeric_quality_macro.get(m, float("nan")))
                if m in numeric_quality_macro.index
                else None
            ),
            "numeric_crps": float(crps_macro.get(m, float("nan"))) if m in crps_macro.index else None,
            "numeric_crps_scaled": (
                float(crps_scaled_macro.get(m, float("nan")))
                if m in crps_scaled_macro.index
                else None
            ),
            "bool_categorical_accuracy": (
                float(disc_macro.get(m, float("nan"))) if m in disc_macro.index else None
            ),
            "bool_categorical_brier": (
                float(disc_brier_macro.get(m, float("nan")))
                if m in disc_brier_macro.index
                else None
            ),
        }
        for col, _ in COVERAGE_BUCKETS:
            macro = coverage_macros[col]
            row[col] = float(macro.get(m, float("nan"))) if m in macro.index else None
        rows.append(row)
    return rows


def _numeric_calibration(scored: pd.DataFrame) -> list[dict]:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["z"].notna()
    ].copy()
    num["abs_z"] = num["z"].abs()
    for col, cutoff in COVERAGE_BUCKETS:
        num[col] = (num["abs_z"] < cutoff).astype(float)
    crps_macro = _paper_macro(num, "crps")
    crps_scaled_macro = (
        _paper_macro(num, "crps_scaled")
        if "crps_scaled" in num.columns
        else pd.Series(dtype=float)
    )
    quality_macro = _paper_macro(num, "quality")
    coverage = {col: _paper_macro(num, col) for col, _ in COVERAGE_BUCKETS}

    out: list[dict] = []
    for m, sub in num.groupby("model"):
        row = {
            "model": m,
            "n_fields": int(len(sub)),
            "n_papers": int(sub["file_id"].nunique()),
            "numeric_crps": float(crps_macro.get(m, float("nan"))) if m in crps_macro.index else None,
            "numeric_crps_scaled": (
                float(crps_scaled_macro.get(m, float("nan")))
                if m in crps_scaled_macro.index
                else None
            ),
            "numeric_quality": (
                float(quality_macro.get(m, float("nan")))
                if m in quality_macro.index
                else None
            ),
        }
        for col, _ in COVERAGE_BUCKETS:
            macro = coverage[col]
            row[col] = float(macro.get(m, float("nan"))) if m in macro.index else None
        out.append(row)
    return out


TRIVIAL_LABEL = "trivial (all correct)"
# 1sigma coverage of a calibrated Gaussian; the natural calibration reference
# for numeric fields, which have no well-defined uniform "random guess".
GAUSSIAN_1SIGMA_COVERAGE = 0.6826894921370859


def _merge_difficulty(scored: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-field difficulty label onto each long (model, field) row."""
    keys = ["file_id", "experiment", "key"]
    return scored.merge(wide[[*keys, "difficulty"]], on=keys, how="left")


def _hard_field_summary(scored: pd.DataFrame, wide: pd.DataFrame) -> dict:
    """Per-model quality restricted to non-trivial fields (not solved by all models).

    The aggregate quality is inflated by the large trivial mass, so we report the
    same paper-macro-of-experiment-macro on the non-trivial subset (directly
    comparable to the headline) alongside a transparent field-level micro mean.
    """
    merged = _merge_difficulty(scored, wide)
    non_trivial = merged[merged["difficulty"] != TRIVIAL_LABEL]
    quality_macro = _paper_macro(non_trivial, "quality")
    micro = non_trivial.groupby("model")["quality"].mean()
    counts = non_trivial.groupby("model")["quality"].size()
    per_model = [
        {
            "model": m,
            "hard_quality_paper_macro": (
                float(quality_macro.get(m)) if m in quality_macro.index else None
            ),
            "hard_quality_micro": float(micro.get(m)) if m in micro.index else None,
            "n_fields_non_trivial": int(counts.get(m, 0)),
        }
        for m in sorted(scored["model"].unique())
    ]
    n_non_trivial = int((wide["difficulty"] != TRIVIAL_LABEL).sum())
    return {
        "non_trivial_n_fields": n_non_trivial,
        "trivial_n_fields": int(len(wide) - n_non_trivial),
        "per_model": per_model,
    }


def _quality_by_difficulty(scored: pd.DataFrame, wide: pd.DataFrame) -> list[dict]:
    """Field-level (micro) mean quality per (difficulty bin, model)."""
    merged = _merge_difficulty(scored, wide)
    rows: list[dict] = []
    models = sorted(scored["model"].unique())
    for difficulty, sub in merged.groupby("difficulty"):
        # n_fields counts distinct fields in the bin, not model-rows.
        n_fields = int(sub[["file_id", "experiment", "key"]].drop_duplicates().shape[0])
        per_model_quality = sub.groupby("model")["quality"].mean()
        rows.append(
            {
                "difficulty": difficulty,
                "n_fields": n_fields,
                **{
                    f"quality__{m}": (
                        float(per_model_quality.get(m))
                        if m in per_model_quality.index
                        else None
                    )
                    for m in models
                },
            }
        )
    return rows


# Grounded centrality scale, most-central first: the objective "appears in the
# abstract" tier on top, then the LLM-assigned levels for everything below it.
GROUNDED_CENTRALITY_ORDER = (
    "in_abstract",
    "headline",
    "key_supporting",
    "secondary",
    "setup_or_control",
)


def _quality_by_label(
    scored: pd.DataFrame, label_col: str, order: tuple[str, ...] | None = None
) -> list[dict]:
    """Paper-macro quality per (annotation label, model).

    Mirrors the headline aggregation so the per-label numbers are directly
    comparable to ``per_model.mean_quality``. ``order`` sorts the rows from
    most- to least-central. Returns ``[]`` when the dataset carries no annotation
    for this axis.
    """
    if label_col not in scored.columns:
        return []
    labelled = scored[scored[label_col].notna()]
    if labelled.empty:
        return []
    models = sorted(scored["model"].unique())
    rows: list[dict] = []
    for label, group in labelled.groupby(label_col):
        macro = _paper_macro(group, "quality")
        n_fields = int(group[["file_id", "experiment", "key"]].drop_duplicates().shape[0])
        rows.append(
            {
                "label": str(label),
                "n_fields": n_fields,
                **{
                    f"quality__{m}": (float(macro.get(m)) if m in macro.index else None)
                    for m in models
                },
            }
        )
    if order is not None:
        rank = {label: i for i, label in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r["label"], len(order)))
    return rows


def _abstract_grounding(scored: pd.DataFrame) -> dict:
    """Validate and report the objective abstract-mention tier of centrality.

    ``appears_in_abstract`` is a reproducible importance marker, so it heads the
    centrality scale. We report (i) its enrichment among the LLM's headline labels
    versus the rest---an independent cross-check of the subjective labels---and
    (ii) per-model quality on the abstract-grounded crucial tier.
    """
    if "appears_in_abstract" not in scored.columns:
        return {}
    fields = scored.drop_duplicates(["file_id", "experiment", "key"])
    in_abstract_field = fields["appears_in_abstract"] == True  # noqa: E712
    if not in_abstract_field.any():
        return {}
    out: dict = {
        "n_in_abstract_fields": int(in_abstract_field.sum()),
        "n_total_fields": int(len(fields)),
    }
    if "is_headline" in fields.columns:
        headline = fields["is_headline"] == True  # noqa: E712
        out["appears_rate_headline"] = (
            float((fields[headline]["appears_in_abstract"] == True).mean()) if headline.any() else None  # noqa: E712
        )
        out["appears_rate_non_headline"] = (
            float((fields[~headline]["appears_in_abstract"] == True).mean()) if (~headline).any() else None  # noqa: E712
        )
    crucial = scored[scored["appears_in_abstract"] == True]  # noqa: E712
    macro = _paper_macro(crucial, "quality")
    out["per_model"] = [
        {
            "model": m,
            "in_abstract_quality_paper_macro": (
                float(macro.get(m)) if m in macro.index else None
            ),
        }
        for m in sorted(scored["model"].unique())
    ]
    return out


def _headline_summary(scored: pd.DataFrame) -> dict:
    """Per-model quality restricted to the papers' headline findings.

    This is the direct answer to "did the models get the crucial variables?":
    quality on the small set of fields the annotation marks ``is_headline``,
    aggregated with the same paper-macro as the overall number.
    """
    if "is_headline" not in scored.columns:
        return {}
    headline = scored[scored["is_headline"] == True]  # noqa: E712 (works for object/bool cols)
    if headline.empty:
        return {}
    macro = _paper_macro(headline, "quality")
    micro = headline.groupby("model")["quality"].mean()
    counts = headline.groupby("model")["quality"].size()
    n = int(headline[["file_id", "experiment", "key"]].drop_duplicates().shape[0])
    return {
        "n_headline_fields": n,
        "per_model": [
            {
                "model": m,
                "headline_quality_paper_macro": (
                    float(macro.get(m)) if m in macro.index else None
                ),
                "headline_quality_micro": float(micro.get(m)) if m in micro.index else None,
                "n_fields": int(counts.get(m, 0)),
            }
            for m in sorted(scored["model"].unique())
        ],
    }


def _dataset_composition(scored: pd.DataFrame) -> dict:
    """Field-count composition of the benchmark (what is actually being scored)."""
    fields = scored.drop_duplicates(["file_id", "experiment", "key"])

    def counts(col: str) -> dict:
        if col not in fields.columns:
            return {}
        return {str(k): int(v) for k, v in fields[col].value_counts(dropna=False).items()}

    per_paper = fields.groupby("file_id").size()
    return {
        "n_fields": int(len(fields)),
        "by_type": counts("type"),
        "by_centrality": counts("centrality"),
        "by_surprise": counts("ex_ante_surprise"),
        "fields_per_paper": {
            "min": int(per_paper.min()),
            "median": float(per_paper.median()),
            "mean": float(per_paper.mean()),
            "max": int(per_paper.max()),
        },
        "most_field_rich_paper": {
            "file_id": str(per_paper.idxmax()),
            "n_fields": int(per_paper.max()),
        },
    }


def _dedup_aggregate(scored: pd.DataFrame, sets_by_file: dict[str, list[dict]]) -> list[dict]:
    """Per-model quality after collapsing each comparison set to one trend vote.

    A swept series (e.g. five spectral-weight ratios) currently contributes as
    many fields as it has points, over-weighting the paper it lives in. Here each
    comparison set is replaced by a single field whose quality is the mean of its
    members, so redundant series count once. Reported next to ``mean_quality`` so
    the inflation from redundant fields is visible.
    """
    member_to_set: dict[tuple[str, int, str], tuple[str, int, int]] = {}
    for file_id, csets in sets_by_file.items():
        for set_index, cset in enumerate(csets):
            experiment = int(cset.get("experiment_index", -1))
            for key in cset.get("member_keys", []):
                member_to_set[(file_id, experiment, str(key))] = (file_id, experiment, set_index)

    rows: list[dict] = []
    for (model, file_id, experiment), group in scored.groupby(["model", "file_id", "experiment"]):
        set_acc: dict[tuple, list[float]] = {}
        singles: list[float] = []
        for _, r in group.iterrows():
            q = r["quality"]
            if pd.isna(q):
                continue
            set_id = member_to_set.get((file_id, int(experiment), str(r["key"])))
            if set_id is None:
                singles.append(float(q))
            else:
                set_acc.setdefault(set_id, []).append(float(q))
        collapsed = singles + [float(np.mean(v)) for v in set_acc.values()]
        if collapsed:
            rows.append({"model": model, "file_id": file_id, "exp_mean": float(np.mean(collapsed))})

    if not rows:
        return []
    df = pd.DataFrame(rows)
    per_model = df.groupby(["model", "file_id"])["exp_mean"].mean().groupby("model").mean()
    return [
        {"model": m, "dedup_quality_paper_macro": float(per_model.get(m))}
        for m in sorted(scored["model"].unique())
        if m in per_model.index
    ]


def _leakage_summary(scored: pd.DataFrame) -> dict:
    """Quality on fields the annotation flags as determinable from the description.

    ``leakage_sufficient`` fields can be answered without doing the experiment
    (e.g. a result that restates a stated sweep range). We report the leaked count
    and a leakage-excluded "clean" aggregate so the headline can be read net of
    near-given fields.
    """
    if "leakage_sufficient" not in scored.columns:
        return {}
    leaked_mask = scored["leakage_sufficient"] == True  # noqa: E712
    if not leaked_mask.any():
        return {}
    keys = ["file_id", "experiment", "key"]
    n_leaked = int(scored[leaked_mask][keys].drop_duplicates().shape[0])
    n_total = int(scored[keys].drop_duplicates().shape[0])
    clean_macro = _paper_macro(scored[~leaked_mask], "quality")
    leaked_macro = _paper_macro(scored[leaked_mask], "quality")
    return {
        "n_leaked_fields": n_leaked,
        "n_total_fields": n_total,
        "per_model": [
            {
                "model": m,
                "clean_quality_paper_macro": (
                    float(clean_macro.get(m)) if m in clean_macro.index else None
                ),
                "leaked_quality_paper_macro": (
                    float(leaked_macro.get(m)) if m in leaked_macro.index else None
                ),
            }
            for m in sorted(scored["model"].unique())
        ],
    }


def _foresight_lift(full_scored: pd.DataFrame) -> list[dict]:
    """Quality gained from the experiment description over name-only priors.

    For every base model that has both a standard run and a ``name-only`` ablation
    run, the foresight lift is the paper-macro quality difference. Lift ≈ 0 means
    the model predicted the field from the typed key alone (a trivial/observable-type
    prior); large positive lift means the experiment description carried real
    predictive content. Also reported on the headline and surprising subsets.
    """
    if "ablation" not in full_scored.columns or "base_model" not in full_scored.columns:
        return []
    if not (full_scored["ablation"] == "name-only").any():
        return []

    overall = _paper_macro(full_scored, "quality")
    headline_mask = (
        full_scored["is_headline"] == True  # noqa: E712
        if "is_headline" in full_scored.columns
        else pd.Series(False, index=full_scored.index)
    )
    surprising_mask = (
        full_scored["ex_ante_surprise"] == "surprising"
        if "ex_ante_surprise" in full_scored.columns
        else pd.Series(False, index=full_scored.index)
    )
    in_abstract_mask = (
        full_scored["appears_in_abstract"] == True  # noqa: E712
        if "appears_in_abstract" in full_scored.columns
        else pd.Series(False, index=full_scored.index)
    )
    headline = _paper_macro(full_scored[headline_mask], "quality")
    surprising = _paper_macro(full_scored[surprising_mask], "quality")
    in_abstract = _paper_macro(full_scored[in_abstract_mask], "quality")

    def _diff(series: pd.Series, full_label: str, name_label: str) -> float | None:
        if full_label in series.index and name_label in series.index:
            return float(series[full_label] - series[name_label])
        return None

    def _get(series: pd.Series, label: str) -> float | None:
        return float(series[label]) if label in series.index else None

    rows: list[dict] = []
    for base_model in sorted(full_scored["base_model"].dropna().unique()):
        full_label, name_label = base_model, f"{base_model} [name-only]"
        if full_label not in overall.index or name_label not in overall.index:
            continue
        rows.append(
            {
                "base_model": base_model,
                "quality_full": float(overall[full_label]),
                "quality_name_only": float(overall[name_label]),
                "foresight_lift": float(overall[full_label] - overall[name_label]),
                "foresight_lift_headline": _diff(headline, full_label, name_label),
                "foresight_lift_surprising": _diff(surprising, full_label, name_label),
                "quality_full_in_abstract": _get(in_abstract, full_label),
                "quality_name_only_in_abstract": _get(in_abstract, name_label),
                "foresight_lift_in_abstract": _diff(in_abstract, full_label, name_label),
            }
        )
    return rows


def _gather_categorical_k(paths: AnalysisPaths) -> list[int]:
    cat_k: list[int] = []
    for jp in sorted(paths.json_dir.glob("*.json")):
        data = json.loads(jp.read_text())
        if not data:
            continue
        for exp in data:
            for v in exp["experiment_results"].values():
                if v.get("type") == "categorical":
                    allowed = v.get("allowed_categorial_values")
                    if isinstance(allowed, list) and len(allowed) > 1:
                        cat_k.append(len(allowed))
    return cat_k


def _categorical_baseline(cat_k: list[int]) -> dict:
    if not cat_k:
        return {}
    return {
        "categorical_random_baseline": float(np.mean([1.0 / k for k in cat_k])),
        "categorical_k_distribution": dict(Counter(cat_k)),
    }


def _random_baselines(cat_k: list[int]) -> dict:
    """Random-guess reference per answer type.

    Quality uses the same per-type rule as the scorer. For an uninformative
    guess: boolean p=0.5 -> Brier 0.25 -> quality 0.75; categorical uniform over
    k -> Brier (k-1)/k -> quality 1 - 0.5*(k-1)/k, averaged over the observed
    k-distribution; formula with uninformative confidence 0.5 -> quality 0.75,
    and a random expression effectively never matches (exact 0). Numeric has no
    well-defined uniform guess (relative-CRPS quality depends on the forecaster's
    chosen scale), so we record the calibrated-Gaussian coverage as the reference.
    """
    cat_exact = float(np.mean([1.0 / k for k in cat_k])) if cat_k else None
    cat_quality = (
        float(np.mean([1.0 - 0.5 * (k - 1) / k for k in cat_k])) if cat_k else None
    )
    return {
        "boolean": {
            "exact": 0.5,
            "quality": 0.75,
            "note": "uninformative p=0.5; Brier 0.25",
        },
        "categorical": {
            "exact": cat_exact,
            "quality": cat_quality,
            "note": "per-question uniform over k allowed values, averaged over the k-distribution",
        },
        "formula": {
            "exact": 0.0,
            "quality": 0.75,
            "note": "random expression ~never matches; uninformative confidence 0.5",
        },
        "numeric": {
            "exact": None,
            "quality": None,
            "note": (
                "no well-defined uniform baseline (relative-CRPS quality depends on the "
                f"forecaster's chosen scale); calibrated-Gaussian 1-sigma coverage is "
                f"{GAUSSIAN_1SIGMA_COVERAGE:.4f}"
            ),
        },
    }


def write_summary(paths: AnalysisPaths | None = None) -> dict:
    paths = paths or default_paths()
    paths.ensure_dirs()

    full_scored = pd.read_parquet(paths.scored_parquet)
    # The headline analyses use only the standard (non-ablation) runs; ablation
    # runs (e.g. name-only) are paired against them solely for the foresight lift.
    if "ablation" in full_scored.columns:
        scored = full_scored[full_scored["ablation"] == "none"].copy()
    else:
        scored = full_scored
    wide = _build_wide(scored)
    wide.to_parquet(paths.per_field_parquet, index=False)

    difficulty_overall = wide["difficulty"].value_counts().to_dict()
    difficulty_by_type = (
        wide.groupby(["type", "difficulty"]).size().unstack(fill_value=0)
        .reset_index().to_dict(orient="records")
    )

    cat_k = _gather_categorical_k(paths)
    summary = {
        "n_models": int(scored["model"].nunique()),
        "n_papers": int(scored["file_id"].nunique()),
        "n_fields": int(len(wide)),
        "aggregation": "paper-macro of experiment-macro (matches per-run aggregate_*)",
        "per_model": _per_model_summary(scored),
        "numeric_calibration": _numeric_calibration(scored),
        "difficulty_distribution_overall": difficulty_overall,
        "difficulty_by_type": difficulty_by_type,
        "hard_field": _hard_field_summary(scored, wide),
        "quality_by_difficulty": _quality_by_difficulty(scored, wide),
        "headline": _headline_summary(scored),
        "abstract_grounding": _abstract_grounding(scored),
        "quality_by_centrality": _quality_by_label(
            scored, "centrality_grounded", order=GROUNDED_CENTRALITY_ORDER
        ),
        "quality_by_centrality_llm": _quality_by_label(scored, "centrality"),
        "quality_by_surprise": _quality_by_label(scored, "ex_ante_surprise"),
        "leakage": _leakage_summary(scored),
        "dataset_composition": _dataset_composition(scored),
        "dedup_aggregate": _dedup_aggregate(scored, load_comparison_sets(paths)),
        "foresight_lift": _foresight_lift(full_scored),
        "decisions": write_decisions(paths),
        "random_baselines": _random_baselines(cat_k),
        **_categorical_baseline(cat_k),
    }
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
