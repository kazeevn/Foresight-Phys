# Foresight-Phys benchmark — analysis of the four checked-in runs

**Run set** (all 2026-05-14):
| model | run-name | aggregate prediction quality | normalized sMAPE | bool/cat acc | formula acc |
|---|---|---|---|---|---|
| gpt-5.4-nano | brisk-mode | 0.504 | 0.512 | 0.884 | 0.333 |
| gpt-5.4-mini | lattice-phase | 0.488 | 0.492 | 0.899 | 0.250 |
| gpt-5.4 | prism-photon | 0.577 | 0.566 | 0.968 | 0.625 |
| gpt-5.5 | rapid-lens | **0.637** | **0.601** | 0.931 | **0.750** |

(These are reproduced from each run's `benchmark_results.json`. The aggregate is a per‑paper mean, so it equally weights a 2‑field paper and the 120‑field CMS top‑quark paper — see §3 for why this matters.)

All plots are in [`plots/`](plots/) (one PDF per figure). See [`README.md`](README.md) for how to regenerate them.

---

## 1. Does the setup make sense as written?

**Mostly yes — but with several seams that bias the headline numbers.** The pipeline does what it says: take a paper, mask every `result`, ask a model to predict each one given only the experiment description + the result's text description, and grade per‑field with sMAPE / exact match / a formula judge. The 4 runs use the exact same masked inputs, system prompt and grader, so they are directly comparable. The arXiv IDs are 2601.* through 2605.* (Jan – May 2026), all newer than the models' training cutoffs, so this is genuinely a *prediction* task rather than recall.

Where the setup is fragile:

1. **One paper dominates the field pool.** The CMS top‑quark spin density‑matrix paper (`2602.15115`) contributes **120 of 347 fields** — 35% of all gradeable items — and is by far the hardest (40 of those 120 are intractable for every model). It pulls every aggregate metric down. See `fig_paper_dominance` in the PDF: dropping this single file lifts every model's mean field score by 4 – 6 points (e.g. gpt‑5.5 goes from 0.564 → 0.617).

2. **Per‑paper averaging is sensitive to where the question count sits.** The benchmark's headline metric is `mean(per‑paper mean)`, which weights a paper with 2 fields the same as one with 120. With this dataset, per‑paper mean and per‑field mean differ by ~5 points for every model.

3. **Almost half of all `JSONs/filtered/` files are empty** (18 of 38). The benchmark silently skips them, so the effective N is 20 papers. This is fine but worth documenting.

4. **Most "result fields" are arbitrary measurements, not predictable outcomes.** Many entries in `experiment_results` are *paper‑specific data points* — a particular value in a particular bin in a particular sample — that no working physicist could predict from a setup description. For instance, of the 91 fields *no* model solved, virtually all are of this form: ARPES band shifts at specific high‑symmetry points in a specific intercalated CoxTaS2 sample, bin‑by‑bin entanglement markers in the CMS analysis, IXS‑plot read‑offs at specific q. The benchmark grades these alongside questions like "is the transport regime localized?" — which is genuinely predictable from the setup. The conflation depresses the scientific signal.

5. **Mild numeric leakage from the question text.** Of the 293 float results with non‑zero ground truth, **21 (~7%)** have their literal numeric value appearing somewhere in the experiment description or the result's own description string. The effect on accuracy is small (see `fig_leakage`) and on gpt‑5.5 leak fields are actually *harder* than non‑leak ones, but it is a small source of measurement noise.

6. **`bool` / `categorical` questions are mostly reading-comprehension, not prediction.** Examples that all 4 models got right include "was beam storage observed with the kicker?" (description states the kicker pulse was the storage trigger), "do entropy and purity increase / decrease with dephasing?" (textbook physics), "did the A1g Raman phonon appear in cross_xy and colinear_xx?" (description named the channels). Only a handful of categorical items actually probe physical intuition. So the ~85 – 95% bool/cat accuracy is largely *not* evidence of foresight.

7. **Setup‑internal consistency questions are mixed with foresight questions.** Some results pairs are mutually-determining (e.g. once a γ‑band dispersion is labelled "inverse_mexican_hat", the number of van Hove singularities along that direction is mechanically 2). Models can be right on the dependent question for the wrong reason.

The setup is *good enough to rank models* — the ordering 5.5 > 5.4 > 5.4‑mini ≈ 5.4‑nano is consistent and falsifiable — but it is not yet a clean test of "scientific intuition" as such.

## 2. Do the models actually have scientific intuition, or are they coasting on the easy bits?

**Both, in measurable proportions.** Defining a field as "correct for a given model" if its normalized score is ≥ 0.5 (i.e. sMAPE < 1, or categorical match, or judged formula match), and counting across the 4 models:

| 4 / 4 correct (trivial) | 3 / 4 (easy) | 2 / 4 (discriminative) | 1 / 4 (hard) | 0 / 4 (intractable) |
|---|---|---|---|---|
| **100** (29%) | 44 (13%) | 59 (17%) | 53 (15%) | **91** (26%) |

(`fig_difficulty` in the PDF.)

- **~30% of items are trivial in this 4‑model panel.** These break down as: 67 numeric, 14 bool, 14 categorical, 4 integer, 1 formula. Most of the 67 trivial floats are quantities whose order of magnitude is dictated by the setup (a 1600 nm pulse → photon energy near 0.78 eV; a Ti:Sapphire 800 nm photon → 1.55 eV; CrI3 Curie temperature near 40 K) or values the description came close to literally giving (the 21 numeric‑leak fields are concentrated in this bucket).
- **~26% of items are intractable.** Of the 91, **88 are numeric and 40 come from the CMS top‑quark paper alone.** These are essentially "guess this specific bin's value to two significant figures" — outside what any physicist can do from a setup description.
- **156 fields (45%) are actually discriminative** between models. This is where the benchmark earns its keep.

Stronger signal lives in the **numeric thresholds** rather than the headline averages. Restricting to numeric fields with non‑zero ground truth and a numeric prediction (n = 297 – 300 per model), and a constant‑predictor baseline that always returns the geometric median of all answers (1.20):

|  | within ±30% | within ×2 | within decade |
|---|---|---|---|
| constant baseline (1.20) | 0.21 | 0.21 | 0.66 |
| gpt-5.4-nano | 0.31 | 0.53 | 0.88 |
| gpt-5.4-mini | 0.28 | 0.52 | 0.85 |
| gpt-5.4 | 0.32 | 0.57 | 0.91 |
| **gpt-5.5** | **0.41** | **0.65** | **0.94** |

(See `fig_thresholds` and the log‑ratio CDF in `fig_log_ratio_cdf`.) The lift from baseline to gpt‑5.5 is large — from 21% to 65% of numeric predictions within a factor of 2 is a real, non‑trivial physics‑intuition signal. The lift on within‑decade is smaller because the geometric‑median baseline already covers a lot of physically reasonable scales for a benchmark whose target values are mostly between 0.01 and 100.

**Where do the models add real value?**

- *Order‑of‑magnitude reasoning on numerics.* Every model lands ≥ 85% of its numeric predictions within a decade — substantially above the 66% you can get by always guessing ~1. The model is using the setup's units, scales and material identity to anchor its guess.
- *Direction / sign questions disguised as numeric.* Many of the small (× 0.6 → × 0.8) sMAPE values for shift / drop quantities reflect getting the sign and rough size right.
- *25 fields are solved only by gpt‑5.5.* These are the strongest evidence of model‑specific intuition — e.g. correctly retrieving that the Co0.32TaS2 spin‑orbit momentum splitting is closer to 0.18 Å⁻¹ than to the pristine 0.06 value (a non‑obvious factor‑of‑3 enhancement on intercalation), or hitting several of the CMS Bell / steering / magic markers in specific (m_tt, |cos θ|) bins close enough to count.

**Where they don't:**

- *Specific data‑point retrieval.* The 91 intractable fields are not solvable by reasoning, only by memorization — and these papers post‑date training. So 0/4 here is the *correct* outcome; the issue is that they shouldn't be in the benchmark in the first place if the goal is to measure foresight.
- *Unit‑scale slip.* gpt‑5.4‑nano made the same class of error multiple times: predicting a binding‑energy shift in eV (e.g. 0.45) when the unit is meV (target 260). All four models did this at least once on the same `multi_frequency_loading_rate_at_12_0_G_cm_s_inv` field (atoms/s vs Hz‑normalized rate, 10⁸ vs 10¹⁰ off). The system prompt explicitly warns about this; nano fails it 5× anyway, mini and 5.4 twice each, 5.5 just once. Useful as a stress test of the scaling instruction.

## 3. So which numbers from `benchmark_results.json` are load-bearing?

- **`aggregate_normalized_smape_score`** is the most meaningful headline for what the benchmark is really probing. It excludes zero‑reference numerics and saturates each field's error at 1, so it cannot be ruined by a single huge outlier — which is appropriate because OOM‑wrong predictions all look the same as "you have no idea". On this metric: 5.5 (0.60) > 5.4 (0.57) > 5.4‑nano (0.51) > 5.4‑mini (0.49). The 5.4 > 5.4‑mini ordering is consistent.
- **`aggregate_bool_categorical_accuracy`** looks impressive (0.88 – 0.97) but, given that the simple "majority‑class" baseline for the 19 bool questions is 68% and the random baseline for the 18 categoricals is 31%, *and* given that most of these questions are answerable from the description's own wording, it is the weakest indicator of scientific foresight. It is doing real work primarily as a regression test for instruction‑following.
- **`aggregate_formula_accuracy`** is computed over only 7 fields total, so the 0.25 → 0.33 → 0.62 → 0.75 staircase has wide error bars. It's a directional signal, not a measurement.
- **`aggregate_smape`** (raw) is dominated by the long tail of OOM‑wrong predictions. Reading it gives a misleading sense of how poor the models are. Treat it as a debugging metric, not a headline.

## 4. Suggested improvements to the benchmark

In order of impact:

1. **Filter the ground truth for "predictability".** Each result field should pass a gate: "given only the experimental setup, could a domain expert reasonably guess this value to within a factor of 2 / could they confidently pick a category?" Anything that fails this gate (most of the CMS bin‑by‑bin entries; ARPES band shifts at random k‑points; IXS plot read‑offs) is measurement‑transcription, not foresight. A cheap version: have a strong model judge each field for predictability and flag the bottom quartile.
2. **Cap the per‑paper field count, or aggregate per‑paper before averaging.** Even after gating, one paper shouldn't be allowed to contribute 100+ fields. Either downsample to ≤ N per paper, or report both per‑paper and per‑field aggregates so the reader knows which one they're looking at.
3. **Audit the bool/categorical pool for description leakage.** A judge model should answer each categorical question *from only the description text*, with no claim about prediction. Items it answers correctly with high confidence are reading‑comprehension and should be dropped or relabelled.
4. **Add a baselines column.** Report at minimum (a) constant‑predictor for numerics (geometric median of training answers), (b) majority‑class for bools, (c) uniform for categoricals (using `allowed_categorial_values`). It is much easier to claim "the model has physics intuition" when the comparison is visible.
5. **Decompose numeric scoring into "order of magnitude" and "within‑OOM fine grain".** The fraction within a decade is what really tells you whether the model placed the physics on the right axis; the within‑factor‑2 score is what tells you whether it modelled the system. Reporting both keeps the metric honest when one is much harder than the other.
6. **Don't grade fields with zero ground‑truth as 0 silently** when the model emits 0.0 elsewhere; the current code does the right thing for zero‑zero matches but the absence of a metric for "the model correctly identified that the answer is 0" weakens reporting on null‑result questions.
7. **Equalize question difficulty per paper, or stratify reporting.** Looking only at the per‑paper bar chart (`fig_per_paper_box`) shows that the model ranking depends on which papers happen to be in the set. With 20 papers, that's a real source of noise; with 40+ and difficulty‑stratified reporting, you'd have a much tighter benchmark.

## 5. Bottom line

- The benchmark *does* discriminate models in the direction one would expect (5.5 > 5.4 > the smaller 5.4 siblings) and the discrimination shows up in genuinely informative places: order‑of‑magnitude correctness on numerics, and a 25‑field set of "only 5.5 got this" hard items that include real physics calls (spin‑orbit splitting enhancement, Bell‑marker bins, propagation trends).
- Roughly **45% of the 347 fields actually do useful work** as a model ranking signal. Another **~30% are trivial** (reading‑comprehension categoricals + setup‑dictated orders of magnitude + literal leakage) and **~26% are intractable** for everyone, mostly because they ask for memorized numbers from unpublished‑at‑training‑time data points.
- The strongest evidence for "scientific intuition" is that gpt‑5.5 lands **65% of its numerics within a factor of 2**, against a constant‑baseline of 21%. The strongest evidence *against* a too‑rosy reading is that the high bool/categorical accuracy is largely reading‑comprehension and that one CMS paper warps the headline numbers by ~5 points.
- With the filtering, per‑paper capping, and explicit baselines suggested in §4, this benchmark could go from a fair pilot to a strong measurement of physics foresight in LLMs.

---

**Files written:**
- `docs/analysis/plots/*.pdf` — 9 individual figures (aggregate bars, difficulty stack, threshold bars, log‑ratio CDF, scatter, paper‑dominance, agreement heatmap, per‑paper bars, leakage)
- `docs/analysis/summary.json` — the numbers used in the writeup
- `.cache/analysis/predictions.parquet` — raw HTML‑parsed predictions
- `.cache/analysis/scored.parquet` — per‑(model, field) tidy table with parsed numerics, score, leak flag
- `.cache/analysis/per_field.parquet` — one row per field with each model's correctness as columns and a difficulty class

The pipeline lives in `src/foresight_phys/analysis/` and is invoked via `foresight-phys-analysis all`. See [`README.md`](README.md).
