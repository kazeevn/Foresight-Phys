"""Audit benchmark JSONs for result values that leak into the (masked) inputs.

The original check only caught exact literal substrings. Published descriptions
leak values in subtler ways: rounded forms, unit-rescaled magnitudes (eV vs meV),
and sweep endpoints that restate a result. This pass adds normalized numeric
matching and, when ``JSONs/annotations/`` is present, cross-references the
semantic ``leakage_sufficient`` flag from the annotation pass.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Unit rescalings a measured value might appear under in the description text.
SCALE_FACTORS = (1e-9, 1e-6, 1e-3, 1e-2, 1e2, 1e3, 1e6, 1e9)


def _fmt(value: float) -> str:
    """Render a number without scientific notation or trailing zero cruft."""
    if value == int(value):
        return str(int(value))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def _is_specific(token: str) -> bool:
    """Filter out tokens too common to be coincidence-free evidence of leakage.

    Keeps multi-digit values (0.29, 22.9, 260) but drops bare small integers and
    single-decimal fractions like 0.4 / 0.7 that routinely appear as bin edges or
    axis ticks in descriptions.
    """
    body = token.lstrip("-")
    if len(body) < 3:
        return False
    if "." in body:
        int_part, frac_part = body.split(".", 1)
        if int_part in ("", "0") and len(frac_part) < 2:
            return False
    elif len(body) < 3:
        return False  # pure integers shorter than 3 digits are ubiquitous
    return True


def _numeric_variants(value: float) -> set[str]:
    if isinstance(value, bool):
        return set()
    variants: set[str] = set()
    for base in (value, abs(value)):
        variants.add(_fmt(base))
        for nd in (1, 2, 3):
            variants.add(_fmt(round(base, nd)))
        for factor in SCALE_FACTORS:
            scaled = base * factor
            if 1e-6 <= abs(scaled) < 1e12:
                variants.add(_fmt(scaled))
                variants.add(_fmt(round(scaled, 3)))
    return {v for v in variants if _is_specific(v)}


def _contains(token: str, text: str) -> bool:
    # Boundaries that respect decimal points (\b breaks around '.').
    return re.search(r"(?<![\d.])" + re.escape(token) + r"(?![\d.])", text) is not None


def _field_leaks(value: Any, description: str, result_description: str) -> dict | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    literal = _fmt(float(value))
    for where, text in (("experiment_description", description), ("result_description", result_description)):
        if _is_specific(literal) and _contains(literal, text):
            return {"kind": "literal", "matched": literal, "where": where}
    for token in _numeric_variants(float(value)):
        for where, text in (("experiment_description", description), ("result_description", result_description)):
            if _contains(token, text):
                return {"kind": "normalized", "matched": token, "where": where}
    return None


def _load_semantic_leakage(annotations_dir: Path) -> set[tuple[str, int, str]]:
    flagged: set[tuple[str, int, str]] = set()
    if not annotations_dir.exists():
        return flagged
    for ap in sorted(annotations_dir.glob("*.json")):
        record = json.loads(ap.read_text(encoding="utf-8"))
        for fa in record.get("field_annotations", []):
            if fa.get("leakage_sufficient"):
                flagged.add((ap.stem, int(fa["experiment_index"]), str(fa["key"])))
    return flagged


def check_leakage(filtered_dir: Path = Path("JSONs/filtered"),
                  annotations_dir: Path = Path("JSONs/annotations")) -> dict:
    numeric_leaks: list[dict] = []
    leaked_keys: set[tuple[str, int, str]] = set()

    for json_file in sorted(filtered_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for exp_idx, exp in enumerate(data):
            desc = exp.get("experiment_description", "")
            for key, meta in exp.get("experiment_results", {}).items():
                hit = _field_leaks(meta.get("result"), desc, meta.get("description", ""))
                if hit is not None:
                    numeric_leaks.append({"file": json_file.stem, "experiment": exp_idx, "key": key, **hit})
                    leaked_keys.add((json_file.stem, exp_idx, key))

    semantic = _load_semantic_leakage(annotations_dir)
    return {
        "numeric_leak_count": len(numeric_leaks),
        "numeric_leaked_fields": sorted(f"{f}/e{e}/{k}" for f, e, k in leaked_keys),
        "semantic_leak_count": len(semantic),
        "union_leaked_count": len(leaked_keys | semantic),
        "numeric_and_semantic_overlap": len(leaked_keys & semantic),
        "numeric_leaks": numeric_leaks,
    }


if __name__ == "__main__":
    report = check_leakage()
    print(json.dumps({k: v for k, v in report.items() if k != "numeric_leaks"}, indent=2))
