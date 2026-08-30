#!/usr/bin/env python3
"""Audit Qwen3-8B teacher outputs with the shared zh-th v2 quality rules."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_ACCEPTED,
        DEFAULT_AUDIT_ALL,
        DEFAULT_AUDIT_SUMMARY,
        DEFAULT_CONFIG,
        DEFAULT_RAW_GENERATIONS,
        DEFAULT_REJECTED,
        DOMAINS,
        PIPELINE_VERSION,
        QUALITY_RULE_VERSION,
        assess_pair,
        atomic_text_writer,
        atomic_write_json,
        count_thai,
        ensure_new_outputs,
        load_config,
        percentile,
        read_jsonl,
        relative_path,
        repeat_score,
        resolve_path,
        sha256_file,
        utc_now,
    )
except ImportError:
    from common import (
        DEFAULT_ACCEPTED,
        DEFAULT_AUDIT_ALL,
        DEFAULT_AUDIT_SUMMARY,
        DEFAULT_CONFIG,
        DEFAULT_RAW_GENERATIONS,
        DEFAULT_REJECTED,
        DOMAINS,
        PIPELINE_VERSION,
        QUALITY_RULE_VERSION,
        assess_pair,
        atomic_text_writer,
        atomic_write_json,
        count_thai,
        ensure_new_outputs,
        load_config,
        percentile,
        read_jsonl,
        relative_path,
        repeat_score,
        resolve_path,
        sha256_file,
        utc_now,
    )


REQUIRED_FIELDS = {
    "candidate_id",
    "original_id",
    "domain",
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "zh_char_count",
    "teacher_model",
    "translation_method",
    "prompt_template_version",
    "generation_config",
    "source_file",
    "source_row",
}


ASCII_LOWER_WORD_RE = re.compile(r"[a-z]{4,}")
THAI_LATIN_UNIT_RE = re.compile(r"[\u0E00-\u0E7FA-Za-z]+")
EMAIL_CONTEXT_RE = re.compile(
    r"(?i)\b(?:contact(?:\s+(?:at|:))?\s+)?"
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]+(?:[-/]\d+)+(?:/\d+)*|\d+(?:/\d+)+)\b"
)


def _is_thai_character(char: str) -> bool:
    """Return whether *char* belongs to the Unicode Thai block."""

    return "\u0E00" <= char <= "\u0E7F"


def _mask_protected_latin_spans(text: str) -> str:
    """Mask URLs, e-mail contexts and identifiers without changing offsets."""

    masked = text
    for pattern in (URL_RE, EMAIL_CONTEXT_RE, IDENTIFIER_RE):
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


def _unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def detect_suspicious_lowercase_latin(text: str) -> list[str]:
    """Find ordinary lowercase Latin words/runs in a Thai target.

    Formal uppercase acronyms are not lowercase matches.  URL, e-mail and
    identifier spans are masked before scanning, so their ASCII content does
    not produce a v3 Latin rejection.
    """

    suspicious: list[str] = []
    for unit in THAI_LATIN_UNIT_RE.findall(_mask_protected_latin_spans(text)):
        has_thai = any(_is_thai_character(char) for char in unit)
        if has_thai:
            suspicious.extend(ASCII_LOWER_WORD_RE.findall(unit))
        elif unit.isascii() and unit.isalpha() and unit.islower() and len(unit) >= 4:
            suspicious.append(unit)
    return _unique_in_order(suspicious)


def detect_thai_latin_mixed_tokens(text: str) -> list[str]:
    """Find contiguous units containing both Thai and lowercase ASCII Latin."""

    mixed: list[str] = []
    for unit in THAI_LATIN_UNIT_RE.findall(_mask_protected_latin_spans(text)):
        if any(_is_thai_character(char) for char in unit) and any(
            "a" <= char <= "z" for char in unit
        ):
            mixed.append(unit)
    return _unique_in_order(mixed)


def build_v3_latin_check(text: str) -> dict[str, Any]:
    suspicious_words = detect_suspicious_lowercase_latin(text)
    mixed_tokens = detect_thai_latin_mixed_tokens(text)
    return {
        "suspicious_lowercase_words": suspicious_words,
        "thai_latin_mixed_tokens": mixed_tokens,
        "has_suspicious_lowercase_latin": bool(suspicious_words),
        "has_thai_latin_mixed_token": bool(mixed_tokens),
    }


def apply_v3_latin_rules(text: str, reasons: list[str]) -> dict[str, Any]:
    diagnostics = build_v3_latin_check(text)
    additions = (
        (
            diagnostics["has_suspicious_lowercase_latin"],
            "thai_side_suspicious_lowercase_latin",
        ),
        (diagnostics["has_thai_latin_mixed_token"], "thai_latin_mixed_token"),
    )
    for matched, reason in additions:
        if matched and reason not in reasons:
            reasons.append(reason)
    return diagnostics


def run_latin_self_test() -> int:
    cases = [
        ("thai_glued_generously", "ข้อความภาษาไทยgenerously", True, True),
        ("thai_glued_acedonia", "ประเทศไทยacedonia", True, True),
        ("separate_osce", "ภาษาไทย OSCE ข้อความ", False, False),
        ("separate_acronyms", "UNDP ภาษาไทย CORINE", False, False),
        ("identifiers", "เอกสาร A/51/495 และ ISO-9001", False, False),
        ("url", "https://example.com", False, False),
        ("contact_email", "contact abc@example.com", False, False),
        ("separate_market", "ข้อมูลตลาด market ภาษาไทย", True, False),
        ("thai_glued_word", "ภาษาไทยword", True, True),
    ]
    results: list[dict[str, Any]] = []
    for name, text, expected_lowercase, expected_mixed in cases:
        reasons: list[str] = []
        diagnostics = apply_v3_latin_rules(text, reasons)
        apply_v3_latin_rules(text, reasons)
        actual = (
            diagnostics["has_suspicious_lowercase_latin"],
            diagnostics["has_thai_latin_mixed_token"],
        )
        expected = (expected_lowercase, expected_mixed)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}: {diagnostics}")
        expected_reasons = {
            reason
            for enabled, reason in (
                (expected_lowercase, "thai_side_suspicious_lowercase_latin"),
                (expected_mixed, "thai_latin_mixed_token"),
            )
            if enabled
        }
        if set(reasons) != expected_reasons or len(reasons) != len(set(reasons)):
            raise AssertionError(
                f"{name}: expected reasons {sorted(expected_reasons)}, got {reasons}"
            )
        results.append({"case": name, "passed": True, "reasons": reasons, **diagnostics})
    # ASCII escapes keep this model-free test printable in Windows GBK consoles.
    print(json.dumps({"valid": True, "cases": results}, ensure_ascii=True, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_RAW_GENERATIONS)
    parser.add_argument("--audit-all", type=Path, default=DEFAULT_AUDIT_ALL)
    parser.add_argument("--accepted-file", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--rejected-file", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the model-free v3 Latin anomaly regression cases and exit.",
    )
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": round(min(values), 6) if values else None,
        "mean": round(statistics.fmean(values), 6) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": round(max(values), 6) if values else None,
    }


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_latin_self_test()
    input_path = resolve_path(args.input_file)
    audit_path = resolve_path(args.audit_all)
    accepted_path = resolve_path(args.accepted_file)
    rejected_path = resolve_path(args.rejected_file)
    summary_path = resolve_path(args.summary_file)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    ensure_new_outputs((audit_path, accepted_path, rejected_path, summary_path), args.overwrite)
    config = load_config(resolve_path(args.config))
    quality_config = config.get("teacher_output_quality") or {}
    min_thai_chars = int(quality_config.get("min_thai_chars", 20))
    max_repeat_score = float(quality_config.get("max_repeat_score_20", 0.5))

    candidate_ids: set[str] = set()
    pair_keys: set[str] = set()
    source_to_target: dict[str, str] = {}
    target_to_source: dict[str, str] = {}
    rejection_reasons: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    accepted_domains: Counter[str] = Counter()
    rejected_domains: Counter[str] = Counter()
    duplicates: Counter[str] = Counter()
    source_lengths: list[float] = []
    target_lengths: list[float] = []
    thai_counts: list[float] = []
    thai_ratios: list[float] = []
    repeat_scores: list[float] = []
    total = 0
    accepted_total = 0

    with atomic_text_writer(audit_path) as audit_handle, atomic_text_writer(
        accepted_path
    ) as accepted_handle, atomic_text_writer(rejected_path) as rejected_handle:
        for line_number, raw in read_jsonl(input_path):
            total += 1
            missing = REQUIRED_FIELDS - set(raw)
            if missing:
                raise ValueError(f"{input_path}:{line_number} missing fields: {sorted(missing)}")
            if raw["domain"] not in DOMAINS:
                raise ValueError(f"{input_path}:{line_number} has invalid domain {raw['domain']!r}")
            if raw["source_lang"] != "zh-CN" or raw["target_lang"] != "th":
                raise ValueError(f"{input_path}:{line_number} is not zh-CN->th")

            result = assess_pair(raw["source_text"], raw["target_text"])
            reasons = list(result["reject_reasons"])
            review_reasons = list(result["review_reasons"])
            warnings = list(result["quality_flags"])
            latin_check = apply_v3_latin_rules(result["th_text"], reasons)
            score = repeat_score(result["th_text"])
            thai_count = count_thai(result["th_text"])
            if thai_count < min_thai_chars:
                reasons.append("target_too_short")
            if score >= max_repeat_score:
                reasons.append("repeat_score_20_too_high")

            candidate_id = str(raw["candidate_id"])
            if candidate_id in candidate_ids:
                reasons.append("duplicate_candidate_id")
                duplicates["duplicate_candidate_id"] += 1
            candidate_ids.add(candidate_id)
            pair_key = result["normalized_pair_sha256"]
            source_key = result["normalized_zh_key"]
            target_key = result["normalized_th_key"]
            if pair_key in pair_keys:
                reasons.append("duplicate_pair")
                duplicates["duplicate_pair"] += 1
            pair_keys.add(pair_key)
            if source_key in source_to_target:
                reasons.append("duplicate_source")
                duplicates["duplicate_source"] += 1
                if source_to_target[source_key] != target_key:
                    reasons.append("one_source_multiple_targets")
                    duplicates["one_source_multiple_targets"] += 1
            else:
                source_to_target[source_key] = target_key
            if target_key in target_to_source:
                reasons.append("duplicate_target")
                duplicates["duplicate_target"] += 1
                if target_to_source[target_key] != source_key:
                    reasons.append("one_target_multiple_sources")
                    duplicates["one_target_multiple_sources"] += 1
            else:
                target_to_source[target_key] = source_key

            reasons = list(dict.fromkeys(reasons))
            review_reasons = list(dict.fromkeys(review_reasons))
            warnings = list(dict.fromkeys(warnings))
            if reasons:
                quality_status = "rejected"
                accepted = False
                final_reasons = reasons
            elif review_reasons:
                quality_status = "review"
                accepted = False
                final_reasons = review_reasons
            else:
                quality_status = "accepted"
                accepted = True
                final_reasons = []

            metrics = {
                **result["metrics"],
                "repeat_score": round(score, 6),
                "repeat_ngram_size": 20,
                "thai_ratio": result["metrics"]["th_script"]["thai_ratio"],
                "source_length": len(result["zh_text"]),
                "target_length": len(result["th_text"]),
                "thai_char_count": thai_count,
                "v3_latin_check": latin_check,
            }
            audited = {
                **raw,
                "source_text": result["zh_text"],
                "target_text": result["th_text"],
                "thai_char_count": thai_count,
                "accepted": accepted,
                "quality_status": quality_status,
                "reasons": final_reasons,
                "warnings": warnings,
                "metrics": metrics,
                "pair_sha256": result["pair_sha256"],
                "normalized_pair_sha256": result["normalized_pair_sha256"],
                "quality_rules_version": QUALITY_RULE_VERSION,
                "semantic_qe_score": None,
                "semantic_status": "not_evaluated",
                "audited_at": utc_now(),
                "pipeline_version": PIPELINE_VERSION,
            }
            serialized = json.dumps(audited, ensure_ascii=False, separators=(",", ":")) + "\n"
            audit_handle.write(serialized)
            if accepted:
                accepted_handle.write(serialized)
                accepted_total += 1
                accepted_domains[raw["domain"]] += 1
            else:
                rejected_handle.write(serialized)
                rejected_domains[raw["domain"]] += 1
                rejection_reasons.update(final_reasons)
            status_counts[quality_status] += 1
            warning_counts.update(warnings)
            source_lengths.append(float(len(result["zh_text"])))
            target_lengths.append(float(len(result["th_text"])))
            thai_counts.append(float(thai_count))
            thai_ratios.append(float(metrics["thai_ratio"]))
            repeat_scores.append(float(score))

    rejected_total = total - accepted_total
    summary = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "stage": "teacher_output_rule_audit",
        "created_at": utc_now(),
        "input_file": {"path": relative_path(input_path), "sha256": sha256_file(input_path)},
        "quality_rules": {
            "module": "tools/zh_th_trans/quality_rules_v2.py",
            "version": QUALITY_RULE_VERSION,
            "min_thai_chars": min_thai_chars,
            "repeat_score_ngram_size": 20,
            "max_repeat_score_20": max_repeat_score,
        },
        "total": total,
        "accepted": accepted_total,
        "rejected": rejected_total,
        "accept_rate": round(accepted_total / max(total, 1), 6),
        "quality_status_counts": dict(status_counts),
        "reject_reason_counts": dict(sorted(rejection_reasons.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "accepted_domain_counts": {domain: accepted_domains[domain] for domain in DOMAINS},
        "rejected_domain_counts": {domain: rejected_domains[domain] for domain in DOMAINS},
        "length_statistics": {
            "source_codepoints": numeric_summary(source_lengths),
            "target_codepoints": numeric_summary(target_lengths),
            "thai_characters": numeric_summary(thai_counts),
            "thai_ratio": numeric_summary(thai_ratios),
        },
        "repeat_distribution": {
            "score": numeric_summary(repeat_scores),
            "at_least_0_10": sum(score >= 0.10 for score in repeat_scores),
            "at_least_0_20": sum(score >= 0.20 for score in repeat_scores),
            "at_least_0_50": sum(score >= 0.50 for score in repeat_scores),
        },
        "duplicate_counts": dict(duplicates),
        "outputs": {
            "audit_all": relative_path(audit_path),
            "accepted": relative_path(accepted_path),
            "rejected": relative_path(rejected_path),
        },
        "semantic_quality_note": (
            "Rule filtering is not semantic translation evaluation; human review or optional QE is still required."
        ),
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Teacher output audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
