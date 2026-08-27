#!/usr/bin/env python3
"""Independently audit the final Chinese--Thai v2 CSV.

This program deliberately starts from the CSV rather than trusting merge/build
reports. It re-runs the shared source-agnostic rules, reconstructs direction-
neutral pairs, checks pair and side uniqueness, and emits a deterministic
manual-review sample. It never modifies the input CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .quality_rules_v2 import QUALITY_RULE_VERSION, assess_pair
except ImportError:  # Direct script execution.
    from quality_rules_v2 import QUALITY_RULE_VERSION, assess_pair


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "final" / "zh_th" / "zh_th_clean_v2.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "dataset" / "final" / "zh_th" / "audit_v2.json"
DEFAULT_SAMPLE = PROJECT_ROOT / "dataset" / "final" / "zh_th" / "sample_review_v2.csv"
REQUIRED_COLUMNS = {
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "zh_char_count",
    "translation_method",
}
VALID_DIRECTIONS = {("zh-CN", "th"), ("th", "zh-CN")}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the final zh-th v2 CSV from scratch.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--sample-review", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_sample(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    columns = [
        "csv_row",
        "pair_group_id",
        "source_lang",
        "target_lang",
        "source_text",
        "target_text",
        "zh_char_count",
        "translation_method",
        "dataset_name",
        "data_origin",
        "audit_decision",
        "audit_flags",
        "review_status",
        "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Final CSV does not exist: {input_path}")

    rows: list[dict[str, Any]] = []
    directions: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    review_reasons: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    exact_pairs: Counter[str] = Counter()
    normalized_pairs: Counter[str] = Counter()
    zh_to_th_keys: dict[str, set[str]] = defaultdict(set)
    th_to_zh_keys: dict[str, set[str]] = defaultdict(set)
    stored_pair_mismatch = 0
    stored_pair_group_mismatch = 0
    invalid_direction = 0
    empty_text = 0
    invalid_zh_count = 0
    severe_repetition = 0
    foreign_letter_mark = 0
    large_abnormal_english = 0
    thai_ratio_below_080 = 0
    mojibake = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for csv_row, record in enumerate(reader, start=2):
            source_lang = str(record.get("source_lang", "")).strip()
            target_lang = str(record.get("target_lang", "")).strip()
            source_text = str(record.get("source_text", ""))
            target_text = str(record.get("target_text", ""))
            direction = f"{source_lang}->{target_lang}"
            directions[direction] += 1
            if (source_lang, target_lang) not in VALID_DIRECTIONS:
                invalid_direction += 1
                zh_text, th_text = "", ""
            elif source_lang == "zh-CN":
                zh_text, th_text = source_text, target_text
            else:
                zh_text, th_text = target_text, source_text
            if not source_text.strip() or not target_text.strip():
                empty_text += 1

            result = assess_pair(zh_text, th_text)
            decisions[result["decision"]] += 1
            reject_reasons.update(result["reject_reasons"])
            review_reasons.update(result["review_reasons"])
            quality_flags.update(result["quality_flags"])
            exact_pairs[result["pair_sha256"]] += 1
            normalized_pairs[result["normalized_pair_sha256"]] += 1
            zh_to_th_keys[result["normalized_zh_key"]].add(result["normalized_th_key"])
            th_to_zh_keys[result["normalized_th_key"]].add(result["normalized_zh_key"])

            repetition = result["metrics"]["repetition"]
            if (
                repetition["continuous_repeat_span"] >= 40
                or (
                    len(result["th_text"]) >= 100
                    and repetition["unique_8gram_ratio"] < 0.20
                    and repetition["compression_ratio"] < 0.20
                )
            ):
                severe_repetition += 1
            if result["metrics"]["th_foreign_letter_marks"]:
                foreign_letter_mark += 1
            if "large_abnormal_english" in result["review_reasons"]:
                large_abnormal_english += 1
            if result["metrics"]["th_script"]["thai_ratio"] < 0.80:
                thai_ratio_below_080 += 1
            if "mojibake" in result["reject_reasons"]:
                mojibake += 1

            stored_hash = str(record.get("pair_sha256", "")).strip()
            if stored_hash and stored_hash != result["pair_sha256"]:
                stored_pair_mismatch += 1
            expected_group = f"zh_th_pair_{result['normalized_pair_sha256']}"
            stored_group = str(record.get("pair_group_id", "")).strip()
            if stored_group and stored_group != expected_group:
                stored_pair_group_mismatch += 1
            try:
                if int(str(record.get("zh_char_count", "")).strip()) != result["metrics"][
                    "zh_script"
                ]["han_count"]:
                    invalid_zh_count += 1
            except ValueError:
                invalid_zh_count += 1

            rows.append(
                {
                    **record,
                    "csv_row": csv_row,
                    "pair_group_id": stored_group or expected_group,
                    "audit_decision": result["decision"],
                    "audit_flags": ";".join(
                        result["reject_reasons"]
                        + result["review_reasons"]
                        + result["quality_flags"]
                    ),
                    "review_status": "pending",
                    "review_notes": "",
                }
            )

    exact_duplicate_groups = sum(count > 1 for count in exact_pairs.values())
    exact_duplicate_extra = sum(count - 1 for count in exact_pairs.values() if count > 1)
    normalized_duplicate_groups = sum(count > 1 for count in normalized_pairs.values())
    normalized_duplicate_extra = sum(count - 1 for count in normalized_pairs.values() if count > 1)
    same_source_multiple_target = sum(len(targets) > 1 for targets in zh_to_th_keys.values())
    same_target_multiple_source = sum(len(sources) > 1 for sources in th_to_zh_keys.values())
    total = len(rows)
    acceptance_checks = {
        "total_is_20000": total == 20_000,
        "zh_to_th_is_10000": directions.get("zh-CN->th", 0) == 10_000,
        "th_to_zh_is_10000": directions.get("th->zh-CN", 0) == 10_000,
        "empty_text_is_zero": empty_text == 0,
        "invalid_direction_is_zero": invalid_direction == 0,
        "exact_pair_duplicates_are_zero": exact_duplicate_extra == 0,
        "normalized_pair_duplicates_are_zero": normalized_duplicate_extra == 0,
        "severe_repetition_is_zero": severe_repetition == 0,
        "foreign_letter_mark_pollution_is_zero": foreign_letter_mark == 0,
        "mojibake_is_zero": mojibake == 0,
        "quality_reject_is_zero": decisions.get("reject", 0) == 0,
        "quality_review_is_zero": decisions.get("review", 0) == 0,
        "same_source_multiple_target_is_zero": same_source_multiple_target == 0,
        "same_target_multiple_source_is_zero": same_target_multiple_source == 0,
        "stored_pair_hashes_match": stored_pair_mismatch == 0,
        "stored_pair_groups_match": stored_pair_group_mismatch == 0,
    }
    report = {
        "schema_version": 2,
        "stage": "independent_final_zh_th_v2_audit",
        "generated_at": utc_now(),
        "input_file": str(input_path),
        "quality_rule_version": QUALITY_RULE_VERSION,
        "csv_columns": columns,
        "total_records": total,
        "direction_counts": dict(directions),
        "empty_text": empty_text,
        "invalid_direction": invalid_direction,
        "invalid_zh_char_count": invalid_zh_count,
        "quality_decisions": dict(decisions),
        "reject_reason_counts": dict(reject_reasons),
        "review_reason_counts": dict(review_reasons),
        "quality_flag_counts": dict(quality_flags),
        "exact_pair_duplicate_groups": exact_duplicate_groups,
        "exact_pair_duplicate_extra_records": exact_duplicate_extra,
        "normalized_pair_duplicate_groups": normalized_duplicate_groups,
        "normalized_pair_duplicate_extra_records": normalized_duplicate_extra,
        "severe_repetition": severe_repetition,
        "foreign_letter_mark_pollution": foreign_letter_mark,
        "large_abnormal_english": large_abnormal_english,
        "thai_script_ratio_below_0_80": thai_ratio_below_080,
        "mojibake": mojibake,
        "same_source_multiple_target_groups": same_source_multiple_target,
        "same_target_multiple_source_groups": same_target_multiple_source,
        "stored_pair_sha256_mismatch": stored_pair_mismatch,
        "stored_pair_group_id_mismatch": stored_pair_group_mismatch,
        "acceptance_checks": acceptance_checks,
        "passed": all(acceptance_checks.values()),
    }
    return report, rows


def main() -> int:
    args = parse_args()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    input_path = resolve_path(args.input)
    audit_path = resolve_path(args.audit)
    sample_path = resolve_path(args.sample_review)
    report, rows = audit(input_path)
    sample = list(rows)
    random.Random(args.seed).shuffle(sample)
    sample = sample[: min(args.sample_size, len(sample))]
    atomic_write_json(audit_path, report)
    write_sample(sample_path, sample)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit: {audit_path}")
    print(f"sample review: {sample_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
