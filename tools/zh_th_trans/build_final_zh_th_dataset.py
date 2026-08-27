#!/usr/bin/env python3
"""Build the submission CSV for the Chinese--Thai parallel corpus.

The input is the direction-neutral merged JSONL.  The script performs no
translation or text rewriting: it only validates unique zh/th pairs, applies a
stable shuffle, and exposes each pair once in one of the two translation
directions.  The submission file deliberately contains only the project's
six CSV columns. Chinese--Thai data is not collected as a three-domain corpus,
so its submission CSV has no ``domain`` column. Any domain metadata in the
intermediate file is retained there for provenance but is not exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "cleaned" / "zh_th_merged.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "final" / "zh_th" / "zh_th.csv"
DEFAULT_REPORT = PROJECT_ROOT / "dataset" / "final" / "zh_th" / "zh_th_build_report.json"
CSV_COLUMNS = [
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "zh_char_count",
    "translation_method",
]
TRACEABILITY_COLUMNS = [
    "pair_group_id",
    "pair_sha256",
    "dataset_name",
    "data_origin",
    "provenance",
]
ALLOWED_METHODS = {"human", "google_mt", "llm_mt"}
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ebef]")
THAI_RE = re.compile(r"[\u0e00-\u0e7f]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the final zh-th submission CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Merged zh-th JSONL input.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final CSV path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Build report path.")
    parser.add_argument("--records-per-direction", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814, help="Stable allocation seed.")
    parser.add_argument(
        "--include-traceability",
        action="store_true",
        help="Append pair/provenance columns required by the v2 corpus.",
    )
    parser.add_argument(
        "--require-quality-v2",
        action="store_true",
        help="Reject inputs that were not accepted by the shared v2 quality rules.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files.")
    return parser.parse_args()


def canonical_pair_hash(zh_text: str, th_text: str) -> str:
    canonical = unicodedata.normalize("NFC", zh_text) + "\x1f" + unicodedata.normalize("NFC", th_text).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_han(text: str) -> int:
    return len(HAN_RE.findall(text))


def load_pairs(
    input_path: Path, *, require_quality_v2: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")

    stats: dict[str, Any] = {
        "input_rows": 0,
        "blank_lines": 0,
        "invalid_json": 0,
        "invalid_structure": 0,
        "missing_text": 0,
        "language_side_error": 0,
        "duplicate_pairs_skipped": 0,
        "stored_hash_mismatch": 0,
        "invalid_translation_method": 0,
        "non_v2_quality_record": 0,
        "input_domain_counts": Counter(),
        "dataset_name_counts": Counter(),
    }
    pairs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            stats["input_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                stats["invalid_structure"] += 1
                continue

            zh_text = record.get("zh_text")
            th_text = record.get("th_text")
            if not isinstance(zh_text, str) or not isinstance(th_text, str) or not zh_text.strip() or not th_text.strip():
                stats["missing_text"] += 1
                continue
            if count_han(zh_text) == 0 or THAI_RE.search(th_text) is None:
                stats["language_side_error"] += 1
                continue

            pair_sha256 = canonical_pair_hash(zh_text, th_text)
            stored_hash = record.get("pair_sha256")
            if isinstance(stored_hash, str) and stored_hash and stored_hash != pair_sha256:
                stats["stored_hash_mismatch"] += 1
            if pair_sha256 in seen_hashes:
                stats["duplicate_pairs_skipped"] += 1
                continue
            seen_hashes.add(pair_sha256)

            method = record.get("translation_method")
            if method not in ALLOWED_METHODS:
                stats["invalid_translation_method"] += 1
                continue

            if require_quality_v2 and not str(record.get("quality_rule_version", "")).startswith(
                "zh_th_quality_v2"
            ):
                stats["non_v2_quality_record"] += 1
                continue

            original_domain = record.get("domain")
            stats["input_domain_counts"][str(original_domain)] += 1

            dataset_name = record.get("dataset_name")
            if isinstance(dataset_name, str) and dataset_name:
                stats["dataset_name_counts"][dataset_name] += 1
            pairs.append(
                {
                    "line_number": line_number,
                    "id": record.get("id"),
                    "zh_text": zh_text,
                    "th_text": th_text,
                    "zh_char_count": count_han(zh_text),
                    "original_domain": original_domain,
                    "translation_method": method,
                    "pair_sha256": pair_sha256,
                    "dataset_name": dataset_name,
                    "data_origin": record.get("data_origin"),
                    "pair_group_id": record.get("pair_group_id")
                    or f"zh_th_pair_{record.get('normalized_pair_sha256') or pair_sha256}",
                    "provenance": record.get("provenance") or {},
                }
            )
    return pairs, stats


def allocate_rows(
    pairs: list[dict[str, Any]],
    records_per_direction: int,
    seed: int,
    *,
    include_traceability: bool = False,
) -> list[dict[str, str | int]]:
    needed = records_per_direction * 2
    selected = list(pairs)
    random.Random(seed).shuffle(selected)
    selected = selected[:needed]
    rows: list[dict[str, str | int]] = []
    for index, pair in enumerate(selected):
        zh_to_th = index < records_per_direction
        row: dict[str, str | int] = {
                "source_lang": "zh-CN" if zh_to_th else "th",
                "target_lang": "th" if zh_to_th else "zh-CN",
                "source_text": pair["zh_text"] if zh_to_th else pair["th_text"],
                "target_text": pair["th_text"] if zh_to_th else pair["zh_text"],
                "zh_char_count": pair["zh_char_count"],
                "translation_method": pair["translation_method"],
            }
        if include_traceability:
            row.update(
                {
                    "pair_group_id": pair["pair_group_id"],
                    "pair_sha256": pair["pair_sha256"],
                    "dataset_name": pair.get("dataset_name") or "",
                    "data_origin": pair.get("data_origin") or "",
                    "provenance": json.dumps(
                        pair.get("provenance") or {}, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        rows.append(row)
    return rows


def write_csv_atomic(
    output_path: Path,
    rows: list[dict[str, str | int]],
    *,
    include_traceability: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", delete=False, dir=output_path.parent, suffix=".tmp") as handle:
        temporary_path = Path(handle.name)
        columns = CSV_COLUMNS + TRACEABILITY_COLUMNS if include_traceability else CSV_COLUMNS
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)


def write_json_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_path.parent, suffix=".tmp") as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(output_path)


def main() -> int:
    args = parse_args()
    if args.records_per_direction <= 0:
        raise ValueError("--records-per-direction must be positive")

    pairs, stats = load_pairs(args.input, require_quality_v2=args.require_quality_v2)
    needed = args.records_per_direction * 2
    can_fill = len(pairs) >= needed
    export_ready = can_fill
    selected = (
        allocate_rows(
            pairs,
            args.records_per_direction,
            args.seed,
            include_traceability=args.include_traceability,
        )
        if can_fill
        else []
    )
    direction_counts = Counter(f"{row['source_lang']}->{row['target_lang']}" for row in selected)

    report = {
        "schema_version": 1,
        "stage": "zh_th_submission_csv_build",
        "input_file": str(args.input),
        "output_file": str(args.output),
        "seed": args.seed,
        "records_per_direction": args.records_per_direction,
        "required_total_records": needed,
        "input_checks": {key: dict(value) if isinstance(value, Counter) else value for key, value in stats.items()},
        "unique_usable_pairs": len(pairs),
        "can_fill_requested_total": can_fill,
        "export_ready": export_ready,
        "selected_total": len(selected),
        "selected_direction_counts": dict(direction_counts),
        "selected_translation_method_counts": dict(Counter(row["translation_method"] for row in selected)),
        "include_traceability": args.include_traceability,
        "csv_columns": CSV_COLUMNS + TRACEABILITY_COLUMNS if args.include_traceability else CSV_COLUMNS,
        "notes": [
            "source_text and target_text are direction-specific final CSV fields.",
            "zh_text and th_text remain only in the intermediate JSONL as direction-neutral canonical fields.",
            "domain is intentionally omitted from the Chinese--Thai submission CSV.",
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if not can_fill:
        raise RuntimeError(f"Only {len(pairs)} unique usable pairs are available; {needed} are required.")
    selected_pair_hashes = {
        str(row.get("pair_sha256"))
        if row.get("pair_sha256")
        else canonical_pair_hash(
            str(row["source_text"] if row["source_lang"] == "zh-CN" else row["target_text"]),
            str(row["target_text"] if row["target_lang"] == "th" else row["source_text"]),
        )
        for row in selected
    }
    if len(selected_pair_hashes) != len(selected):
        # This should not happen because source/target swapping does not alter the
        # canonical zh/th pair; it protects future changes to allocation code.
        raise RuntimeError("Internal error: selected rows are not pair-unique")

    write_csv_atomic(args.output, selected, include_traceability=args.include_traceability)
    write_json_atomic(args.report, report)
    print(f"Wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise, actionable command-line failure
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
