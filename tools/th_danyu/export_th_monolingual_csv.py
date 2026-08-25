#!/usr/bin/env python3
"""Export cleaned Thai Wikipedia monolingual JSONL as a submission CSV.

The source JSONL remains untouched.  This exporter streams one record at a
time, preserves the cleaned text exactly as supplied, and writes an Excel-safe
UTF-8 CSV with traceable source and licence metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "dataset" / "external" / "thwiki" / "cleaned" / "th_monolingual.jsonl"
DEFAULT_OUTPUT = ROOT / "dataset" / "final" / "th" / "th_monolingual.csv"
DEFAULT_REPORT = ROOT / "dataset" / "final" / "th" / "th_monolingual_export_report.json"
COLUMNS = [
    "id",
    "language",
    "text",
    "th_char_count",
    "dataset_name",
    "data_origin",
    "source_url",
    "license",
    "license_url",
    "text_sha256",
]
THAI_RE = re.compile(r"[\u0e00-\u0e7f]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Thai monolingual JSONL as final CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true", help="Check input and print statistics only.")
    return parser.parse_args()


def empty_stats() -> dict[str, Any]:
    return {
        "input_rows": 0,
        "blank_lines": 0,
        "invalid_json": 0,
        "invalid_structure": 0,
        "missing_required_field": 0,
        "non_thai_text": 0,
        "duplicate_id": 0,
        "duplicate_text_sha256": 0,
        "exported_rows": 0,
        "dataset_name_counts": Counter(),
        "license_counts": Counter(),
    }


def csv_row(record: dict[str, Any]) -> dict[str, str | int]:
    return {
        "id": record["id"],
        "language": record["language"],
        "text": record["text"],
        "th_char_count": record["th_char_count"],
        "dataset_name": record["dataset_name"],
        "data_origin": record["data_origin"],
        "source_url": record["source_url"],
        "license": record["license"],
        "license_url": record["license_url"],
        "text_sha256": record["text_sha256"],
    }


def valid_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    required = ["id", "language", "text", "th_char_count", "dataset_name", "data_origin", "source_url", "license", "license_url", "text_sha256"]
    if any(field not in record or record[field] is None or str(record[field]).strip() == "" for field in required):
        return False
    if record["language"] != "th" or not isinstance(record["text"], str) or THAI_RE.search(record["text"]) is None:
        return False
    try:
        return int(record["th_char_count"]) >= 0
    except (TypeError, ValueError):
        return False


def export(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {args.input}")

    stats = empty_stats()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    temporary_path: Path | None = None
    writer: csv.DictWriter | None = None
    handle = None
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", delete=False, dir=args.output.parent, suffix=".tmp")
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()

    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line in source:
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
                if not valid_record(record):
                    if isinstance(record.get("text"), str) and record.get("text") and THAI_RE.search(record["text"]) is None:
                        stats["non_thai_text"] += 1
                    else:
                        stats["missing_required_field"] += 1
                    continue
                record_id = str(record["id"])
                text_hash = str(record["text_sha256"])
                if record_id in seen_ids:
                    stats["duplicate_id"] += 1
                    continue
                if text_hash in seen_hashes:
                    stats["duplicate_text_sha256"] += 1
                    continue
                seen_ids.add(record_id)
                seen_hashes.add(text_hash)
                stats["exported_rows"] += 1
                stats["dataset_name_counts"][str(record["dataset_name"])] += 1
                stats["license_counts"][str(record["license"])] += 1
                if writer is not None:
                    writer.writerow(csv_row(record))
    except Exception:
        if handle is not None:
            handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    else:
        if handle is not None:
            handle.close()
            assert temporary_path is not None
            temporary_path.replace(args.output)

    return {
        "schema_version": 1,
        "stage": "thai_monolingual_csv_export",
        "input_file": str(args.input),
        "output_file": str(args.output),
        "columns": COLUMNS,
        "minimum_text_length": None,
        "statistics": {key: dict(value) if isinstance(value, Counter) else value for key, value in stats.items()},
    }


def main() -> int:
    args = parse_args()
    report = export(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
