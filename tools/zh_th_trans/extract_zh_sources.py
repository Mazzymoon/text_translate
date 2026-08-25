#!/usr/bin/env python3
"""Extract all native Chinese source records for local Chinese-Thai translation.
这个脚本用于从中英清洗语料中提取“原生中文源文本”，为后续中译泰提供统一输入。
This script deliberately performs no text selection, deduplication, rewriting, or
translation.  It copies every record whose source language is ``zh-CN`` from the
three existing Chinese-English cleaned files into one traceable Chinese-Thai
translation-input JSON document.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "crawled" / "zh_en" / "cleaned"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "zh_source.json"
DOMAINS = ("education", "technology", "finance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all zh-CN source records from the three crawled zh-en "
            "cleaned files for later local Qwen Chinese-Thai translation."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing the three cleaned JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read, validate, and report counts without writing the output file.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError(f"Expected an object with a records array: {path}")
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_han(text: str) -> int:
    count = 0
    for char in text:
        codepoint = ord(char)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2FA1F
            or 0x30000 <= codepoint <= 0x323AF
        ):
            count += 1
    return count


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def copy_source_record(record: dict[str, Any], domain: str, input_path: Path) -> dict[str, Any]:
    record_id = str(record.get("id", "")).strip()
    source_text = record.get("source_text")
    if not record_id:
        raise ValueError(f"A zh-CN source record in {input_path} has no id")
    if not isinstance(source_text, str) or not source_text:
        raise ValueError(f"Chinese source record {record_id} in {input_path} has empty source_text")
    if record.get("domain") != domain:
        raise ValueError(
            f"Domain mismatch for {record_id}: file={domain}, record={record.get('domain')!r}"
        )

    calculated_count = count_han(source_text)
    recorded_count = record.get("zh_char_count")
    if recorded_count is not None and recorded_count != calculated_count:
        raise ValueError(
            f"Chinese character count mismatch for {record_id}: "
            f"recorded={recorded_count}, calculated={calculated_count}"
        )

    provenance = copy.deepcopy(record.get("provenance") or {})
    provenance["source_clean_record_id"] = record_id
    provenance["source_clean_file"] = project_relative(input_path)

    quality = copy.deepcopy(record.get("quality") or {})
    quality["source_sha256"] = sha256_text(source_text)
    quality["source_han_count"] = calculated_count

    return {
        "id": record_id,
        "language_pair": "zh_th",
        "source_lang": "zh-CN",
        "target_lang": "th",
        "source_text": source_text,
        "target_text": "",
        "zh_char_count": calculated_count,
        "domain": domain,
        "translation_method": None,
        "status": "needs_translation",
        "provenance": provenance,
        "quality": quality,
    }


def build_document(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_records: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    extracted_counts: Counter[str] = Counter()

    for domain in DOMAINS:
        input_path = input_dir / f"{domain}.json"
        if not input_path.is_file():
            raise FileNotFoundError(f"Required cleaned input file does not exist: {input_path}")

        document = read_json(input_path)
        if document.get("domain") not in (None, domain):
            raise ValueError(
                f"Input document domain mismatch in {input_path}: {document.get('domain')!r}"
            )

        records = document["records"]
        input_counts[domain] = len(records)
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record found in {input_path}")
            if record.get("source_lang") != "zh-CN":
                continue
            output_records.append(copy_source_record(record, domain, input_path))
            extracted_counts[domain] += 1

    ids = [record["id"] for record in output_records]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate source record IDs found: {duplicates[:10]}")

    text_hashes = [record["quality"]["source_sha256"] for record in output_records]
    duplicate_text_records = len(text_hashes) - len(set(text_hashes))
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = {
        "input_records_by_domain": input_counts,
        "extracted_zh_source_records_by_domain": {
            domain: extracted_counts[domain] for domain in DOMAINS
        },
        "total_extracted_records": len(output_records),
        "duplicate_source_text_records_reported_only": duplicate_text_records,
        "selection_rule": 'source_lang == "zh-CN"',
        "text_modified": False,
        "deduplicated": False,
        "translated": False,
    }
    document = {
        "schema_version": 1,
        "language_pair": "zh_th",
        "stage": "qwen_translation_input",
        "generated_at": generated_at,
        "source_language": "zh-CN",
        "target_language": "th",
        "records": output_records,
        "extraction_summary": summary,
    }
    return document, summary


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def print_summary(summary: dict[str, Any], output: Path, dry_run: bool) -> None:
    print("domain       input records   extracted zh-CN")
    for domain in DOMAINS:
        print(
            f"{domain:<12}"
            f"{summary['input_records_by_domain'][domain]:>13,}"
            f"{summary['extracted_zh_source_records_by_domain'][domain]:>18,}"
        )
    print(f"total extracted: {summary['total_extracted_records']:,}")
    print(
        "duplicate source texts (reported, not removed): "
        f"{summary['duplicate_source_text_records_reported_only']:,}"
    )
    if dry_run:
        print("dry-run: no output file was written")
    else:
        print(f"written: {output}")


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output = resolve_path(args.output)
    try:
        document, summary = build_document(input_dir)
        if not args.dry_run:
            write_json_atomic(output, document)
        print_summary(summary, output, args.dry_run)
        return 0
    except (OSError, ValueError) as error:
        print(f"Chinese source extraction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
