#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "external" / "translatefx" / "raw"
DEFAULT_CLEANED_FILE = (
    PROJECT_ROOT / "dataset" / "external" / "translatefx" / "cleaned" / "finance.json"
)
DEFAULT_REJECTED_FILE = (
    PROJECT_ROOT / "dataset" / "external" / "translatefx" / "rejected" / "finance.json"
)

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,500}>")
BOILERPLATE_RE = re.compile(
    r"^(?:Ends(?:/|$)|Issued\s+at\s+HKT\b|NNNN\b)",
    re.IGNORECASE,
)
HEADER_EN = {"en", "english", "en_text", "english_text"}
HEADER_ZH = {"zh", "zh-cn", "chinese", "zh_text", "chinese_text"}

SOURCE_URL = "https://translatefx.s3-us-west-2.amazonaws.com/downloads/HK-FSO.tsv"
REFERRER_URL = "https://www.translatefx.com/"
RIGHTS_NOTE = (
    "Publicly downloadable corpus, but no explicit redistribution or model-training "
    "license was found on the TranslateFX download page or Terms of Service. Review "
    "rights before public redistribution."
)


@dataclass(frozen=True)
class RawPair:
    source_file: Path
    row_number: int
    line_start: int
    line_end: int
    en_text: str
    zh_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean TranslateFX TSV/CSV files into direction-independent parallel JSON."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Input TSV/CSV file. Repeat for multiple files. Defaults to every file in --raw-dir.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLEANED_FILE)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED_FILE)
    parser.add_argument("--min-han", type=int, default=100)
    parser.add_argument("--max-han", type=int, default=220)
    parser.add_argument("--min-english-words", type=int, default=30)
    parser.add_argument("--max-english-words", type=int, default=180)
    parser.add_argument("--min-row-han", type=int, default=3)
    parser.add_argument("--min-row-english-words", type=int, default=3)
    parser.add_argument("--min-length-ratio", type=float, default=0.70)
    parser.add_argument("--max-length-ratio", type=float, default=3.50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def normalize_text(value: str) -> str:
    text = html.unescape(str(value)).replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def count_han(value: str) -> int:
    return len(HAN_RE.findall(value))


def count_english_words(value: str) -> int:
    return len(ENGLISH_WORD_RE.findall(value))


def looks_like_header(en_text: str, zh_text: str) -> bool:
    return en_text.casefold().strip() in HEADER_EN and zh_text.casefold().strip() in HEADER_ZH


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    paths = args.input or sorted(
        path for path in args.raw_dir.glob("*") if path.suffix.casefold() in {".tsv", ".csv"}
    )
    resolved: list[Path] = []
    for path in paths:
        candidate = path if path.is_absolute() else (PROJECT_ROOT / path)
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Input file does not exist: {candidate}")
        if candidate.suffix.casefold() not in {".tsv", ".csv"}:
            raise ValueError(f"Unsupported input format: {candidate}")
        resolved.append(candidate)
    if not resolved:
        raise FileNotFoundError(f"No TSV/CSV files found in {args.raw_dir}")
    return sorted(set(resolved))


def iter_tsv(path: Path) -> Iterator[tuple[int, int, int, list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            yield line_number, line_number, line_number, line.rstrip("\r\n").split("\t")


def iter_csv(path: Path) -> Iterator[tuple[int, int, int, list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        previous_line = 0
        for row_number, row in enumerate(reader, start=1):
            line_start = previous_line + 1
            line_end = reader.line_num
            previous_line = line_end
            yield row_number, line_start, line_end, row


def iter_rows(path: Path) -> Iterator[tuple[int, int, int, list[str]]]:
    if path.suffix.casefold() == ".tsv":
        yield from iter_tsv(path)
    else:
        yield from iter_csv(path)


def rejection_record(
    reason: str,
    rows: Iterable[RawPair],
    *,
    details: str | None = None,
) -> dict:
    grouped = list(rows)
    if not grouped:
        raise ValueError("A rejection must reference at least one raw pair")
    en_text = " ".join(row.en_text for row in grouped).strip()
    zh_text = " ".join(row.zh_text for row in grouped).strip()
    identity = "\x1f".join(
        [
            reason,
            project_path(grouped[0].source_file),
            str(grouped[0].line_start),
            str(grouped[-1].line_end),
            en_text,
            zh_text,
        ]
    )
    record = {
        "id": f"translatefx_reject_{sha256_text(identity)[:20]}",
        "reason": reason,
        "source_file": project_path(grouped[0].source_file),
        "row_number_start": grouped[0].row_number,
        "row_number_end": grouped[-1].row_number,
        "line_start": grouped[0].line_start,
        "line_end": grouped[-1].line_end,
        "raw_pair_count": len(grouped),
        "en_text": en_text,
        "zh_text": zh_text,
    }
    if details:
        record["details"] = details
    return record


def validate_row(row: RawPair, args: argparse.Namespace) -> tuple[str | None, str | None]:
    if not row.en_text or not row.zh_text:
        return "empty_field", None
    if looks_like_header(row.en_text, row.zh_text):
        return "header_row", None
    if BOILERPLATE_RE.match(row.en_text):
        return "publication_boilerplate", None

    en_words = count_english_words(row.en_text)
    zh_han = count_han(row.zh_text)
    en_han = count_han(row.en_text)
    if en_words < args.min_row_english_words or zh_han < args.min_row_han:
        return "row_too_short", f"english_words={en_words}, chinese_han={zh_han}"
    if en_han > 2:
        return "english_column_language_mismatch", f"english_column_han={en_han}"

    ratio = zh_han / en_words
    if ratio < args.min_length_ratio or ratio > args.max_length_ratio:
        return "length_ratio_outlier", f"chinese_han_per_english_word={ratio:.4f}"
    return None, None


def build_clean_record(rows: list[RawPair], generated_at: str, input_hash: str) -> dict:
    en_text = " ".join(row.en_text for row in rows).strip()
    zh_text = " ".join(row.zh_text for row in rows).strip()
    pair_hash = sha256_text(f"{en_text.casefold()}\x1f{zh_text}")
    return {
        "id": f"translatefx_{pair_hash[:20]}",
        "language_pair": "zh_en",
        "domain": "finance",
        "en_text": en_text,
        "zh_text": zh_text,
        "zh_char_count": count_han(zh_text),
        "translation_method": "human",
        "status": "ready",
        "provenance": {
            "dataset": "TranslateFX",
            "corpus": "HK-FSO",
            "source_file": project_path(rows[0].source_file),
            "source_file_sha256": input_hash,
            "row_number_start": rows[0].row_number,
            "row_number_end": rows[-1].row_number,
            "line_start": rows[0].line_start,
            "line_end": rows[-1].line_end,
            "raw_pair_count": len(rows),
            "source_url": SOURCE_URL,
            "referrer_url": REFERRER_URL,
            "rights_status": "review_required",
            "rights_note": RIGHTS_NOTE,
        },
        "quality": {
            "pair_sha256": pair_hash,
            "english_word_count": count_english_words(en_text),
            "chinese_han_count": count_han(zh_text),
            "merged_pair_count": len(rows),
            "cleaned_at": generated_at,
            "review_status": "pending",
            "review_notes": "",
        },
    }


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_args(args: argparse.Namespace) -> None:
    integer_fields = [
        "min_han",
        "max_han",
        "min_english_words",
        "max_english_words",
        "min_row_han",
        "min_row_english_words",
    ]
    for field in integer_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.min_han > args.max_han:
        raise ValueError("--min-han cannot exceed --max-han")
    if args.min_english_words > args.max_english_words:
        raise ValueError("--min-english-words cannot exceed --max-english-words")
    if not 0 < args.min_length_ratio <= args.max_length_ratio:
        raise ValueError("Length ratio limits are invalid")


def clean(args: argparse.Namespace) -> tuple[dict, dict]:
    validate_args(args)
    inputs = discover_inputs(args)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    input_metadata = [
        {
            "path": project_path(path),
            "format": path.suffix.casefold().lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in inputs
    ]
    cleaned_records: list[dict] = []
    rejected_records: list[dict] = []
    rejection_counts: Counter[str] = Counter()
    rejection_raw_pair_counts: Counter[str] = Counter()
    seen_raw_pairs: set[tuple[str, str]] = set()
    seen_clean_pairs: set[tuple[str, str]] = set()
    total_rows = 0
    valid_source_rows = 0

    def reject(reason: str, rows: list[RawPair], details: str | None = None) -> None:
        rejected_records.append(rejection_record(reason, rows, details=details))
        rejection_counts[reason] += 1
        rejection_raw_pair_counts[reason] += len(rows)

    def flush_buffer(buffer: list[RawPair], reason: str) -> None:
        if buffer:
            en_words = sum(count_english_words(row.en_text) for row in buffer)
            zh_han = sum(count_han(row.zh_text) for row in buffer)
            reject(
                reason,
                buffer,
                f"merged_english_words={en_words}, merged_chinese_han={zh_han}",
            )
            buffer.clear()

    for input_path, metadata in zip(inputs, input_metadata, strict=True):
        buffer: list[RawPair] = []
        for row_number, line_start, line_end, columns in iter_rows(input_path):
            total_rows += 1
            if len(columns) != 2:
                placeholder = RawPair(
                    input_path,
                    row_number,
                    line_start,
                    line_end,
                    normalize_text(columns[0]) if columns else "",
                    normalize_text(" ".join(columns[1:])) if len(columns) > 1 else "",
                )
                flush_buffer(buffer, "incomplete_block_before_rejected_row")
                reject("invalid_column_count", [placeholder], f"columns={len(columns)}")
                continue

            row = RawPair(
                input_path,
                row_number,
                line_start,
                line_end,
                normalize_text(columns[0]),
                normalize_text(columns[1]),
            )
            reason, details = validate_row(row, args)
            raw_key = (row.en_text.casefold(), row.zh_text)
            if reason is None and raw_key in seen_raw_pairs:
                reason = "duplicate_raw_pair"
            if reason is not None:
                flush_buffer(buffer, "incomplete_block_before_rejected_row")
                reject(reason, [row], details)
                continue

            seen_raw_pairs.add(raw_key)
            valid_source_rows += 1
            if buffer and row.line_start != buffer[-1].line_end + 1:
                flush_buffer(buffer, "non_contiguous_incomplete_block")
            buffer.append(row)

            merged_han = sum(count_han(item.zh_text) for item in buffer)
            merged_words = sum(count_english_words(item.en_text) for item in buffer)
            if merged_han > args.max_han or merged_words > args.max_english_words:
                reject(
                    "merged_block_too_long",
                    buffer,
                    f"merged_english_words={merged_words}, merged_chinese_han={merged_han}",
                )
                buffer.clear()
                continue
            if merged_han < args.min_han or merged_words < args.min_english_words:
                continue

            record = build_clean_record(buffer, generated_at, metadata["sha256"])
            clean_key = (record["en_text"].casefold(), record["zh_text"])
            if clean_key in seen_clean_pairs:
                reject("duplicate_merged_pair", buffer)
            else:
                seen_clean_pairs.add(clean_key)
                cleaned_records.append(record)
            buffer.clear()

        flush_buffer(buffer, "incomplete_block_at_end_of_file")

    accepted_raw_pairs = sum(record["provenance"]["raw_pair_count"] for record in cleaned_records)
    rejected_raw_pairs = sum(record["raw_pair_count"] for record in rejected_records)
    if accepted_raw_pairs + rejected_raw_pairs != total_rows:
        raise RuntimeError(
            "Accounting error: accepted and rejected raw-pair counts do not match input rows"
        )
    if len({record["id"] for record in cleaned_records}) != len(cleaned_records):
        raise RuntimeError("Duplicate cleaned IDs detected")

    config = {
        "min_han": args.min_han,
        "max_han": args.max_han,
        "min_english_words": args.min_english_words,
        "max_english_words": args.max_english_words,
        "min_row_han": args.min_row_han,
        "min_row_english_words": args.min_row_english_words,
        "min_length_ratio": args.min_length_ratio,
        "max_length_ratio": args.max_length_ratio,
        "merge_policy": "only_adjacent_valid_pairs_with_no_rejected_row_between_them",
    }
    summary = {
        "input_files": input_metadata,
        "raw_rows": total_rows,
        "valid_source_rows_before_merge": valid_source_rows,
        "accepted_records": len(cleaned_records),
        "accepted_raw_pairs": accepted_raw_pairs,
        "rejected_items": len(rejected_records),
        "rejected_raw_pairs": rejected_raw_pairs,
        "rejection_items_by_reason": dict(sorted(rejection_counts.items())),
        "rejection_raw_pairs_by_reason": dict(sorted(rejection_raw_pair_counts.items())),
        "config": config,
    }
    cleaned_document = {
        "schema_version": 2,
        "dataset": "translatefx",
        "language_pair": "zh_en",
        "domain": "finance",
        "generated_at": generated_at,
        "records": cleaned_records,
        "cleaning_summary": summary,
    }
    rejected_document = {
        "schema_version": 2,
        "dataset": "translatefx",
        "language_pair": "zh_en",
        "domain": "finance",
        "generated_at": generated_at,
        "records": rejected_records,
        "rejection_summary": summary,
    }
    return cleaned_document, rejected_document


def main() -> int:
    args = parse_args()
    try:
        cleaned_document, rejected_document = clean(args)
        summary = cleaned_document["cleaning_summary"]
        if not args.dry_run:
            output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
            rejected_output = (
                args.rejected_output
                if args.rejected_output.is_absolute()
                else PROJECT_ROOT / args.rejected_output
            )
            write_json_atomic(output, cleaned_document)
            write_json_atomic(rejected_output, rejected_document)
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "raw_rows": summary["raw_rows"],
                    "accepted_records": summary["accepted_records"],
                    "accepted_raw_pairs": summary["accepted_raw_pairs"],
                    "rejected_items": summary["rejected_items"],
                    "rejected_raw_pairs": summary["rejected_raw_pairs"],
                    "rejection_raw_pairs_by_reason": summary[
                        "rejection_raw_pairs_by_reason"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError, csv.Error) as error:
        print(f"TranslateFX cleaning failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
