#!/usr/bin/env python3
"""Clean and merge Qwen/NLLB Chinese-Thai translations, then fill from ALT.

The translation jobs save atomic JSONL chunks while running.  This script reads
the latest event for every source ID from those chunks (falling back to
accepted/rejected JSONL only when no chunks exist), performs lightweight text
and language checks, selects one translation per Chinese source record, and
deduplicates normalized bilingual pairs.  If fewer than the requested number
remain, aligned Thai/Chinese rows from ALT are added until the target is met.

No input file is modified.  The default target is exactly 20,000 records.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSLATIONS_DIR = PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "translations"
DEFAULT_ALT_DIR = PROJECT_ROOT / "dataset" / "external" / "th_ALT" / "th-zh.txt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "cleaned" / "zh_th_merged.jsonl"
)
DEFAULT_REJECTED = (
    PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "rejected" / "zh_th_merge_rejected.jsonl"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "reports" / "zh_th_merge_report.json"
)

SOURCE_CONFIGS = (
    {
        "directory": "qwen3_4b",
        "dataset_name": "Qwen3-4B-Instruct-2507",
        "provider": "qwen3_4b",
        "preference": 0,
    },
    {
        "directory": "qwen3_4b_4bit",
        "dataset_name": "Qwen3-4B-Instruct-2507 4-bit NF4",
        "provider": "qwen3_4b_4bit",
        "preference": 1,
    },
    {
        "directory": "nllb_600m",
        "dataset_name": "NLLB-200-distilled-600M",
        "provider": "nllb_600m",
        "preference": 2,
    },
)

ALT_THAI_FILENAME = "ALT.th-zh.th"
ALT_CHINESE_FILENAME = "ALT.th-zh.zh"
ALT_DATASET_NAME = "Asian Language Treebank (ALT) v20191206"
ALT_LICENSE = "CC BY 4.0"
ALT_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DOMAINS = {"education", "technology", "finance"}

HAN_RE = re.compile(
    "[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    "\U00020000-\U0002FA1F\U00030000-\U000323AF]"
)
THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]")
SPACE_RE = re.compile(r"[\s\u00A0\u202F\u3000]+")
MOJIBAKE_RE = re.compile(r"(?:�|鏁|鈥|锟|à¸|à¹|Ã.|Â.)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cleaned Qwen/NLLB zh-th translations and fill to target from ALT."
    )
    parser.add_argument("--translations-dir", type=Path, default=DEFAULT_TRANSLATIONS_DIR)
    parser.add_argument("--alt-dir", type=Path, default=DEFAULT_ALT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-records", type=int, default=20_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate everything, but do not create or modify output files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = CONTROL_RE.sub("", text)
    text = ZERO_WIDTH_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def count_han(text: str) -> int:
    return len(HAN_RE.findall(text))


def count_thai(text: str) -> int:
    return len(THAI_RE.findall(text))


def thai_letter_ratio(text: str) -> float:
    thai_letters = sum("\u0E00" <= char <= "\u0E7F" and char.isalpha() for char in text)
    all_letters = sum(char.isalpha() for char in text)
    return thai_letters / max(all_letters, 1)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_sha256(zh_text: str, th_text: str) -> str:
    return sha256_text(zh_text + "\x1f" + th_text)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield line_number, value


def chunk_number(path: Path) -> int:
    match = re.fullmatch(r"chunk_(\d+)\.jsonl", path.name)
    if not match:
        raise ValueError(f"Invalid chunk filename: {path}")
    return int(match.group(1))


def event_paths(source_dir: Path) -> tuple[list[Path], str]:
    chunks_dir = source_dir / "chunks"
    chunks = sorted(chunks_dir.glob("chunk_*.jsonl"), key=chunk_number) if chunks_dir.is_dir() else []
    consolidated = [
        path
        for path in (source_dir / "accepted.jsonl", source_dir / "rejected.jsonl")
        if path.is_file()
    ]
    # A normally completed translator rewrites accepted/rejected after its last
    # chunk. Prefer those final, compact files when they are at least as new as
    # the newest chunk. If translation is still running, chunks are newer and
    # therefore contain the more complete crash-safe state.
    if consolidated and (
        not chunks
        or min(path.stat().st_mtime_ns for path in consolidated)
        >= chunks[-1].stat().st_mtime_ns
    ):
        return consolidated, "final_consolidated_jsonl"
    if chunks:
        return chunks, "progress_chunks_fallback"
    return consolidated, "consolidated_jsonl_without_chunks"


def load_latest_events(
    source_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths, mode = event_paths(source_dir)
    if not paths:
        raise FileNotFoundError(f"No translation chunks or accepted/rejected JSONL found: {source_dir}")
    latest: dict[str, dict[str, Any]] = {}
    events = 0
    statuses: Counter[str] = Counter()
    for path in paths:
        for line_number, event in read_jsonl(path):
            record_id = str(event.get("source_record_id", "")).strip()
            if not record_id:
                raise ValueError(f"Missing source_record_id at {path}:{line_number}")
            status = str(event.get("status", "")).strip()
            if status not in {"success", "failed"}:
                raise ValueError(f"Unknown status {status!r} at {path}:{line_number}")
            latest[record_id] = event
            statuses[status] += 1
            events += 1
    latest_statuses = Counter(str(event["status"]) for event in latest.values())
    return latest, {
        "directory": relative_path(source_dir),
        "read_mode": mode,
        "files_read": [relative_path(path) for path in paths],
        "events_read": events,
        "event_statuses": dict(statuses),
        "latest_unique_source_ids": len(latest),
        "latest_statuses": dict(latest_statuses),
    }


def validate_pair(zh_text: str, th_text: str) -> tuple[list[str], list[str], dict[str, Any]]:
    hard_reasons: list[str] = []
    flags: list[str] = []
    zh_han = count_han(zh_text)
    zh_thai = count_thai(zh_text)
    th_han = count_han(th_text)
    th_thai = count_thai(th_text)
    th_ratio = thai_letter_ratio(th_text)
    length_ratio = len(th_text) / max(len(zh_text), 1)

    if not zh_text:
        hard_reasons.append("empty_chinese")
    if not th_text:
        hard_reasons.append("empty_thai")
    if zh_text and zh_han == 0:
        hard_reasons.append("chinese_side_has_no_han")
    if th_text and th_thai == 0:
        hard_reasons.append("thai_side_has_no_thai")
    if zh_text and th_text and zh_text == th_text:
        hard_reasons.append("identical_sides")
    if MOJIBAKE_RE.search(zh_text) or MOJIBAKE_RE.search(th_text):
        hard_reasons.append("mojibake")
    if th_han:
        hard_reasons.append("thai_side_contains_han")
    if th_text and th_ratio < 0.50:
        hard_reasons.append("thai_side_not_thai_dominant")
    if zh_text and zh_thai > max(5, zh_han):
        hard_reasons.append("chinese_side_looks_thai")
    if zh_text and th_text and (length_ratio < 0.10 or length_ratio > 8.0):
        hard_reasons.append("extreme_length_ratio")
    elif zh_text and th_text and (length_ratio < 0.25 or length_ratio > 4.0):
        flags.append("unusual_length_ratio")
    if any(char.isdigit() for char in zh_text) and not any(char.isdigit() for char in th_text):
        flags.append("source_has_digits_target_has_none")

    return list(dict.fromkeys(hard_reasons)), flags, {
        "zh_han_count": zh_han,
        "th_thai_character_count": th_thai,
        "th_han_count": th_han,
        "thai_letter_ratio": round(th_ratio, 6),
        "target_to_source_codepoint_ratio": round(length_ratio, 6),
    }


def machine_candidate(
    event: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    record_id = str(event.get("source_record_id", "")).strip()
    if event.get("status") != "success":
        return None, {
            "source_type": "machine_translation",
            "provider": config["provider"],
            "source_record_id": record_id,
            "reason": "latest_translation_status_failed",
            "error_type": event.get("error_type"),
            "error_message": event.get("error_message"),
        }
    zh_text = normalize_text(event.get("source_text"))
    th_text = normalize_text(event.get("target_text"))
    hard_reasons, flags, metrics = validate_pair(zh_text, th_text)
    if hard_reasons:
        return None, {
            "source_type": "machine_translation",
            "provider": config["provider"],
            "source_record_id": record_id,
            "reasons": hard_reasons,
            "source_text": event.get("source_text"),
            "target_text": event.get("target_text"),
        }
    return {
        "source_record_id": record_id,
        "zh_text": zh_text,
        "th_text": th_text,
        "domain": event.get("domain") if event.get("domain") in DOMAINS else None,
        "dataset_name": config["dataset_name"],
        "data_origin": "machine_translation",
        "translation_method": "llm_mt",
        "provider": config["provider"],
        "provider_preference": config["preference"],
        "model_name": event.get("model_name"),
        "model_quantization": event.get("model_quantization"),
        "model_weight_format": event.get("model_weight_format"),
        "generated_at": event.get("generated_at"),
        "quality_flags": flags,
        "quality_metrics": metrics,
        "original_quality_flags": event.get("quality_flags") or [],
        "provenance": event.get("provenance") or {},
        "original_translation_id": event.get("id"),
    }, None


def candidate_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["quality_metrics"]
    return (
        len(candidate["quality_flags"]),
        len(candidate.get("original_quality_flags") or []),
        metrics["th_han_count"],
        -metrics["thai_letter_ratio"],
        candidate["provider_preference"],
        candidate["provider"],
    )


def merge_machine_sources(
    translations_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    source_reports: dict[str, Any] = {}
    provider_valid: Counter[str] = Counter()
    provider_rejected: Counter[str] = Counter()

    for config in SOURCE_CONFIGS:
        source_dir = translations_dir / config["directory"]
        latest, source_report = load_latest_events(source_dir)
        source_reports[config["provider"]] = source_report
        for event in latest.values():
            candidate, rejection = machine_candidate(event, config)
            if candidate is not None:
                candidates_by_source[candidate["source_record_id"]].append(candidate)
                provider_valid[config["provider"]] += 1
            else:
                assert rejection is not None
                rejected.append(rejection)
                provider_rejected[config["provider"]] += 1

    selected: list[dict[str, Any]] = []
    overlapping_source_ids = 0
    alternatives_discarded = 0
    for record_id, candidates in candidates_by_source.items():
        if len(candidates) > 1:
            overlapping_source_ids += 1
            alternatives_discarded += len(candidates) - 1
        candidates.sort(key=candidate_rank)
        chosen = candidates[0]
        chosen["translation_sources"] = [
            {
                "provider": item["provider"],
                "dataset_name": item["dataset_name"],
                "original_translation_id": item["original_translation_id"],
                "selected": item is chosen,
            }
            for item in candidates
        ]
        selected.append(chosen)

    selected.sort(key=lambda item: (item["domain"] or "", item["source_record_id"]))
    return selected, rejected, {
        "sources": source_reports,
        "valid_candidates_by_provider": dict(provider_valid),
        "rejected_latest_records_by_provider": dict(provider_rejected),
        "unique_machine_source_ids_with_valid_candidate": len(candidates_by_source),
        "overlapping_machine_source_id_groups": overlapping_source_ids,
        "alternative_machine_candidates_discarded": alternatives_discarded,
    }


def alt_candidates(alt_dir: Path) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    thai_path = alt_dir / ALT_THAI_FILENAME
    chinese_path = alt_dir / ALT_CHINESE_FILENAME
    if not thai_path.is_file() or not chinese_path.is_file():
        raise FileNotFoundError(
            f"ALT requires both {thai_path} and {chinese_path}; do not use the one-sided zh.txt.gz."
        )
    with thai_path.open("r", encoding="utf-8-sig") as thai_handle, chinese_path.open(
        "r", encoding="utf-8-sig"
    ) as chinese_handle:
        for line_number, (thai_line, chinese_line) in enumerate(
            itertools.zip_longest(thai_handle, chinese_handle), start=1
        ):
            if thai_line is None or chinese_line is None:
                raise ValueError(f"ALT Thai/Chinese line count mismatch at line {line_number}")
            zh_text = normalize_text(chinese_line)
            th_text = normalize_text(thai_line)
            hard_reasons, flags, metrics = validate_pair(zh_text, th_text)
            if hard_reasons:
                yield None, {
                    "source_type": "public_parallel",
                    "dataset_name": ALT_DATASET_NAME,
                    "source_line": line_number,
                    "reasons": hard_reasons,
                    "zh_text": chinese_line.rstrip("\r\n"),
                    "th_text": thai_line.rstrip("\r\n"),
                }
                continue
            yield {
                "source_record_id": f"alt_th_zh_line_{line_number}",
                "zh_text": zh_text,
                "th_text": th_text,
                "domain": None,
                "dataset_name": ALT_DATASET_NAME,
                "data_origin": "public_parallel",
                "translation_method": "human",
                "provider": "alt",
                "model_name": None,
                "model_quantization": None,
                "model_weight_format": None,
                "generated_at": None,
                "quality_flags": flags,
                "quality_metrics": metrics,
                "translation_sources": [
                    {
                        "provider": "alt",
                        "dataset_name": ALT_DATASET_NAME,
                        "source_line": line_number,
                        "selected": True,
                    }
                ],
                "provenance": {
                    "source_files": [
                        relative_path(chinese_path),
                        relative_path(thai_path),
                    ],
                    "source_line": line_number,
                    "license": ALT_LICENSE,
                    "license_url": ALT_LICENSE_URL,
                },
                "original_translation_id": None,
            }, None


def final_record(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    zh_text = candidate["zh_text"]
    th_text = candidate["th_text"]
    pair_hash = pair_sha256(zh_text, th_text)
    return {
        "id": f"zh_th_merged_{index:06d}",
        "source_lang": "zh-CN",
        "target_lang": "th",
        "source_text": zh_text,
        "target_text": th_text,
        "zh_text": zh_text,
        "th_text": th_text,
        "zh_char_count": count_han(zh_text),
        "domain": candidate["domain"],
        "translation_method": candidate["translation_method"],
        "dataset_name": candidate["dataset_name"],
        "data_origin": candidate["data_origin"],
        "model_name": candidate.get("model_name"),
        "model_quantization": candidate.get("model_quantization"),
        "model_weight_format": candidate.get("model_weight_format"),
        "pair_sha256": pair_hash,
        "quality_flags": candidate["quality_flags"],
        "quality_metrics": candidate["quality_metrics"],
        "translation_sources": candidate["translation_sources"],
        "provenance": candidate["provenance"],
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    translations_dir = resolve_path(args.translations_dir)
    alt_dir = resolve_path(args.alt_dir)
    if args.target_records <= 0:
        raise ValueError("--target-records must be positive")

    machine, rejected, machine_report = merge_machine_sources(translations_dir)
    selected_candidates: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    pair_duplicates = 0
    machine_selected_by_provider: Counter[str] = Counter()

    for candidate in machine:
        pair_hash = pair_sha256(candidate["zh_text"], candidate["th_text"])
        if pair_hash in seen_pairs:
            pair_duplicates += 1
            rejected.append(
                {
                    "source_type": "machine_translation",
                    "provider": candidate["provider"],
                    "source_record_id": candidate["source_record_id"],
                    "reason": "duplicate_normalized_pair",
                    "pair_sha256": pair_hash,
                }
            )
            continue
        seen_pairs.add(pair_hash)
        selected_candidates.append(candidate)
        machine_selected_by_provider[candidate["provider"]] += 1

    if len(selected_candidates) > args.target_records:
        selected_candidates = selected_candidates[: args.target_records]

    machine_records_kept = len(selected_candidates)
    alt_lines_read = 0
    alt_valid = 0
    alt_duplicates = 0
    alt_added = 0
    alt_rejected = 0

    # Always read the complete ALT pair so line-count alignment is verified, even
    # after enough rows have been selected.
    for candidate, rejection in alt_candidates(alt_dir):
        alt_lines_read += 1
        if candidate is None:
            assert rejection is not None
            alt_rejected += 1
            rejected.append(rejection)
            continue
        alt_valid += 1
        pair_hash = pair_sha256(candidate["zh_text"], candidate["th_text"])
        if pair_hash in seen_pairs:
            alt_duplicates += 1
            continue
        if len(selected_candidates) < args.target_records:
            seen_pairs.add(pair_hash)
            selected_candidates.append(candidate)
            alt_added += 1

    if len(selected_candidates) < args.target_records:
        raise RuntimeError(
            f"Only {len(selected_candidates):,} unique valid pairs are available; "
            f"target is {args.target_records:,}. No records were duplicated to fill the gap."
        )

    records = [final_record(candidate, index) for index, candidate in enumerate(selected_candidates, 1)]
    hashes = [record["pair_sha256"] for record in records]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Internal verification failed: final pair_sha256 values are not unique")
    if len(records) != args.target_records:
        raise RuntimeError("Internal verification failed: final record count differs from target")

    report = {
        "schema_version": 1,
        "stage": "zh_th_cleaned_merge",
        "generated_at": utc_now(),
        "target_records": args.target_records,
        "dry_run": args.dry_run,
        "input": {
            "translations_directory": relative_path(translations_dir),
            "alt_directory": relative_path(alt_dir),
            "machine_sources": [config["directory"] for config in SOURCE_CONFIGS],
            "alt_files": [ALT_CHINESE_FILENAME, ALT_THAI_FILENAME],
        },
        "machine_translation": {
            **machine_report,
            "selected_after_pair_dedup": machine_records_kept,
            "selected_by_provider": dict(machine_selected_by_provider),
            "normalized_pair_duplicates_removed": pair_duplicates,
        },
        "alt": {
            "dataset_name": ALT_DATASET_NAME,
            "license": ALT_LICENSE,
            "lines_read_per_side": alt_lines_read,
            "valid_pairs": alt_valid,
            "rejected_pairs": alt_rejected,
            "duplicates_against_selected_pairs": alt_duplicates,
            "pairs_added_to_reach_target": alt_added,
        },
        "final": {
            "records": len(records),
            "machine_translation_records": sum(
                record["data_origin"] == "machine_translation" for record in records
            ),
            "public_parallel_records": sum(record["data_origin"] == "public_parallel" for record in records),
            "unique_pair_sha256": len(set(hashes)),
            "empty_source_text": sum(not record["source_text"] for record in records),
            "empty_target_text": sum(not record["target_text"] for record in records),
            "records_by_dataset_name": dict(Counter(record["dataset_name"] for record in records)),
            "records_by_data_origin": dict(Counter(record["data_origin"] for record in records)),
            "records_by_domain": dict(Counter(str(record["domain"]) for record in records)),
        },
        "rejected_records_written": len(rejected),
        "rules": {
            "unicode_normalization": "NFC",
            "exact_deduplication": 'SHA256(zh_text + "\\x1f" + th_text)',
            "minimum_100_characters_required": False,
            "machine_source_id_policy": "one selected translation per original Chinese source ID",
            "alt_manual_decompression_required": False,
        },
    }
    return records, rejected, report


def print_summary(report: dict[str, Any]) -> None:
    machine = report["machine_translation"]
    alt = report["alt"]
    final = report["final"]
    print("Machine translation inputs (latest state)")
    for provider, source in machine["sources"].items():
        statuses = source["latest_statuses"]
        print(
            f"  {provider}: unique={source['latest_unique_source_ids']:,} "
            f"success={statuses.get('success', 0):,} failed={statuses.get('failed', 0):,}"
        )
    print(f"valid unique machine records kept: {final['machine_translation_records']:,}")
    print(f"ALT lines checked per side: {alt['lines_read_per_side']:,}")
    print(f"ALT records added: {alt['pairs_added_to_reach_target']:,}")
    print(f"final records: {final['records']:,}")
    print(f"unique pair SHA256: {final['unique_pair_sha256']:,}")
    print(f"rejected records: {report['rejected_records_written']:,}")
    if report["dry_run"]:
        print("dry-run: no output files were written")


def main() -> int:
    args = parse_args()
    output = resolve_path(args.output)
    rejected_path = resolve_path(args.rejected)
    report_path = resolve_path(args.report)
    try:
        records, rejected, report = build(args)
        print_summary(report)
        if not args.dry_run:
            atomic_write_jsonl(output, records)
            atomic_write_jsonl(rejected_path, rejected)
            atomic_write_json(report_path, report)
            print(f"output: {output}")
            print(f"report: {report_path}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Chinese-Thai merge failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
