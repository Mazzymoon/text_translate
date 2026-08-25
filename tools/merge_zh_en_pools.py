#!/usr/bin/env python3

"""Merge existing Chinese-English pairs into unfiltered domain pools.

This is intentionally a structure-only stage.  It does not clean, normalize,
filter by domain, calculate hashes, deduplicate, sample, translate, or assign a
translation direction.  Later pipeline stages must perform those operations.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("finance", "education", "technology")
TRANSLATION_DIR = PROJECT_ROOT / "dataset" / "crawled" / "zh_en" / "translations"
UNPC_DIR = PROJECT_ROOT / "dataset" / "external" / "unpc" / "cleaned"
TRANSLATEFX_DIR = PROJECT_ROOT / "dataset" / "external" / "translatefx" / "cleaned"
POOL_DIR = PROJECT_ROOT / "dataset" / "final" / "zh_en" / "pools"
REPORT_DIR = PROJECT_ROOT / "dataset" / "final" / "zh_en" / "reports"

REQUESTED_TRANSLATION_PATTERN = "dataset/translations/zh_en/{domain}.{provider}_mt.json"
ACTUAL_TRANSLATION_PATTERN = "dataset/crawled/zh_en/translations/{domain}.{provider}_mt.json"

SUCCESS_STATUS_VALUES = {
    "accepted",
    "complete",
    "completed",
    "ready",
    "success",
    "succeeded",
    "translated",
}
TRANSLATION_STATUS_FIELDS = ("translation_status", "provider_status", "status")


@dataclass(frozen=True)
class SourceConfig:
    key: str
    adapter: str
    path: Path
    dataset_name: str
    data_origin: str
    requested_path: str | None = None


@dataclass
class SourceStats:
    read_records: int = 0
    converted_records: int = 0
    non_success_status: int = 0
    unrecognized_language_direction: int = 0
    missing_language_text: int = 0
    json_structure_errors: int = 0
    load_error: str | None = None

    def as_dict(self) -> dict:
        return {
            "read_records": self.read_records,
            "converted_records": self.converted_records,
            "non_success_status": self.non_success_status,
            "unrecognized_language_direction": self.unrecognized_language_direction,
            "missing_language_text": self.missing_language_text,
            "json_structure_errors": self.json_structure_errors,
            "load_error": self.load_error,
        }


Adapter = Callable[[dict, SourceConfig, SourceStats], dict | None]


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def source_configs(domain: str) -> list[SourceConfig]:
    configs = [
        SourceConfig(
            key="tencent_mt",
            adapter="machine_translation",
            path=TRANSLATION_DIR / f"{domain}.tencent_mt.json",
            dataset_name="Tencent MT",
            data_origin="machine_translation",
            requested_path=REQUESTED_TRANSLATION_PATTERN.format(
                domain=domain, provider="tencent"
            ),
        ),
        SourceConfig(
            key="baidu_mt",
            adapter="machine_translation",
            path=TRANSLATION_DIR / f"{domain}.baidu_mt.json",
            dataset_name="Baidu MT",
            data_origin="machine_translation",
            requested_path=REQUESTED_TRANSLATION_PATTERN.format(
                domain=domain, provider="baidu"
            ),
        ),
        SourceConfig(
            key="unpc",
            adapter="public_parallel",
            path=UNPC_DIR / f"{domain}_pairs.json",
            dataset_name="UN Parallel Corpus v1.0",
            data_origin="public_parallel",
        ),
    ]
    if domain == "finance":
        configs.append(
            SourceConfig(
                key="translatefx",
                adapter="translatefx_parallel",
                path=TRANSLATEFX_DIR / "finance.json",
                dataset_name="TranslateFX",
                data_origin="public_parallel",
            )
        )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge translated and public parallel records into unfiltered pools."
    )
    parser.add_argument(
        "--domain",
        choices=(*DOMAINS, "all"),
        required=True,
        help="Domain to merge, or all three domains.",
    )
    return parser.parse_args()


def load_records(path: Path, stats: SourceStats) -> list:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        stats.json_structure_errors += 1
        stats.load_error = f"{type(error).__name__}: {error}"
        return []

    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    stats.json_structure_errors += 1
    stats.load_error = "Expected a root array or an object containing a records array"
    return []


def present_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def translation_is_successful(record: dict) -> bool:
    """Check execution status only; review_status is deliberately not used."""

    for field in TRANSLATION_STATUS_FIELDS:
        if field in record and record[field] is not None:
            return str(record[field]).strip().casefold() in SUCCESS_STATUS_VALUES
    # These files are the provider-success outputs.  Absence of an execution
    # status is accepted only when both texts later pass structural checks.
    return True


def base_provenance(record: dict, config: SourceConfig) -> dict:
    provenance = {"source_file": project_relative(config.path)}
    if record.get("id") is not None:
        provenance["original_id"] = record["id"]
    return provenance


def adapt_machine_translation(
    record: dict, config: SourceConfig, stats: SourceStats
) -> dict | None:
    if not translation_is_successful(record):
        stats.non_success_status += 1
        return None

    source_lang = record.get("source_lang")
    target_lang = record.get("target_lang")
    source_text = record.get("source_text")
    target_text = record.get("target_text")
    if (source_lang, target_lang) == ("zh-CN", "en"):
        zh_text, en_text = source_text, target_text
    elif (source_lang, target_lang) == ("en", "zh-CN"):
        en_text, zh_text = source_text, target_text
    else:
        stats.unrecognized_language_direction += 1
        return None

    if not present_text(zh_text) or not present_text(en_text):
        stats.missing_language_text += 1
        return None
    return {
        "zh_text": zh_text,
        "en_text": en_text,
        "data_origin": config.data_origin,
        "dataset_name": config.dataset_name,
        "provenance": base_provenance(record, config),
    }


def adapt_public_parallel(
    record: dict, config: SourceConfig, stats: SourceStats
) -> dict | None:
    zh_text = record.get("zh_text")
    en_text = record.get("en_text")
    if not present_text(zh_text) or not present_text(en_text):
        stats.missing_language_text += 1
        return None

    provenance = base_provenance(record, config)
    for field in ("document_id", "start_line", "end_line"):
        if record.get(field) is not None:
            provenance[field] = record[field]
    return {
        "zh_text": zh_text,
        "en_text": en_text,
        "data_origin": config.data_origin,
        "dataset_name": config.dataset_name,
        "provenance": provenance,
    }


def adapt_translatefx_parallel(
    record: dict, config: SourceConfig, stats: SourceStats
) -> dict | None:
    converted = adapt_public_parallel(record, config, stats)
    if converted is None:
        return None

    original_provenance = record.get("provenance")
    if original_provenance is None:
        return converted
    if not isinstance(original_provenance, dict):
        stats.json_structure_errors += 1
        return converted

    provenance = converted["provenance"]
    if present_text(original_provenance.get("source_file")):
        provenance["source_file"] = original_provenance["source_file"]
    if present_text(original_provenance.get("source_url")):
        provenance["source_url"] = original_provenance["source_url"]
    if original_provenance.get("line_start") is not None:
        provenance["start_line"] = original_provenance["line_start"]
    if original_provenance.get("line_end") is not None:
        provenance["end_line"] = original_provenance["line_end"]
    return converted


ADAPTERS: dict[str, Adapter] = {
    "machine_translation": adapt_machine_translation,
    "public_parallel": adapt_public_parallel,
    "translatefx_parallel": adapt_translatefx_parallel,
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


def merge_domain(domain: str, generated_at: str) -> tuple[dict, dict]:
    merged_records: list[dict] = []
    source_reports: list[dict] = []
    dataset_counts: Counter[str] = Counter()

    for config in source_configs(domain):
        stats = SourceStats()
        exists = config.path.is_file()
        records = load_records(config.path, stats) if exists else []
        if not exists:
            stats.load_error = "Configured input file does not exist"

        adapter = ADAPTERS[config.adapter]
        for record in records:
            stats.read_records += 1
            if not isinstance(record, dict):
                stats.json_structure_errors += 1
                continue
            converted = adapter(record, config, stats)
            if converted is None:
                continue
            stats.converted_records += 1
            dataset_counts[converted["dataset_name"]] += 1
            merged_records.append(converted)

        source_reports.append(
            {
                "source_key": config.key,
                "adapter": config.adapter,
                "dataset_name": config.dataset_name,
                "configured_input_file": project_relative(config.path),
                "requested_input_file": config.requested_path,
                "path_adjusted": bool(
                    config.requested_path
                    and config.requested_path != project_relative(config.path)
                ),
                "exists": exists,
                **stats.as_dict(),
            }
        )

    for index, record in enumerate(merged_records, start=1):
        record["pool_record_id"] = f"{domain}_pool_{index:06d}"

    # Put the generated identifier first without modifying any source text.
    ordered_records = [
        {
            "pool_record_id": record["pool_record_id"],
            "zh_text": record["zh_text"],
            "en_text": record["en_text"],
            "domain": domain,
            "data_origin": record["data_origin"],
            "dataset_name": record["dataset_name"],
            "provenance": record["provenance"],
        }
        for record in merged_records
    ]

    pool = {
        "schema_version": 1,
        "domain": domain,
        "stage": "merged_unfiltered_pool",
        "generated_at": generated_at,
        "records": ordered_records,
    }
    report = {
        "schema_version": 1,
        "domain": domain,
        "stage": "merged_unfiltered_pool",
        "generated_at": generated_at,
        "scope_guarantees": {
            "quality_cleaning_performed": False,
            "domain_filtering_performed": False,
            "deduplication_performed": False,
            "pair_sha256_generated": False,
            "quantity_selection_performed": False,
            "translation_direction_assigned": False,
        },
        "path_adjustments": [
            {
                "requested_pattern": REQUESTED_TRANSLATION_PATTERN,
                "actual_pattern": ACTUAL_TRANSLATION_PATTERN,
                "reason": "The project stores provider outputs under dataset/crawled/zh_en/translations.",
            }
        ],
        "status_interpretation": (
            "review_status is a human-review field and is not treated as translation execution status; "
            "provider-success files without an explicit execution status are accepted when both texts exist."
        ),
        "configured_sources": source_reports,
        "summary": {
            "final_pool_records": len(ordered_records),
            "dataset_name_counts": dict(dataset_counts),
            "read_records": sum(item["read_records"] for item in source_reports),
            "converted_records": sum(item["converted_records"] for item in source_reports),
            "non_success_status": sum(item["non_success_status"] for item in source_reports),
            "unrecognized_language_direction": sum(
                item["unrecognized_language_direction"] for item in source_reports
            ),
            "missing_language_text": sum(
                item["missing_language_text"] for item in source_reports
            ),
            "json_structure_errors": sum(
                item["json_structure_errors"] for item in source_reports
            ),
        },
    }
    return pool, report


def main() -> int:
    args = parse_args()
    domains = DOMAINS if args.domain == "all" else (args.domain,)
    generated_at = utc_now()
    missing_inputs = False
    for domain in domains:
        pool, report = merge_domain(domain, generated_at)
        write_json_atomic(POOL_DIR / f"{domain}_pool.json", pool)
        write_json_atomic(REPORT_DIR / f"{domain}_pool_merge_report.json", report)
        if any(not item["exists"] for item in report["configured_sources"]):
            missing_inputs = True
        print(f"{domain}: {report['summary']['final_pool_records']:,} records")
        for dataset_name, count in report["summary"]["dataset_name_counts"].items():
            print(f"  {dataset_name}: {count:,}")
    return 1 if missing_inputs else 0


if __name__ == "__main__":
    raise SystemExit(main())
