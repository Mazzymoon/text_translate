#!/usr/bin/env python3
"""Build the final 30,000-row Chinese-English dataset from prepared pools.

This is intentionally a final selection and direction-assignment stage.  It
does not read raw corpora, crawler records, translation-provider records, or
perform domain scoring / source-text cleaning again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_DIR = PROJECT_ROOT / "dataset" / "final" / "zh_en"
DOMAINS = ("education", "technology", "finance")
SEED = 20260812
RECORDS_PER_DOMAIN = 10_000
RECORDS_PER_DIRECTION = 5_000
CSV_COLUMNS = (
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "zh_char_count",
    "domain",
    "translation_method",
)

# Match the existing Node validator's ``\p{Script=Han}`` definition rather than
# only the BMP ranges.  Extension-B-and-later Han characters do occur in the
# source corpora and must be included in zh_char_count.
HAN_RE = re.compile(r"[^\W\d_]", re.UNICODE)
LATIN_RE = re.compile(r"[A-Za-z]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
WHITESPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_CHARACTERS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select, split, and export the final 30,000-row zh-en dataset."
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=DEFAULT_FINAL_DIR,
        help="Directory containing pools/ (and optionally prepared intermediate results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Fixed deterministic seed (default: {SEED}).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != SEED:
        parser.error(f"This deliverable requires the fixed seed {SEED}; --seed cannot be changed.")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_han(text: str) -> int:
    return sum(
        1
        for character in text
        if "CJK UNIFIED IDEOGRAPH" in unicodedata.name(character, "")
        or "CJK COMPATIBILITY IDEOGRAPH" in unicodedata.name(character, "")
        or character == "\u3007"
    )


def count_latin(text: str) -> int:
    return len(LATIN_RE.findall(text))


def count_english_words(text: str) -> int:
    return len(ENGLISH_WORD_RE.findall(text))


def canonical_for_hash(value: str) -> str:
    """Normalize only for exact-pair identity; never rewrite exported text."""

    text = unicodedata.normalize("NFC", value)
    normalized: list[str] = []
    for character in text:
        if character in ZERO_WIDTH_CHARACTERS:
            continue
        if character in {"\u00a0", "\u3000", "\r", "\n", "\t", "\f", "\v"}:
            normalized.append(" ")
        elif unicodedata.category(character) in {"Cc", "Cf"}:
            continue
        elif unicodedata.category(character) == "Zs":
            normalized.append(" ")
        else:
            normalized.append(character)
    return WHITESPACE_RE.sub(" ", "".join(normalized)).strip()


def calculate_pair_sha256(zh_text: str, en_text: str) -> str:
    canonical_zh = canonical_for_hash(zh_text)
    canonical_en = canonical_for_hash(en_text)
    value = f"{canonical_zh}\x1f{canonical_en.casefold()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_shuffle(records: list[dict[str, Any]], seed: int, purpose: str) -> list[dict[str, Any]]:
    """Return a reproducible shuffle without relying on input-file ordering."""

    copied = list(records)
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{purpose}".encode("utf-8")).digest()[:8], "big"
    )
    random.Random(stable_seed).shuffle(copied)
    return copied


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON input {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise RuntimeError(f"Unrecognized JSON structure in {path}: expected object with records[]")
    return value


def discover_inputs(final_dir: Path) -> tuple[str, dict[str, Path], dict[Path, str]]:
    """Prefer the already-prepared global pool only when all views exist.

    The current project has no intermediate output yet, so the intended fallback
    is the three merged pools.  This function refuses partial intermediate state
    rather than silently mixing stages.
    """

    intermediate_dir = final_dir / "intermediate"
    unique_path = intermediate_dir / "zh_en_unique_pool.json"
    eligible_paths = {
        domain: intermediate_dir / "eligible" / f"{domain}_eligible.json"
        for domain in DOMAINS
    }
    if unique_path.is_file() and all(path.is_file() for path in eligible_paths.values()):
        inputs = {"unique": unique_path, **eligible_paths}
        return "prepared_intermediate", inputs, {path: file_sha256(path) for path in inputs.values()}

    pool_paths = {domain: final_dir / "pools" / f"{domain}_pool.json" for domain in DOMAINS}
    missing = [path for path in pool_paths.values() if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"No complete prepared intermediate input and missing pool inputs: {missing_text}")
    return "merged_pools_fallback", pool_paths, {
        path: file_sha256(path) for path in pool_paths.values()
    }


def is_obviously_reversed(zh_text: str, en_text: str) -> str | None:
    """Detect only clear field reversal; this is not a fresh language-quality filter."""

    zh_han = count_han(zh_text)
    zh_words = count_english_words(zh_text)
    en_han = count_han(en_text)
    en_words = count_english_words(en_text)
    if zh_words >= 20 and zh_han < 10:
        return "zh_text_appears_to_be_english"
    if en_han >= 20 and en_words < 5:
        return "en_text_appears_to_be_chinese"
    return None


def translation_method(record: dict[str, Any]) -> str:
    existing = record.get("translation_method")
    if existing in {"human", "google_mt", "llm_mt"}:
        return existing
    data_origin = str(record.get("data_origin", "")).casefold()
    dataset_name = str(record.get("dataset_name", "")).casefold()
    if "machine_translation" in data_origin or "tencent" in dataset_name or "baidu" in dataset_name:
        return "llm_mt"
    return "human"


def source_reference(record: dict[str, Any], source_domain: str) -> dict[str, Any]:
    source: dict[str, Any] = {
        "source_domain": source_domain,
        "dataset_name": record.get("dataset_name"),
        "data_origin": record.get("data_origin"),
        "pool_record_id": record.get("pool_record_id"),
        "id": record.get("id"),
    }
    if isinstance(record.get("provenance"), dict):
        source["provenance"] = deepcopy(record["provenance"])
    return source


def adapt_pool_record(record: Any, domain: str, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, dict):
        return None, "invalid_record_structure"
    zh_text = record.get("zh_text")
    en_text = record.get("en_text")
    if not isinstance(zh_text, str) or not zh_text.strip():
        return None, "missing_zh_text"
    if not isinstance(en_text, str) or not en_text.strip():
        return None, "missing_en_text"
    reversed_reason = is_obviously_reversed(zh_text, en_text)
    if reversed_reason:
        return None, reversed_reason

    supplied_hash = record.get("pair_sha256")
    if supplied_hash is not None and (not isinstance(supplied_hash, str) or not supplied_hash.strip()):
        return None, "invalid_pair_sha256"
    digest = supplied_hash.strip() if isinstance(supplied_hash, str) else calculate_pair_sha256(zh_text, en_text)
    return {
        "pair_sha256": digest,
        "zh_text": zh_text,
        "en_text": en_text,
        "zh_char_count": count_han(zh_text),
        "domain": domain,
        "dataset_name": record.get("dataset_name"),
        "data_origin": record.get("data_origin"),
        "translation_method": translation_method(record),
        "id": record.get("id") or record.get("pool_record_id") or f"{domain}_input_{index:06d}",
        "pool_record_id": record.get("pool_record_id"),
        "provenance": deepcopy(record.get("provenance")) if isinstance(record.get("provenance"), dict) else None,
        "source_references": [source_reference(record, domain)],
    }, None


def adapt_intermediate_inputs(paths: dict[str, Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    unique = load_json(paths["unique"])
    records_by_hash: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(unique["records"], start=1):
        if not isinstance(record, dict):
            raise RuntimeError(f"Intermediate unique pool has a non-object record at index {index}")
        digest = record.get("pair_sha256")
        if not isinstance(digest, str) or not digest:
            raise RuntimeError(f"Intermediate unique pool record {index} lacks pair_sha256")
        records_by_hash[digest] = record

    candidates: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    invalid = Counter()
    input_counts: dict[str, int] = {}
    for domain in DOMAINS:
        view = load_json(paths[domain])
        input_counts[domain] = len(view["records"])
        seen: set[str] = set()
        for index, item in enumerate(view["records"], start=1):
            if not isinstance(item, dict) or not isinstance(item.get("pair_sha256"), str):
                invalid["invalid_eligible_view_record"] += 1
                continue
            digest = item["pair_sha256"]
            if digest in seen:
                invalid["duplicate_hash_in_eligible_view"] += 1
                continue
            seen.add(digest)
            source = records_by_hash.get(digest)
            if source is None:
                invalid["eligible_hash_missing_from_unique_pool"] += 1
                continue
            adapted, reason = adapt_pool_record(source, domain, index)
            if reason:
                invalid[reason] += 1
                continue
            adapted["pair_sha256"] = digest  # Preserve the processed-stage hash.
            adapted["source_references"] = deepcopy(source.get("sources", [])) or adapted["source_references"]
            candidates[domain].append(adapted)
    return candidates, {"input_record_counts": input_counts, "invalid_reasons": dict(invalid)}


def adapt_merged_pools(paths: dict[str, Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    invalid_by_domain: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    input_counts: dict[str, int] = {}
    for domain in DOMAINS:
        pool = load_json(paths[domain])
        if pool.get("domain") not in {None, domain}:
            raise RuntimeError(
                f"Pool domain mismatch in {paths[domain]}: expected {domain!r}, got {pool.get('domain')!r}"
            )
        input_counts[domain] = len(pool["records"])
        for index, record in enumerate(pool["records"], start=1):
            adapted, reason = adapt_pool_record(record, domain, index)
            if reason:
                invalid_by_domain[domain][reason] += 1
            else:
                candidates[domain].append(adapted)
    return candidates, {
        "input_record_counts": input_counts,
        "invalid_reasons": {domain: dict(counter) for domain, counter in invalid_by_domain.items()},
    }


def deduplicate_within_domain(
    candidates: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    deduplicated: dict[str, list[dict[str, Any]]] = {}
    statistics: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in candidates[domain]:
            grouped[record["pair_sha256"]].append(record)
        winners: list[dict[str, Any]] = []
        duplicate_groups = 0
        extra_records = 0
        for digest in sorted(grouped):
            group = grouped[digest]
            winner = deepcopy(group[0])
            if len(group) > 1:
                duplicate_groups += 1
                extra_records += len(group) - 1
                references = winner["source_references"]
                seen_references = {
                    json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    for item in references
                }
                for duplicate in group[1:]:
                    for reference in duplicate["source_references"]:
                        key = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        if key not in seen_references:
                            seen_references.add(key)
                            references.append(reference)
            winners.append(winner)
        deduplicated[domain] = winners
        statistics[domain] = {
            "valid_before_within_domain_dedup": len(candidates[domain]),
            "available_after_within_domain_dedup": len(winners),
            "duplicate_groups": duplicate_groups,
            "extra_records_removed": extra_records,
        }
    return deduplicated, statistics


def select_globally_unique(
    candidates: dict[str, list[dict[str, Any]]], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select exactly 10,000 stable-random records per domain with no overlap.

    Domains are handled in ascending candidate-count order, so a smaller pool
    reserves its stochastic selection before larger pools consume shared pairs.
    No domain scoring or source-quality ranking is involved.
    """

    input_domains_by_hash: dict[str, set[str]] = defaultdict(set)
    for domain in DOMAINS:
        for record in candidates[domain]:
            input_domains_by_hash[record["pair_sha256"]].add(domain)

    domain_order = tuple(sorted(DOMAINS, key=lambda name: (len(candidates[name]), name)))
    selected: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    claimed_hashes: set[str] = set()
    selection_skips: dict[str, int] = {domain: 0 for domain in DOMAINS}
    for domain in domain_order:
        shuffled = stable_shuffle(candidates[domain], seed, f"selection:{domain}")
        for record in shuffled:
            if record["pair_sha256"] in claimed_hashes:
                selection_skips[domain] += 1
                continue
            selected[domain].append(record)
            claimed_hashes.add(record["pair_sha256"])
            if len(selected[domain]) == RECORDS_PER_DOMAIN:
                break
        if len(selected[domain]) < RECORDS_PER_DOMAIN:
            raise RuntimeError(
                f"{domain} has {len(selected[domain])} globally unique usable records; "
                f"needs {RECORDS_PER_DOMAIN}, short by {RECORDS_PER_DOMAIN - len(selected[domain])}."
            )

    cross_domain_groups = sum(1 for domains in input_domains_by_hash.values() if len(domains) > 1)
    return selected, {
        "domain_selection_order": list(domain_order),
        "cross_domain_duplicate_groups_in_input": cross_domain_groups,
        "records_skipped_due_to_prior_global_claim": selection_skips,
        "global_selected_unique_pairs": len(claimed_hashes),
    }


def build_domain_records(
    domain: str, selected: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    shuffled = stable_shuffle(selected, seed, f"direction:{domain}")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(shuffled, start=1):
        zh_to_en = index <= RECORDS_PER_DIRECTION
        source_lang, target_lang = ("zh-CN", "en") if zh_to_en else ("en", "zh-CN")
        source_text = record["zh_text"] if zh_to_en else record["en_text"]
        target_text = record["en_text"] if zh_to_en else record["zh_text"]
        final_record: dict[str, Any] = {
            "id": record["id"],
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_text": source_text,
            "target_text": target_text,
            "zh_text": record["zh_text"],
            "en_text": record["en_text"],
            "zh_char_count": record["zh_char_count"],
            "domain": domain,
            "translation_method": record["translation_method"],
            "dataset_name": record.get("dataset_name"),
            "data_origin": record.get("data_origin"),
            "pair_sha256": record["pair_sha256"],
            "provenance": record.get("provenance"),
            "source_references": record["source_references"],
        }
        if record.get("pool_record_id") is not None:
            final_record["pool_record_id"] = record["pool_record_id"]
        records.append(final_record)
    return records


def csv_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "source_lang": record["source_lang"],
        "target_lang": record["target_lang"],
        "source_text": record["source_text"],
        "target_text": record["target_text"],
        "zh_char_count": str(record["zh_char_count"]),
        "domain": record["domain"],
        "translation_method": record["translation_method"],
    }


def validate_final_records(domain_records: dict[str, list[dict[str, Any]]]) -> None:
    global_hashes: set[str] = set()
    for domain in DOMAINS:
        records = domain_records[domain]
        if len(records) != RECORDS_PER_DOMAIN:
            raise RuntimeError(f"{domain} has {len(records)}, expected {RECORDS_PER_DOMAIN}")
        directions = Counter((record["source_lang"], record["target_lang"]) for record in records)
        if directions[("zh-CN", "en")] != RECORDS_PER_DIRECTION:
            raise RuntimeError(f"{domain} zh-CN->en count is wrong")
        if directions[("en", "zh-CN")] != RECORDS_PER_DIRECTION:
            raise RuntimeError(f"{domain} en->zh-CN count is wrong")
        for record in records:
            if record["domain"] != domain:
                raise RuntimeError(f"Domain mismatch in final {domain} record {record['id']}")
            if not record["source_text"].strip() or not record["target_text"].strip():
                raise RuntimeError(f"Empty source or target text in final record {record['id']}")
            if record["source_lang"] == "zh-CN":
                if record["source_text"] != record["zh_text"] or record["target_text"] != record["en_text"]:
                    raise RuntimeError(f"zh-CN->en mapping error in {record['id']}")
            else:
                if record["source_text"] != record["en_text"] or record["target_text"] != record["zh_text"]:
                    raise RuntimeError(f"en->zh-CN mapping error in {record['id']}")
            if record["zh_char_count"] != count_han(record["zh_text"]):
                raise RuntimeError(f"zh_char_count mismatch in {record['id']}")
            if record["pair_sha256"] in global_hashes:
                raise RuntimeError(f"Global duplicate pair_sha256: {record['pair_sha256']}")
            global_hashes.add(record["pair_sha256"])
    if len(global_hashes) != 30_000:
        raise RuntimeError(f"Global unique count is {len(global_hashes)}, expected 30000")


def validate_csv_header(final_dir: Path) -> tuple[str, ...]:
    existing_csv = final_dir / "zh_en.csv"
    if not existing_csv.is_file():
        return CSV_COLUMNS
    with existing_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if tuple(header or ()) != CSV_COLUMNS:
        got = ",".join(header or ())
        raise RuntimeError(
            f"Existing final CSV has an unexpected header ({got}); expected {','.join(CSV_COLUMNS)}"
        )
    return CSV_COLUMNS


def build_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], dict[Path, str]]:
    final_dir = resolve_path(args.final_dir)
    input_mode, paths, input_hashes = discover_inputs(final_dir)
    if input_mode == "prepared_intermediate":
        candidates, input_statistics = adapt_intermediate_inputs(paths)
    else:
        candidates, input_statistics = adapt_merged_pools(paths)
    candidates, within_domain_stats = deduplicate_within_domain(candidates)
    selected, global_selection_stats = select_globally_unique(candidates, args.seed)
    domain_records = {
        domain: build_domain_records(domain, selected[domain], args.seed)
        for domain in DOMAINS
    }
    validate_final_records(domain_records)
    csv_columns = validate_csv_header(final_dir)
    csv_records = [
        csv_row(record)
        for domain in DOMAINS
        for record in domain_records[domain]
    ]
    if len(csv_records) != 30_000:
        raise RuntimeError(f"CSV selection has {len(csv_records)}, expected 30000")

    for path, original_hash in input_hashes.items():
        if not path.is_file() or file_sha256(path) != original_hash:
            raise RuntimeError(f"Input changed while building: {path}")

    summary = {
        "schema_version": 1,
        "stage": "final_zh_en_dataset",
        "generated_at": utc_now(),
        "selection_seed": args.seed,
        "input_mode": input_mode,
        "input_files": {name: display_path(path) for name, path in paths.items()},
        "input_record_counts": input_statistics["input_record_counts"],
        "input_invalid_records": input_statistics["invalid_reasons"],
        "within_domain_deduplication": within_domain_stats,
        "global_selection": global_selection_stats,
        "final_counts": {
            domain: {
                "records": len(domain_records[domain]),
                "zh-CN_to_en": sum(
                    record["source_lang"] == "zh-CN" for record in domain_records[domain]
                ),
                "en_to_zh-CN": sum(
                    record["source_lang"] == "en" for record in domain_records[domain]
                ),
            }
            for domain in DOMAINS
        },
        "csv_columns": list(csv_columns),
        "csv_records": len(csv_records),
        "scope_guarantees": {
            "domain_scoring_performed": False,
            "raw_or_external_data_read": False,
            "input_pools_modified": False,
            "text_rewritten": False,
            "final_domain_reassigned_after_selection": False,
        },
    }
    outputs = {
        "domain_records": domain_records,
        "csv_columns": csv_columns,
        "csv_records": csv_records,
        "summary": summary,
    }
    return outputs, input_hashes


def stable_signature(outputs: dict[str, Any]) -> str:
    payload = {
        domain: [
            (record["pair_sha256"], record["source_lang"], record["target_lang"])
            for record in outputs["domain_records"][domain]
        ]
        for domain in DOMAINS
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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


def write_csv_atomic(path: Path, columns: Iterable[str], records: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="", errors="strict") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(final_dir: Path, outputs: dict[str, Any]) -> None:
    for domain in DOMAINS:
        value = {
            "schema_version": 1,
            "language_pair": "zh_en",
            "domain": domain,
            "stage": "final_selected_directional_dataset",
            "generated_at": outputs["summary"]["generated_at"],
            "selection_seed": SEED,
            "records": outputs["domain_records"][domain],
        }
        write_json_atomic(final_dir / f"{domain}.json", value)
    write_csv_atomic(final_dir / "zh_en.csv", outputs["csv_columns"], outputs["csv_records"])


def validate_written_outputs(final_dir: Path, outputs: dict[str, Any]) -> None:
    read_domain_records: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        path = final_dir / f"{domain}.json"
        value = load_json(path)
        if value.get("domain") != domain:
            raise RuntimeError(f"Written final JSON domain mismatch: {path}")
        read_domain_records[domain] = value["records"]
    validate_final_records(read_domain_records)

    expected_rows = [
        csv_row(record)
        for domain in DOMAINS
        for record in read_domain_records[domain]
    ]
    with (final_dir / "zh_en.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(outputs["csv_columns"]):
            raise RuntimeError("Written CSV header does not match the project final schema")
        actual_rows = list(reader)
    if actual_rows != expected_rows:
        raise RuntimeError("Written CSV rows are not identical to the final JSON records")
    if len(actual_rows) != 30_000:
        raise RuntimeError("Written CSV row count is not 30000")


def print_summary(summary: dict[str, Any], dry_run: bool) -> None:
    print(f"Input mode: {summary['input_mode']}")
    print("Input pools / views")
    print(f"{'domain':<12} {'input':>8} {'available':>10} {'selected':>10} {'zh->en':>8} {'en->zh':>8}")
    for domain in DOMAINS:
        available = summary["within_domain_deduplication"][domain]["available_after_within_domain_dedup"]
        final_counts = summary["final_counts"][domain]
        print(
            f"{domain:<12} {summary['input_record_counts'][domain]:>8,} {available:>10,} "
            f"{final_counts['records']:>10,} {final_counts['zh-CN_to_en']:>8,} "
            f"{final_counts['en_to_zh-CN']:>8,}"
        )
    print(
        "Global duplicates in input: "
        f"{summary['global_selection']['cross_domain_duplicate_groups_in_input']:,} groups; "
        "global final duplicates: 0"
    )
    print(f"CSV rows: {summary['csv_records']:,}")
    print("Dry run complete; no files written." if dry_run else "Final JSON and CSV written and revalidated.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    final_dir = resolve_path(args.final_dir)
    try:
        outputs, input_hashes = build_dataset(args)
        # Rebuild selection in memory to prove the fixed seed produces the same
        # records and direction assignments before any output is overwritten.
        repeat_outputs, repeat_hashes = build_dataset(args)
        if input_hashes != repeat_hashes or stable_signature(outputs) != stable_signature(repeat_outputs):
            raise RuntimeError("Determinism verification failed: repeated build differs")
        print_summary(outputs["summary"], args.dry_run)
        if args.dry_run:
            return 0
        write_outputs(final_dir, outputs)
        validate_written_outputs(final_dir, outputs)
        print("Formal output validation passed: 30,000 CSV rows and 3 x 10,000 JSON records.")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Final dataset build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
