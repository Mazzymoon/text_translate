#!/usr/bin/env python3
"""Extract 24,000 domain-balanced Chinese teacher inputs from frozen zh-en CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_CANDIDATES,
        DEFAULT_CANDIDATE_MANIFEST,
        DEFAULT_CONFIG,
        DEFAULT_SOURCE_CSV,
        DOMAINS,
        PIPELINE_VERSION,
        atomic_write_json,
        atomic_write_jsonl,
        count_han,
        ensure_new_outputs,
        load_config,
        normalize_text,
        relative_path,
        resolve_path,
        sha256_file,
        sha256_text,
        stable_derived_seed,
        utc_now,
    )
except ImportError:
    from common import (
        DEFAULT_CANDIDATES,
        DEFAULT_CANDIDATE_MANIFEST,
        DEFAULT_CONFIG,
        DEFAULT_SOURCE_CSV,
        DOMAINS,
        PIPELINE_VERSION,
        atomic_write_json,
        atomic_write_jsonl,
        count_han,
        ensure_new_outputs,
        load_config,
        normalize_text,
        relative_path,
        resolve_path,
        sha256_file,
        sha256_text,
        stable_derived_seed,
        utc_now,
    )


REQUIRED_FIELDS = {
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "zh_char_count",
    "domain",
    "translation_method",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def letter_count(text: str) -> int:
    return sum(unicodedata.category(char).startswith(("L", "M")) for char in text)


def candidate_id_for(zh_text: str) -> str:
    digest = sha256_text(normalize_text(zh_text).casefold())
    return f"zh_th_qwen8b_v3_candidate_{digest[:24]}"


def load_candidates(
    input_path: Path, config: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    quality = config.get("candidate_quality") or {}
    min_chars = int(quality.get("min_zh_chars", 100))
    max_chars = int(quality.get("max_zh_chars", 400))
    min_ratio = float(quality.get("min_han_letter_ratio", 0.5))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source: set[str] = set()
    rejects: Counter[str] = Counter()
    raw_rows = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(f"{input_path} is missing fields: {sorted(missing)}")
        for source_row, row in enumerate(reader, start=2):
            raw_rows += 1
            domain = str(row.get("domain", "")).strip()
            if domain not in DOMAINS:
                rejects["invalid_domain"] += 1
                continue
            source_lang = str(row.get("source_lang", "")).strip()
            target_lang = str(row.get("target_lang", "")).strip()
            if (source_lang, target_lang) == ("zh-CN", "en"):
                zh_raw = row.get("source_text") or ""
                en_text = row.get("target_text") or ""
            elif (source_lang, target_lang) == ("en", "zh-CN"):
                zh_raw = row.get("target_text") or ""
                en_text = row.get("source_text") or ""
            else:
                rejects["invalid_language_pair"] += 1
                continue
            zh_text = normalize_text(zh_raw)
            if not zh_text:
                rejects["empty_chinese"] += 1
                continue
            zh_chars = count_han(zh_text)
            if zh_chars < min_chars:
                rejects["chinese_too_short"] += 1
                continue
            if zh_chars > max_chars:
                rejects["chinese_too_long"] += 1
                continue
            han_ratio = zh_chars / max(letter_count(zh_text), 1)
            if han_ratio < min_ratio:
                rejects["chinese_not_han_dominant"] += 1
                continue
            source_key = sha256_text(zh_text.casefold())
            if source_key in seen_source:
                rejects["duplicate_chinese_source"] += 1
                continue
            seen_source.add(source_key)
            data_index = source_row - 1
            original_id = f"zh_en_csv_{data_index:06d}"
            pools[domain].append(
                {
                    "candidate_id": candidate_id_for(zh_text),
                    "original_id": original_id,
                    "domain": domain,
                    "source_lang": "zh-CN",
                    "target_lang": "th",
                    "source_text": zh_text,
                    "zh_char_count": zh_chars,
                    "source_file": relative_path(input_path),
                    "source_row": source_row,
                    "provenance": {
                        "source_file": relative_path(input_path),
                        "source_row": source_row,
                        "original_id": original_id,
                        "original_direction": f"{source_lang}->{target_lang}",
                        "original_english_text": en_text,
                        "original_translation_method": row.get("translation_method") or None,
                        "original_zh_char_count": row.get("zh_char_count") or None,
                    },
                }
            )
    return pools, {
        "raw_rows": raw_rows,
        "rejection_reason_counts": dict(sorted(rejects.items())),
        "unique_sane_chinese_sources": len(seen_source),
        "available_by_domain": {domain: len(pools[domain]) for domain in DOMAINS},
        "candidate_quality": {
            "min_zh_chars": min_chars,
            "max_zh_chars": max_chars,
            "min_han_letter_ratio": min_ratio,
        },
    }


def select_candidates(
    pools: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    seed = int(config["seed"])
    targets = {domain: int(config["candidate_domain_targets"][domain]) for domain in DOMAINS}
    selected: list[dict[str, Any]] = []
    shortages: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        candidates = list(pools[domain])
        random.Random(stable_derived_seed(seed, f"candidate-domain:{domain}")).shuffle(candidates)
        if len(candidates) < targets[domain]:
            shortages[domain] = {"available": len(candidates), "required": targets[domain]}
            continue
        selected.extend(candidates[: targets[domain]])
    if shortages:
        raise RuntimeError(
            "Insufficient unique Chinese candidates; no resampling is allowed: "
            + json.dumps(shortages, ensure_ascii=False)
        )
    random.Random(stable_derived_seed(seed, "candidate-final-order")).shuffle(selected)
    candidate_ids = [row["candidate_id"] for row in selected]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("Internal error: selected candidate_id values are not unique")
    return selected


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    input_path = resolve_path(args.input_csv)
    output_path = resolve_path(args.output_file)
    manifest_path = resolve_path(args.manifest_file)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    config = load_config(config_path)
    pools, scan = load_candidates(input_path, config)
    selected = select_candidates(pools, config)
    selected_counts = Counter(row["domain"] for row in selected)
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "stage": "teacher_candidate_preparation",
        "created_at": utc_now(),
        "seed": int(config["seed"]),
        "input_file": {
            "path": relative_path(input_path),
            "sha256": sha256_file(input_path),
            "schema": sorted(REQUIRED_FIELDS),
        },
        **scan,
        "candidate_count": len(selected),
        "selected_by_domain": dict(selected_counts),
        "candidate_id_policy": "SHA256(NFC normalized Chinese source), first 24 hex characters",
        "original_id_policy": "stable 1-based CSV data row index",
        "output_file": relative_path(output_path),
        "dry_run": args.dry_run,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    ensure_new_outputs((output_path, manifest_path), args.overwrite)
    atomic_write_jsonl(output_path, selected)
    manifest["output_sha256"] = sha256_file(output_path)
    manifest["dry_run"] = False
    atomic_write_json(manifest_path, manifest)
    print(f"candidates: {output_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Candidate preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
