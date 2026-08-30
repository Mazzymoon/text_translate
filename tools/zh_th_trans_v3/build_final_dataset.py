#!/usr/bin/env python3
"""Build the deterministic 20k Chinese--Thai v3 submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_ACCEPTED,
        DEFAULT_AUDIT_SUMMARY,
        DEFAULT_CANDIDATE_MANIFEST,
        DEFAULT_CONFIG,
        DEFAULT_FINAL_CSV,
        DEFAULT_FINAL_MANIFEST,
        DOMAINS,
        PIPELINE_VERSION,
        PROMPT_TEMPLATE_VERSION,
        QUALITY_RULE_VERSION,
        atomic_text_writer,
        atomic_write_json,
        canonical_pair_group_id,
        ensure_new_outputs,
        exact_pair_key,
        load_config,
        read_json,
        read_jsonl,
        relative_path,
        resolve_path,
        sha256_file,
        stable_derived_seed,
        utc_now,
    )
except ImportError:
    from common import (
        DEFAULT_ACCEPTED,
        DEFAULT_AUDIT_SUMMARY,
        DEFAULT_CANDIDATE_MANIFEST,
        DEFAULT_CONFIG,
        DEFAULT_FINAL_CSV,
        DEFAULT_FINAL_MANIFEST,
        DOMAINS,
        PIPELINE_VERSION,
        PROMPT_TEMPLATE_VERSION,
        QUALITY_RULE_VERSION,
        atomic_text_writer,
        atomic_write_json,
        canonical_pair_group_id,
        ensure_new_outputs,
        exact_pair_key,
        load_config,
        read_json,
        read_jsonl,
        relative_path,
        resolve_path,
        sha256_file,
        stable_derived_seed,
        utc_now,
    )


CSV_COLUMNS = [
    "record_id",
    "original_id",
    "candidate_id",
    "pair_group_id",
    "pair_sha256",
    "direction",
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "domain",
    "translation_method",
    "teacher_model",
    "prompt_template_version",
    "source_file",
    "source_row",
    "zh_char_count",
    "thai_char_count",
    "repeat_score",
    "thai_ratio",
    "quality_status",
    "quality_warnings",
    "semantic_qe_score",
    "semantic_status",
    "dataset_name",
    "data_origin",
    "generation_config",
    "provenance",
]
REQUIRED_ACCEPTED_FIELDS = {
    "candidate_id",
    "original_id",
    "domain",
    "source_text",
    "target_text",
    "zh_char_count",
    "thai_char_count",
    "teacher_model",
    "translation_method",
    "prompt_template_version",
    "generation_config",
    "source_file",
    "source_row",
    "accepted",
    "quality_status",
    "warnings",
    "metrics",
    "pair_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-file", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qe-score-file", type=Path)
    parser.add_argument("--min-qe-score", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def largest_remainder(total: int, weights: list[int]) -> list[int]:
    weight_total = sum(weights)
    exact = [total * weight / weight_total for weight in weights]
    counts = [math.floor(value) for value in exact]
    missing = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:missing]:
        counts[index] += 1
    return counts


def load_qe_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}

    def add(candidate_id: Any, score: Any) -> None:
        key = str(candidate_id or "").strip()
        if not key:
            raise ValueError("QE record has no candidate_id")
        if key in scores:
            raise ValueError(f"Duplicate candidate_id in QE scores: {key}")
        scores[key] = float(score)

    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                add(row.get("candidate_id"), row.get("semantic_qe_score"))
    elif path.suffix.casefold() == ".jsonl":
        for _, row in read_jsonl(path):
            add(row.get("candidate_id"), row.get("semantic_qe_score"))
    else:
        value = read_json(path)
        if isinstance(value, dict):
            for candidate_id, score in value.items():
                add(candidate_id, score)
        elif isinstance(value, list):
            for row in value:
                if not isinstance(row, dict):
                    raise ValueError("QE JSON list items must be objects")
                add(row.get("candidate_id"), row.get("semantic_qe_score"))
        else:
            raise ValueError("QE score file must be CSV, JSONL, object JSON, or list JSON")
    return scores


def load_eligible(
    path: Path, qe_scores: dict[str, float] | None, min_qe_score: float | None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: set[str] = set()
    seen_pairs: set[str] = set()
    filtered = Counter()
    teacher_models: set[str] = set()
    prompt_versions: set[str] = set()
    generation_configs: set[str] = set()
    total = 0
    for line_number, row in read_jsonl(path):
        total += 1
        missing = REQUIRED_ACCEPTED_FIELDS - set(row)
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        if row.get("accepted") is not True or row.get("quality_status") != "accepted":
            filtered["not_rule_accepted"] += 1
            continue
        domain = str(row.get("domain", ""))
        if domain not in DOMAINS:
            filtered["invalid_domain"] += 1
            continue
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_candidates:
            filtered["duplicate_candidate_id"] += 1
            continue
        seen_candidates.add(candidate_id)
        pair_hash = exact_pair_key(str(row["source_text"]), str(row["target_text"]))
        if pair_hash != row["pair_sha256"]:
            raise ValueError(f"{path}:{line_number} stored pair_sha256 mismatch")
        if pair_hash in seen_pairs:
            filtered["duplicate_pair"] += 1
            continue
        seen_pairs.add(pair_hash)

        score = qe_scores.get(candidate_id) if qe_scores is not None else None
        if min_qe_score is not None and (score is None or score < min_qe_score):
            filtered["below_or_missing_qe_threshold"] += 1
            continue
        row["semantic_qe_score"] = score
        row["semantic_status"] = "evaluated" if score is not None else "not_evaluated"
        by_domain[domain].append(row)
        teacher_models.add(str(row["teacher_model"]))
        prompt_versions.add(str(row["prompt_template_version"]))
        generation_configs.add(json.dumps(row["generation_config"], sort_keys=True))
    if len(teacher_models) != 1:
        raise ValueError(f"Expected one teacher model, found: {sorted(teacher_models)}")
    if prompt_versions != {PROMPT_TEMPLATE_VERSION}:
        raise ValueError(f"Unexpected prompt template versions: {sorted(prompt_versions)}")
    if len(generation_configs) != 1:
        raise ValueError("Accepted rows contain multiple generation configurations")
    return by_domain, {
        "input_rows": total,
        "eligible_by_domain": {domain: len(by_domain[domain]) for domain in DOMAINS},
        "filtered_reason_counts": dict(filtered),
        "teacher_model": next(iter(teacher_models)),
        "prompt_template_version": next(iter(prompt_versions)),
        "generation_config": json.loads(next(iter(generation_configs))),
    }


def select_and_allocate(
    by_domain: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, int]]]:
    seed = int(config["seed"])
    final_total = int(config["final_pair_count"])
    domain_counts_list = largest_remainder(final_total, [1] * len(DOMAINS))
    domain_targets = dict(zip(DOMAINS, domain_counts_list))
    zh_total = int(config["final_direction_targets"]["zh-CN->th"])
    zh_by_domain_list = largest_remainder(zh_total, domain_counts_list)
    direction_domain_counts: dict[str, dict[str, int]] = {}
    allocated: list[dict[str, Any]] = []

    shortages = {
        domain: {"available": len(by_domain[domain]), "required": domain_targets[domain]}
        for domain in DOMAINS
        if len(by_domain[domain]) < domain_targets[domain]
    }
    if shortages:
        raise RuntimeError(
            "Not enough accepted pairs; rejected/review records cannot be recycled: "
            + json.dumps(shortages, ensure_ascii=False)
        )

    for domain, zh_count in zip(DOMAINS, zh_by_domain_list):
        candidates = list(by_domain[domain])
        random.Random(stable_derived_seed(seed, f"final-domain:{domain}")).shuffle(candidates)
        chosen = candidates[: domain_targets[domain]]
        direction_domain_counts[domain] = {
            "zh-CN->th": zh_count,
            "th->zh-CN": len(chosen) - zh_count,
        }
        for index, row in enumerate(chosen):
            item = dict(row)
            item["direction"] = "zh-CN->th" if index < zh_count else "th->zh-CN"
            allocated.append(item)
    random.Random(stable_derived_seed(seed, "final-record-order")).shuffle(allocated)
    return allocated, domain_targets, direction_domain_counts


def csv_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    zh_text = str(row["source_text"])
    th_text = str(row["target_text"])
    direction = row["direction"]
    if direction == "zh-CN->th":
        source_lang, target_lang = "zh-CN", "th"
        source_text, target_text = zh_text, th_text
    else:
        source_lang, target_lang = "th", "zh-CN"
        source_text, target_text = th_text, zh_text
    metrics = row["metrics"]
    provenance = row.get("provenance") or {
        "source_file": row.get("source_file"),
        "source_row": row.get("source_row"),
        "original_id": row.get("original_id"),
    }
    return {
        "record_id": f"zh_th_qwen8b_v3_{index:06d}",
        "original_id": row["original_id"],
        "candidate_id": row["candidate_id"],
        "pair_group_id": canonical_pair_group_id(zh_text, th_text),
        "pair_sha256": row["pair_sha256"],
        "direction": direction,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_text": source_text,
        "target_text": target_text,
        "domain": row["domain"],
        "translation_method": "qwen3_8b_teacher",
        "teacher_model": row["teacher_model"],
        "prompt_template_version": row["prompt_template_version"],
        "source_file": row["source_file"],
        "source_row": row["source_row"],
        "zh_char_count": row["zh_char_count"],
        "thai_char_count": row["thai_char_count"],
        "repeat_score": metrics["repeat_score"],
        "thai_ratio": metrics["thai_ratio"],
        "quality_status": "accepted",
        "quality_warnings": json.dumps(row.get("warnings") or [], ensure_ascii=False),
        "semantic_qe_score": "" if row.get("semantic_qe_score") is None else row["semantic_qe_score"],
        "semantic_status": row.get("semantic_status") or "not_evaluated",
        "dataset_name": "Qwen3-8B Teacher Chinese-Thai v3",
        "data_origin": "teacher_generated_parallel",
        "generation_config": json.dumps(row["generation_config"], ensure_ascii=False, sort_keys=True),
        "provenance": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with atomic_text_writer(path, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    accepted_path = resolve_path(args.accepted_file)
    output_path = resolve_path(args.output_csv)
    manifest_path = resolve_path(args.manifest_file)
    v2_path = resolve_path(Path("dataset/final/zh_th/zh_th_clean_v2.csv"))
    if output_path.resolve() == v2_path.resolve():
        raise ValueError("Refusing to overwrite the frozen zh_th_clean_v2.csv")
    config = load_config(resolve_path(args.config))
    if args.min_qe_score is not None and args.qe_score_file is None:
        raise ValueError("--min-qe-score requires --qe-score-file")
    qe_path = resolve_path(args.qe_score_file) if args.qe_score_file else None
    qe_scores = load_qe_scores(qe_path) if qe_path else None
    by_domain, load_report = load_eligible(accepted_path, qe_scores, args.min_qe_score)
    allocated, domain_targets, direction_domain_counts = select_and_allocate(by_domain, config)
    rows = [csv_record(row, index) for index, row in enumerate(allocated, 1)]

    record_ids = [row["record_id"] for row in rows]
    candidate_ids = [row["candidate_id"] for row in rows]
    pair_groups = [row["pair_group_id"] for row in rows]
    pair_hashes = [row["pair_sha256"] for row in rows]
    direction_counts = Counter(row["direction"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    source_target_duplicates = sum(
        str(row["source_text"]).casefold() == str(row["target_text"]).casefold() for row in rows
    )
    duplicate_audit = {
        "unique_record_ids": len(set(record_ids)),
        "unique_candidate_ids": len(set(candidate_ids)),
        "unique_pair_group_ids": len(set(pair_groups)),
        "unique_pair_sha256": len(set(pair_hashes)),
        "source_target_duplicate": source_target_duplicates,
    }
    expected_total = int(config["final_pair_count"])
    if not all(
        count == expected_total
        for count in (
            len(rows),
            duplicate_audit["unique_record_ids"],
            duplicate_audit["unique_candidate_ids"],
            duplicate_audit["unique_pair_group_ids"],
            duplicate_audit["unique_pair_sha256"],
        )
    ) or source_target_duplicates:
        raise RuntimeError(f"Final uniqueness audit failed: {duplicate_audit}")
    expected_directions = {
        direction: int(count) for direction, count in config["final_direction_targets"].items()
    }
    if dict(direction_counts) != expected_directions:
        raise RuntimeError(
            f"Direction allocation mismatch: actual={dict(direction_counts)} expected={expected_directions}"
        )

    candidate_manifest_path = resolve_path(args.candidate_manifest)
    audit_summary_path = resolve_path(args.audit_summary)
    candidate_manifest = read_json(candidate_manifest_path)
    audit_summary = read_json(audit_summary_path)
    manifest = {
        "schema_version": 1,
        "dataset_version": "zh_th_qwen3_8b_v3",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": utc_now(),
        "seed": int(config["seed"]),
        "candidate_count": candidate_manifest.get("candidate_count"),
        "teacher_generated_count": audit_summary.get("total"),
        "audit_accepted_count": audit_summary.get("accepted"),
        "audit_rejected_count": audit_summary.get("rejected"),
        "final_pair_count": len(rows),
        "direction_counts": dict(direction_counts),
        "domain_counts": dict(domain_counts),
        "direction_domain_counts": direction_domain_counts,
        "domain_targets": domain_targets,
        "teacher_model": load_report["teacher_model"],
        "teacher_generated_target": "Qwen3-8B",
        "planned_student_target": "Qwen3-4B",
        "prompt_template_version": load_report["prompt_template_version"],
        "generation_config": load_report["generation_config"],
        "quality_rules": {
            "path": "tools/zh_th_trans/quality_rules_v2.py",
            "version": QUALITY_RULE_VERSION,
        },
        "semantic_qe": {
            "score_file": relative_path(qe_path) if qe_path else None,
            "minimum_score": args.min_qe_score,
            "performed": qe_scores is not None,
        },
        "source_dataset_provenance": candidate_manifest.get("input_file"),
        "inputs": {
            "accepted": {"path": relative_path(accepted_path), "sha256": sha256_file(accepted_path)},
            "candidate_manifest": {
                "path": relative_path(candidate_manifest_path),
                "sha256": sha256_file(candidate_manifest_path),
            },
            "audit_summary": {
                "path": relative_path(audit_summary_path),
                "sha256": sha256_file(audit_summary_path),
            },
        },
        "eligible_load_report": load_report,
        "duplicate_audit": duplicate_audit,
        "csv_columns": CSV_COLUMNS,
        "output_csv": relative_path(output_path),
        "output_csv_sha256": None,
        "dry_run": args.dry_run,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    ensure_new_outputs((output_path, manifest_path), args.overwrite)
    write_csv(output_path, rows)
    manifest["output_csv_sha256"] = sha256_file(output_path)
    manifest["dry_run"] = False
    atomic_write_json(manifest_path, manifest)
    print(f"final CSV: {output_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Final dataset build failed: {error}", file=sys.stderr)
        raise SystemExit(1)
