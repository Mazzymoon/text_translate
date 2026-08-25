#!/usr/bin/env python3
"""Prepare leakage-safe Qwen3 translation SFT splits from the two final CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZH_EN = ROOT / "dataset/final/zh_en/zh_en.csv"
DEFAULT_ZH_TH = ROOT / "dataset/final/zh_th/zh_th.csv"
DEFAULT_OUTPUT = ROOT / "outputs/sft_data"
ALLOWED_DIRECTIONS = {"zh-CN->en", "en->zh-CN", "zh-CN->th", "th->zh-CN"}
LANGUAGE_NAMES = {"zh-CN": "Chinese", "en": "English", "th": "Thai"}
SYSTEM_PROMPT = (
    "You are a professional translator. Translate accurately and output only the "
    "translation, without explanations or additional text."
)
PROMPT_TEMPLATE_VERSION = "qwen3_translation_non_thinking_v1"
PROMPT_TEMPLATE_SPEC = {
    "version": PROMPT_TEMPLATE_VERSION,
    "system": SYSTEM_PROMPT,
    "user": "Translate the following text from {source_language} to {target_language}:\n\n{source_text}",
    "enable_thinking": False,
}
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    json.dumps(PROMPT_TEMPLATE_SPEC, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()
ZERO_WIDTH = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u2060"), None)
LOOP_PATTERN = re.compile(r"(.{2,50}?)\1{4,}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh-en-csv", type=Path, default=DEFAULT_ZH_EN)
    parser.add_argument("--zh-th-csv", type=Path, default=DEFAULT_ZH_TH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name-or-path", help="Local Qwen3 model/tokenizer path for formal output")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow Hugging Face network access")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Read and report only; write no output")
    return parser.parse_args()


def normalize_for_comparison(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(ZERO_WIDTH)
    text = text.replace("\u3000", " ").replace("\u00a0", " ")
    return " ".join(text.split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def script_counts(text: str) -> dict[str, int]:
    counts = {"han": 0, "latin": 0, "thai": 0, "letters": 0}
    for char in text:
        if is_han(char):
            counts["han"] += 1
            counts["letters"] += 1
        elif "\u0e00" <= char <= "\u0e7f":
            counts["thai"] += 1
            if not char.isspace():
                counts["letters"] += 1
        elif "LATIN" in unicodedata.name(char, "") and char.isalpha():
            counts["latin"] += 1
            counts["letters"] += 1
        elif char.isalpha():
            counts["letters"] += 1
    return counts


def language_quality(text: str, lang: str) -> tuple[bool, float, int]:
    counts = script_counts(text)
    denominator = max(1, counts["han"] + counts["latin"] + counts["thai"])
    if lang == "zh-CN":
        ratio, minimum, threshold = counts["han"] / denominator, 10, 0.40
        amount = counts["han"]
    elif lang == "en":
        ratio, minimum, threshold = counts["latin"] / denominator, 20, 0.70
        amount = counts["latin"]
    elif lang == "th":
        ratio, minimum, threshold = counts["thai"] / denominator, 10, 0.50
        amount = counts["thai"]
    else:
        return False, 0.0, 0
    return amount >= minimum and ratio >= threshold, ratio, amount


def has_severe_repetition(text: str) -> tuple[bool, dict[str, Any]]:
    normalized = normalize_for_comparison(text)
    if not normalized:
        return False, {}
    raw = normalized.encode("utf-8")
    compression_ratio = len(zlib.compress(raw, 9)) / max(1, len(raw))
    loop = LOOP_PATTERN.search(normalized)
    if loop and len(loop.group(0)) >= 80:
        return True, {
            "rule": "continuous_loop",
            "repeated_span_chars": len(loop.group(0)),
            "unit_chars": len(loop.group(1)),
            "compression_ratio": round(compression_ratio, 6),
        }
    if len(normalized) >= 100:
        grams = [normalized[index : index + 8] for index in range(len(normalized) - 7)]
        unique_ratio = len(set(grams)) / max(1, len(grams))
        if unique_ratio < 0.15 and compression_ratio < 0.12:
            return True, {
                "rule": "global_ngram_collapse",
                "unique_8gram_ratio": round(unique_ratio, 6),
                "compression_ratio": round(compression_ratio, 6),
            }
    return False, {"compression_ratio": round(compression_ratio, 6)}


def build_messages(source_lang: str, target_lang: str, source_text: str) -> list[dict[str, str]]:
    user = PROMPT_TEMPLATE_SPEC["user"].format(
        source_language=LANGUAGE_NAMES[source_lang],
        target_language=LANGUAGE_NAMES[target_lang],
        source_text=source_text,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def canonical_pair_id(source_lang: str, target_lang: str, source_text: str, target_text: str) -> str:
    texts = {
        source_lang: normalize_for_comparison(source_text).casefold(),
        target_lang: normalize_for_comparison(target_text).casefold(),
    }
    payload = "\x1f".join(f"{lang}\x1e{texts[lang]}" for lang in sorted(texts))
    return "pair_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_tokenizer(model_name_or_path: str, allow_model_download: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenizer validation") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=not allow_model_download,
        trust_remote_code=False,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer has no eos_token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def inspect_completion_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    completion: str,
    max_length: int,
) -> dict[str, int | bool]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    available = max(0, max_length - len(prompt_ids))
    valid_target_tokens = min(len(target_ids), available)
    full_length = len(prompt_ids) + len(target_ids) + len(eos_ids)
    return {
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "valid_target_tokens": valid_target_tokens,
        "full_tokens": full_length,
        "truncated": full_length > max_length,
    }


def primary_rejection(reasons: list[str]) -> str:
    priority = [
        "empty_source",
        "empty_target",
        "source_equals_target",
        "invalid_direction",
        "source_language_mismatch",
        "target_language_mismatch",
        "severe_target_repetition",
        "duplicate_same_direction",
        "completion_no_supervision",
        "tokenization_error",
    ]
    for reason in priority:
        if reason in reasons:
            return reason
    return reasons[0]


def rejection_record(record: dict[str, Any], reasons: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_id": record["original_id"],
        "direction": record["direction"],
        "source_text": record["source_text"],
        "target_text": record["target_text"],
        "rejection_reason": primary_rejection(reasons),
        "all_rejection_reasons": reasons,
        "rejection_details": details,
        "source_file": record["source_file"],
        "source_row": record["source_row"],
    }


def read_csv_records(path: Path, prefix: str, required_fields: set[str]) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required_fields - fields
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        for data_index, row in enumerate(reader, start=1):
            source_lang = (row.get("source_lang") or "").strip()
            target_lang = (row.get("target_lang") or "").strip()
            yield {
                "record_id": f"{prefix}_{data_index:06d}",
                "original_id": f"{prefix}_{data_index:06d}",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "direction": f"{source_lang}->{target_lang}",
                "source_text": row.get("source_text") or "",
                "target_text": row.get("target_text") or "",
                "zh_char_count": row.get("zh_char_count"),
                "domain": row.get("domain") or None,
                "translation_method": row.get("translation_method") or None,
                "source_file": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                "source_row": data_index + 1,
            }


def largest_remainder_counts(total: int, ratios: list[float]) -> list[int]:
    exact = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    missing = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:missing]:
        counts[index] += 1
    return counts


def split_by_pair_group(
    records: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["pair_group_id"]].append(record)
    strata: dict[str, list[str]] = defaultdict(list)
    for pair_id, members in groups.items():
        language_pair = "-".join(sorted({members[0]["source_lang"], members[0]["target_lang"]}))
        composition = "+".join(sorted(member["direction"] for member in members))
        strata[f"{language_pair}|{composition}"].append(pair_id)

    assignments: dict[str, str] = {}
    split_names = ("train", "validation", "test")
    for stratum, pair_ids in sorted(strata.items()):
        derived_seed = seed ^ int(hashlib.sha256(stratum.encode("utf-8")).hexdigest()[:16], 16)
        random.Random(derived_seed).shuffle(pair_ids)
        counts = largest_remainder_counts(len(pair_ids), list(ratios))
        cursor = 0
        for split_name, count in zip(split_names, counts):
            for pair_id in pair_ids[cursor : cursor + count]:
                assignments[pair_id] = split_name
            cursor += count

    splits = {name: [] for name in split_names}
    for pair_id, members in groups.items():
        splits[assignments[pair_id]].extend(members)
    for index, name in enumerate(split_names):
        random.Random(seed + index).shuffle(splits[name])
        for record in splits[name]:
            record["split"] = name
    return splits


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("train/validation/test ratios must be non-negative and sum to 1")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    if not args.dry_run and not args.model_name_or_path:
        raise ValueError("Formal output requires --model-name-or-path for completion-token validation")

    input_paths = [args.zh_en_csv.resolve(), args.zh_th_csv.resolve()]
    before_hashes = {str(path): sha256_file(path) for path in input_paths}
    tokenizer = None
    if args.model_name_or_path:
        tokenizer = load_tokenizer(args.model_name_or_path, args.allow_model_download)

    specs = [
        (input_paths[0], "zh_en_csv", {"source_lang", "target_lang", "source_text", "target_text", "zh_char_count", "domain", "translation_method"}),
        (input_paths[1], "zh_th_csv", {"source_lang", "target_lang", "source_text", "target_text", "zh_char_count", "translation_method"}),
    ]
    raw_total = 0
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    seen_pair_directions: set[tuple[str, str]] = set()

    for path, prefix, required_fields in specs:
        for record in read_csv_records(path, prefix, required_fields):
            raw_total += 1
            reasons: list[str] = []
            details: dict[str, Any] = {}
            source_norm = normalize_for_comparison(record["source_text"])
            target_norm = normalize_for_comparison(record["target_text"])
            if not source_norm:
                reasons.append("empty_source")
            if not target_norm:
                reasons.append("empty_target")
            if source_norm and target_norm and source_norm.casefold() == target_norm.casefold():
                reasons.append("source_equals_target")
            if record["direction"] not in ALLOWED_DIRECTIONS:
                reasons.append("invalid_direction")

            if record["source_lang"] in LANGUAGE_NAMES and source_norm:
                valid, ratio, amount = language_quality(source_norm, record["source_lang"])
                details["source_language_ratio"] = round(ratio, 6)
                details["source_language_chars"] = amount
                if not valid:
                    reasons.append("source_language_mismatch")
                elif ratio < 0.85:
                    record.setdefault("warnings", []).append("source_language_ratio_borderline")
            if record["target_lang"] in LANGUAGE_NAMES and target_norm:
                valid, ratio, amount = language_quality(target_norm, record["target_lang"])
                details["target_language_ratio"] = round(ratio, 6)
                details["target_language_chars"] = amount
                if not valid:
                    reasons.append("target_language_mismatch")
                elif ratio < 0.85:
                    record.setdefault("warnings", []).append("target_language_ratio_borderline")

            if target_norm:
                repeated, repeat_details = has_severe_repetition(target_norm)
                if repeated:
                    reasons.append("severe_target_repetition")
                    details["repetition"] = repeat_details

            if source_norm and target_norm:
                length_ratio = len(target_norm) / max(1, len(source_norm))
                details["target_source_length_ratio"] = round(length_ratio, 6)
                if length_ratio < 0.20 or length_ratio > 5.0:
                    record.setdefault("warnings", []).append("length_ratio_extreme")
                if max(len(source_norm), len(target_norm)) > 2000:
                    record.setdefault("warnings", []).append("long_text")

            if record["source_lang"] in LANGUAGE_NAMES and record["target_lang"] in LANGUAGE_NAMES:
                record["pair_group_id"] = canonical_pair_id(
                    record["source_lang"], record["target_lang"], record["source_text"], record["target_text"]
                )
                duplicate_key = (record["pair_group_id"], record["direction"])
                if not reasons and duplicate_key in seen_pair_directions:
                    reasons.append("duplicate_same_direction")

            record["prompt_messages"] = build_messages(
                record["source_lang"], record["target_lang"], record["source_text"]
            ) if record["direction"] in ALLOWED_DIRECTIONS else []
            record["completion"] = record["target_text"]
            record["prompt_template_version"] = PROMPT_TEMPLATE_VERSION

            if not reasons and tokenizer is not None:
                try:
                    token_info = inspect_completion_tokens(
                        tokenizer, record["prompt_messages"], record["completion"], args.max_length
                    )
                    details["tokenization"] = token_info
                    if int(token_info["valid_target_tokens"]) <= 0:
                        reasons.append("completion_no_supervision")
                    elif bool(token_info["truncated"]):
                        record.setdefault("warnings", []).append("sequence_truncated")
                except Exception as exc:  # keep the failing record traceable
                    reasons.append("tokenization_error")
                    details["tokenization_error"] = f"{type(exc).__name__}: {exc}"

            if reasons:
                item = rejection_record(record, list(dict.fromkeys(reasons)), details)
                rejected.append(item)
                rejection_counts[item["rejection_reason"]] += 1
                continue

            seen_pair_directions.add((record["pair_group_id"], record["direction"]))
            record["warnings"] = sorted(set(record.get("warnings", [])))
            for warning in record["warnings"]:
                warning_counts[warning] += 1
            accepted.append(record)

    splits = split_by_pair_group(accepted, ratios, args.seed)
    direction_counts = {
        split: dict(sorted(Counter(row["direction"] for row in rows).items()))
        for split, rows in splits.items()
    }
    pair_group_counts = {split: len({row["pair_group_id"] for row in rows}) for split, rows in splits.items()}
    after_hashes = {str(path): sha256_file(path) for path in input_paths}
    if before_hashes != after_hashes:
        raise RuntimeError("An input CSV changed while preparing data")

    summary = {
        "schema_version": 1,
        "dry_run": args.dry_run,
        "tokenizer_validation_performed": tokenizer is not None,
        "raw_total": raw_total,
        "accepted_total": len(accepted),
        "rejected_total": len(rejected),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "pair_group_counts": pair_group_counts,
        "direction_counts": direction_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    output_dir = args.output_dir.resolve()
    for split_name, rows in splits.items():
        atomic_write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)
    rejection_report = {
        "schema_version": 1,
        "raw_total": raw_total,
        "accepted_total": len(accepted),
        "rejected_total": len(rejected),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }
    atomic_write_json(output_dir / "rejection_report.json", rejection_report)
    manifest = {
        **summary,
        "dry_run": False,
        "seed": args.seed,
        "split_ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]},
        "max_length": args.max_length,
        "prompt_template": PROMPT_TEMPLATE_SPEC,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "model_name_or_path": args.model_name_or_path,
        "input_files": [
            {"path": str(path), "sha256": before_hashes[str(path)]} for path in input_paths
        ],
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
