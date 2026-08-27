#!/usr/bin/env python3
"""Validate prepared Qwen3 translation SFT splits and pair-group isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "outputs/sft_data"
SPLITS = ("train", "validation", "test")
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name-or-path", help="Local tokenizer path for token-length validation")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--skip-tokenizer-check",
        action="store_true",
        help="Allow structural-only local validation; never use this for the formal AutoDL check",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def normalize_for_comparison(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(ZERO_WIDTH)
    text = text.replace("\u3000", " ").replace("\u00a0", " ")
    return " ".join(text.split())


def canonical_pair_id(source_lang: str, target_lang: str, source_text: str, target_text: str) -> str:
    texts = {
        source_lang: normalize_for_comparison(source_text).casefold(),
        target_lang: normalize_for_comparison(target_text).casefold(),
    }
    payload = "\x1f".join(f"{lang}\x1e{texts[lang]}" for lang in sorted(texts))
    return "pair_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def language_quality(text: str, lang: str) -> tuple[bool, float]:
    counts = {"han": 0, "latin": 0, "thai": 0}
    for char in text:
        if is_han(char):
            counts["han"] += 1
        elif "\u0e00" <= char <= "\u0e7f":
            counts["thai"] += 1
        elif "LATIN" in unicodedata.name(char, "") and char.isalpha():
            counts["latin"] += 1
    denominator = max(1, sum(counts.values()))
    if lang == "zh-CN":
        return counts["han"] >= 10 and counts["han"] / denominator >= 0.40, counts["han"] / denominator
    if lang == "en":
        return counts["latin"] >= 20 and counts["latin"] / denominator >= 0.70, counts["latin"] / denominator
    if lang == "th":
        return counts["thai"] >= 10 and counts["thai"] / denominator >= 0.50, counts["thai"] / denominator
    return False, 0.0


def has_severe_repetition(text: str) -> bool:
    normalized = normalize_for_comparison(text)
    loop = LOOP_PATTERN.search(normalized)
    if loop and len(loop.group(0)) >= 80:
        return True
    if len(normalized) < 100:
        return False
    raw = normalized.encode("utf-8")
    compression_ratio = len(zlib.compress(raw, 9)) / max(1, len(raw))
    grams = [normalized[index : index + 8] for index in range(len(normalized) - 7)]
    unique_ratio = len(set(grams)) / max(1, len(grams))
    return unique_ratio < 0.15 and compression_ratio < 0.12


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, value


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


def completion_token_info(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, int | bool]:
    prompt = tokenizer.apply_chat_template(
        record["prompt_messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(record["completion"], add_special_tokens=False)["input_ids"]
    eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    valid = min(len(completion_ids), max(0, max_length - len(prompt_ids)))
    return {
        "valid_completion_tokens": valid,
        "truncated": len(prompt_ids) + len(completion_ids) + len(eos_ids) > max_length,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    if not args.model_name_or_path and not args.skip_tokenizer_check:
        raise ValueError(
            "Formal validation requires --model-name-or-path; use --skip-tokenizer-check only for local structural tests"
        )
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    rejection_report_path = data_dir / "rejection_report.json"
    rejected_path = data_dir / "rejected.jsonl"
    manifest = read_json(manifest_path)
    rejection_report = read_json(rejection_report_path)
    tokenizer = load_tokenizer(args.model_name_or_path, args.allow_model_download) if args.model_name_or_path else None

    errors: list[str] = []
    warnings: list[str] = []
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    record_locations: dict[str, list[str]] = defaultdict(list)
    pair_locations: dict[str, set[str]] = defaultdict(set)
    pair_directions: set[tuple[str, str]] = set()
    direction_counts: dict[str, dict[str, int]] = {}
    truncated_count = 0
    accepted_quality_checks: Counter[str] = Counter()

    required = {
        "record_id", "original_id", "pair_group_id", "direction", "source_lang", "target_lang",
        "source_text", "target_text", "prompt_messages", "completion", "split",
    }
    for split in SPLITS:
        rows: list[dict[str, Any]] = []
        for line_number, record in read_jsonl(data_dir / f"{split}.jsonl"):
            location = f"{split}.jsonl:{line_number}"
            missing = required - set(record)
            if missing:
                errors.append(f"{location}: missing fields {sorted(missing)}")
                continue
            if record["split"] != split:
                errors.append(f"{location}: split field is {record['split']!r}")
            if record["direction"] not in ALLOWED_DIRECTIONS:
                errors.append(f"{location}: invalid direction {record['direction']!r}")
            if not normalize_for_comparison(record["source_text"]):
                accepted_quality_checks["empty_source_text"] += 1
                errors.append(f"{location}: empty source_text")
            if not normalize_for_comparison(record["target_text"]):
                accepted_quality_checks["empty_target_text"] += 1
                errors.append(f"{location}: empty target_text")
            if normalize_for_comparison(record["source_text"]).casefold() == normalize_for_comparison(
                record["target_text"]
            ).casefold():
                accepted_quality_checks["source_equals_target"] += 1
                errors.append(f"{location}: source_text equals target_text")
            expected_pair = canonical_pair_id(
                record["source_lang"], record["target_lang"], record["source_text"], record["target_text"]
            )
            if record["pair_group_id"] != expected_pair:
                errors.append(f"{location}: pair_group_id does not match canonical text")
            source_ok, _ = language_quality(record["source_text"], record["source_lang"])
            target_ok, _ = language_quality(record["target_text"], record["target_lang"])
            if not source_ok:
                accepted_quality_checks["source_language_anomaly"] += 1
                errors.append(f"{location}: severe source language mismatch")
            if not target_ok:
                accepted_quality_checks["target_language_anomaly"] += 1
                errors.append(f"{location}: severe target language mismatch")
            if has_severe_repetition(record["target_text"]):
                accepted_quality_checks["severe_target_repetition"] += 1
                errors.append(f"{location}: accepted target still has severe repetition")
            duplicate_key = (record["pair_group_id"], record["direction"])
            if duplicate_key in pair_directions:
                accepted_quality_checks["duplicate_pair_same_direction"] += 1
                errors.append(f"{location}: duplicate pair in the same direction")
            pair_directions.add(duplicate_key)
            record_locations[record["record_id"]].append(split)
            pair_locations[record["pair_group_id"]].add(split)
            if record.get("prompt_template_version") != PROMPT_TEMPLATE_VERSION:
                errors.append(f"{location}: prompt template version mismatch")
            if record.get("completion") != record.get("target_text"):
                errors.append(f"{location}: completion differs from target_text")
            if tokenizer is not None:
                token_info = completion_token_info(tokenizer, record, args.max_length)
                if int(token_info["valid_completion_tokens"]) <= 0:
                    errors.append(f"{location}: no completion supervision survives truncation")
                if bool(token_info["truncated"]):
                    truncated_count += 1
            rows.append(record)
        records_by_split[split] = rows
        direction_counts[split] = dict(sorted(Counter(row["direction"] for row in rows).items()))

    for record_id, locations in record_locations.items():
        if len(locations) > 1:
            errors.append(f"record_id leakage: {record_id} appears in {locations}")
    for pair_id, locations in pair_locations.items():
        if len(locations) > 1:
            errors.append(f"pair_group leakage: {pair_id} appears in {sorted(locations)}")

    rejected_count = sum(1 for _ in read_jsonl(rejected_path))
    actual_split_counts = {split: len(rows) for split, rows in records_by_split.items()}
    if actual_split_counts != manifest.get("split_counts"):
        errors.append(f"manifest split_counts mismatch: actual={actual_split_counts}")
    if direction_counts != manifest.get("direction_counts"):
        errors.append("manifest direction_counts mismatch")
    if sum(actual_split_counts.values()) != manifest.get("accepted_total"):
        errors.append("manifest accepted_total mismatch")
    if rejected_count != manifest.get("rejected_total"):
        errors.append("manifest rejected_total mismatch")
    if rejected_count != rejection_report.get("rejected_total"):
        errors.append("rejection_report rejected_total mismatch")
    if manifest.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256:
        errors.append("manifest prompt_template_sha256 mismatch")
    if tokenizer is None:
        warnings.append("Tokenizer validation was skipped; rerun on AutoDL with --model-name-or-path")

    report = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "split_counts": actual_split_counts,
        "direction_counts": direction_counts,
        "unique_record_ids": len(record_locations),
        "unique_pair_groups": len(pair_locations),
        "pair_group_leakage_count": sum(len(locations) > 1 for locations in pair_locations.values()),
        "accepted_quality_check_counts": {
            name: accepted_quality_checks.get(name, 0)
            for name in (
                "empty_source_text",
                "empty_target_text",
                "source_equals_target",
                "duplicate_pair_same_direction",
                "severe_target_repetition",
                "source_language_anomaly",
                "target_language_anomaly",
            )
        },
        "rejected_count": rejected_count,
        "tokenizer_validation_performed": tokenizer is not None,
        "truncated_count": truncated_count,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.no_write_report:
        atomic_write_json(data_dir / "validation_report.json", report)
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
