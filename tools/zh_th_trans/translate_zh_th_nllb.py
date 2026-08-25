#!/usr/bin/env python3
"""Continue Chinese-to-Thai translation with a local NLLB-200 model.

Inputs are the native Chinese source records in the three crawled Chinese-
English cleaned JSON files.  By default, source IDs already translated
successfully by Qwen under ``dataset/crawled/zh-th/translations/qwen*`` are
skipped, so NLLB continues from the remaining records rather than duplicating
work.

NLLB progress is stored as atomically written JSONL chunks.  Re-running the
same command skips NLLB successes (and failures unless ``--retry-failed`` is
used).  A failed batch is retried record by record; one bad record never aborts
the complete translation job.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "crawled" / "zh_en" / "cleaned"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "models" / "nllb-200-distilled-600M"
DEFAULT_TRANSLATIONS_ROOT = PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "translations"
DEFAULT_OUTPUT_DIR = DEFAULT_TRANSLATIONS_ROOT / "nllb_600m"
DOMAINS = ("education", "technology", "finance")
SOURCE_LANGUAGE = "zho_Hans"
TARGET_LANGUAGE = "tha_Thai"
MODEL_LABEL = "facebook/nllb-200-distilled-600M"
MODEL_WEIGHT_FORMAT = "safetensors"
THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
HAN_RE = re.compile(
    "[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    "\U00020000-\U0002FA1F\U00030000-\U000323AF]"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue translating crawled Chinese source records into Thai with local NLLB."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prior-translations-dir",
        type=Path,
        default=DEFAULT_TRANSLATIONS_ROOT,
        help="Root scanned for qwen*/accepted.jsonl records to skip.",
    )
    parser.add_argument(
        "--include-qwen-translated",
        action="store_true",
        help="Do not skip source IDs already translated successfully by Qwen.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum records attempted in this invocation; 0 means all remaining records.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry NLLB records whose latest saved event is failed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and resume state without loading the model or writing files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError(f"Expected a JSON object with a records array: {path}")
    return value


def load_sources(input_dir: Path) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    fingerprint = hashlib.sha256()

    for domain in DOMAINS:
        path = input_dir / f"{domain}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Required cleaned source file does not exist: {path}")
        document = read_json(path)
        domain_count = 0
        for index, record in enumerate(document["records"], start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record at {path}:{index}")
            if record.get("source_lang") != "zh-CN":
                continue
            record_id = str(record.get("id", "")).strip()
            source_text = record.get("source_text")
            if not record_id:
                raise ValueError(f"Chinese source record at {path}:{index} has no id")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate source record id: {record_id}")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError(f"Chinese source record {record_id} has empty source_text")
            if record.get("domain") != domain:
                raise ValueError(f"Domain mismatch for {record_id} in {path}")

            source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            provenance = dict(record.get("provenance") or {})
            provenance["source_clean_file"] = relative_path(path)
            provenance["source_clean_record_id"] = record_id
            records.append(
                {
                    "id": record_id,
                    "source_text": source_text,
                    "zh_char_count": record.get("zh_char_count"),
                    "domain": domain,
                    "source_sha256": source_hash,
                    "provenance": provenance,
                }
            )
            seen_ids.add(record_id)
            domain_count += 1
            fingerprint.update(record_id.encode("utf-8"))
            fingerprint.update(b"\x1f")
            fingerprint.update(source_hash.encode("ascii"))
            fingerprint.update(b"\n")
        counts[domain] = domain_count
    return records, fingerprint.hexdigest(), counts


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL record at {path}:{line_number}")
            yield line_number, value


def load_prior_successes(root: Path) -> tuple[set[str], dict[str, int], list[str]]:
    successful_ids: set[str] = set()
    per_file: dict[str, int] = {}
    files: list[str] = []
    if not root.exists():
        return successful_ids, per_file, files
    for path in sorted(root.glob("qwen*/accepted.jsonl")):
        count = 0
        for line_number, record in read_jsonl(path):
            if record.get("status") != "success" or not str(record.get("target_text", "")).strip():
                continue
            record_id = str(record.get("source_record_id", "")).strip()
            if not record_id:
                raise ValueError(f"Qwen success lacks source_record_id at {path}:{line_number}")
            successful_ids.add(record_id)
            count += 1
        rel = relative_path(path)
        files.append(rel)
        per_file[rel] = count
    return successful_ids, per_file, files


def chunk_number(path: Path) -> int:
    match = re.fullmatch(r"chunk_(\d+)\.jsonl", path.name)
    if not match:
        raise ValueError(f"Invalid chunk filename: {path.name}")
    return int(match.group(1))


def load_nllb_events(
    chunks_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str], int]:
    all_events: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    maximum_chunk = 0
    if not chunks_dir.exists():
        return all_events, latest, attempts, maximum_chunk
    for path in sorted(chunks_dir.glob("chunk_*.jsonl"), key=chunk_number):
        maximum_chunk = max(maximum_chunk, chunk_number(path))
        for line_number, event in read_jsonl(path):
            record_id = str(event.get("source_record_id", "")).strip()
            if not record_id or event.get("status") not in {"success", "failed"}:
                raise ValueError(f"Invalid NLLB event at {path}:{line_number}")
            all_events.append(event)
            latest[record_id] = event
            attempts[record_id] += 1
    return all_events, latest, attempts, maximum_chunk


def validate_resume(
    source_ids: set[str], latest: dict[str, dict[str, Any]], checkpoint: Path, fingerprint: str
) -> None:
    unknown = set(latest) - source_ids
    if unknown:
        raise ValueError(f"NLLB progress contains unknown source IDs: {sorted(unknown)[:5]}")
    if checkpoint.is_file():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid checkpoint JSON in {checkpoint}: {error}") from error
        if not isinstance(saved, dict):
            raise ValueError(f"Checkpoint must contain a JSON object: {checkpoint}")
        saved_fingerprint = saved.get("input_fingerprint")
        if saved_fingerprint and saved_fingerprint != fingerprint:
            raise ValueError(
                "Cleaned Chinese inputs changed after NLLB progress was created. Restore the inputs "
                "or use a new --output-dir."
            )


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


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )


def dtype_value(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def import_ml_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Activate an environment containing torch, transformers and accelerate before "
            "running NLLB translation."
        ) from error
    return torch, AutoTokenizer, AutoModelForSeq2SeqLM


def validate_model_weight_compatibility(model_path: Path) -> None:
    safetensors = list(model_path.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(
            f"No safetensors weights found under {model_path}. This script is configured to load "
            "NLLB with use_safetensors=True and will not fall back to pytorch_model.bin."
        )


def load_model(args: argparse.Namespace, logger: logging.Logger) -> tuple[Any, Any, Any, str]:
    torch, auto_tokenizer, auto_model = import_ml_dependencies()
    model_path = resolve_path(args.model_path)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"NLLB model config not found: {model_path / 'config.json'}")
    validate_model_weight_compatibility(model_path)
    if not torch.cuda.is_available() and args.dtype == "float16":
        raise RuntimeError("float16 NLLB inference requires CUDA; use --dtype float32 for CPU.")

    logger.info("Loading NLLB tokenizer from %s", model_path)
    tokenizer = auto_tokenizer.from_pretrained(
        str(model_path), local_files_only=True, src_lang=SOURCE_LANGUAGE
    )
    target_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Tokenizer does not recognize target language token {TARGET_LANGUAGE}")

    compute_dtype = dtype_value(torch, args.dtype)
    logger.info(
        "Loading NLLB model from %s (dtype=%s, device_map=%s)",
        model_path,
        args.dtype,
        args.device_map,
    )
    model = auto_model.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_safetensors=True,
        dtype=compute_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return torch, tokenizer, model, str(model_path)


def model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_batch(
    torch: Any, tokenizer: Any, model: Any, records: list[dict[str, Any]], args: argparse.Namespace
) -> list[str]:
    tokenizer.src_lang = SOURCE_LANGUAGE
    encoded = tokenizer(
        [record["source_text"] for record in records],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    encoded = {key: value.to(model_device(model)) for key, value in encoded.items()}
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            # NLLB is encoder-decoder, so max_length limits decoded output length.
            # Using it avoids conflict with the model's bundled max_length setting.
            max_length=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def output_quality(source: str, target: str) -> tuple[list[str], dict[str, Any]]:
    thai_count = len(THAI_RE.findall(target))
    thai_letter_count = sum(
        "\u0E00" <= character <= "\u0E7F" and character.isalpha() for character in target
    )
    han_count = len(HAN_RE.findall(target))
    alphabetic_count = sum(character.isalpha() for character in target)
    thai_ratio = thai_letter_count / max(alphabetic_count, 1)
    length_ratio = len(target) / max(len(source), 1)
    flags: list[str] = []
    if thai_ratio < 0.60:
        flags.append("low_thai_letter_ratio")
    if han_count:
        flags.append("contains_han_characters")
    if length_ratio < 0.25:
        flags.append("target_much_shorter_than_source")
    elif length_ratio > 4.0:
        flags.append("target_much_longer_than_source")
    return flags, {
        "thai_character_count": thai_count,
        "target_han_character_count": han_count,
        "thai_letter_ratio": round(thai_ratio, 6),
        "target_to_source_codepoint_ratio": round(length_ratio, 6),
    }


def success_event(
    record: dict[str, Any], target_text: str, model_path: str, attempt: int
) -> dict[str, Any]:
    target = CONTROL_RE.sub("", target_text).strip()
    if not target:
        raise ValueError("NLLB returned empty text")
    if target == record["source_text"].strip():
        raise ValueError("NLLB returned unchanged Chinese source text")
    if not THAI_RE.search(target):
        raise ValueError("NLLB output contains no Thai characters")
    flags, metrics = output_quality(record["source_text"], target)
    provenance = dict(record["provenance"])
    provenance["translation_input_id"] = record["id"]
    return {
        "id": f"nllb_zh_th_{record['id']}",
        "source_record_id": record["id"],
        "language_pair": "zh_th",
        "source_lang": "zh-CN",
        "target_lang": "th",
        "source_text": record["source_text"],
        "target_text": target,
        "zh_char_count": record["zh_char_count"],
        "domain": record["domain"],
        "translation_method": "llm_mt",
        "model_name": MODEL_LABEL,
        "model_path": model_path,
        "model_weight_format": MODEL_WEIGHT_FORMAT,
        "nllb_source_language": SOURCE_LANGUAGE,
        "nllb_target_language": TARGET_LANGUAGE,
        "status": "success",
        "attempt": attempt,
        "generated_at": utc_now(),
        "quality_flags": flags,
        "quality_metrics": metrics,
        "provenance": provenance,
    }


def failure_event(
    record: dict[str, Any], error: BaseException, model_path: str, attempt: int
) -> dict[str, Any]:
    return {
        "id": f"nllb_zh_th_failed_{record['id']}_attempt_{attempt}",
        "source_record_id": record["id"],
        "source_lang": "zh-CN",
        "target_lang": "th",
        "source_text": record["source_text"],
        "zh_char_count": record["zh_char_count"],
        "domain": record["domain"],
        "translation_method": "llm_mt",
        "model_name": MODEL_LABEL,
        "model_path": model_path,
        "status": "failed",
        "attempt": attempt,
        "failed_at": utc_now(),
        "error_type": type(error).__name__,
        "error_message": str(error)[:2000],
        "provenance": {
            "translation_input_id": record["id"],
            "source_clean_file": record["provenance"].get("source_clean_file"),
            "source_clean_record_id": record["provenance"].get("source_clean_record_id"),
            "source_url": record["provenance"].get("source_url"),
        },
    }


def clear_cache(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def translate_resilient(
    torch: Any,
    tokenizer: Any,
    model: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    model_path: str,
    attempts: Counter[str],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    try:
        outputs = generate_batch(torch, tokenizer, model, records, args)
        if len(outputs) != len(records):
            raise RuntimeError(f"Expected {len(records)} outputs, received {len(outputs)}")
        events: list[dict[str, Any]] = []
        for record, target in zip(records, outputs, strict=True):
            record_id = record["id"]
            attempt = attempts[record_id] + 1
            try:
                event = success_event(record, target, model_path, attempt)
            except Exception as error:
                event = failure_event(record, error, model_path, attempt)
                logger.warning("Record %s failed validation: %s", record_id, error)
            attempts[record_id] = attempt
            events.append(event)
        return events
    except Exception as batch_error:
        clear_cache(torch)
        if len(records) == 1:
            record = records[0]
            record_id = record["id"]
            attempt = attempts[record_id] + 1
            attempts[record_id] = attempt
            logger.exception("Record %s failed", record_id)
            return [failure_event(record, batch_error, model_path, attempt)]
        logger.warning("Batch of %d failed (%s); retrying singly", len(records), batch_error)
        events = []
        for record in records:
            events.extend(
                translate_resilient(
                    torch, tokenizer, model, [record], args, model_path, attempts, logger
                )
            )
        return events


def latest_counts(latest: dict[str, dict[str, Any]]) -> Counter[str]:
    return Counter(str(event.get("status")) for event in latest.values())


def domain_counts(latest: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    for event in latest.values():
        result.setdefault(str(event.get("domain")), Counter())[str(event.get("status"))] += 1
    return {
        domain: {"success": values["success"], "failed": values["failed"]}
        for domain, values in result.items()
    }


def build_report(
    *,
    input_dir: Path,
    input_counts: dict[str, int],
    fingerprint: str,
    prior_ids: set[str],
    prior_files: list[str],
    prior_file_counts: dict[str, int],
    latest: dict[str, dict[str, Any]],
    all_events: list[dict[str, Any]],
    args: argparse.Namespace,
    model_path: str,
    started_at: str,
    elapsed: float,
    run_attempted: int,
    stage: str,
) -> dict[str, Any]:
    total = sum(input_counts.values())
    counts = latest_counts(latest)
    covered = prior_ids | {record_id for record_id, event in latest.items() if event["status"] == "success"}
    failures = Counter(
        f"{event.get('error_type')}: {event.get('error_message')}"
        for event in latest.values()
        if event.get("status") == "failed"
    )
    quality_flags = Counter(
        flag
        for event in latest.values()
        if event.get("status") == "success"
        for flag in event.get("quality_flags", [])
    )
    return {
        "schema_version": 1,
        "stage": stage,
        "generated_at": utc_now(),
        "run_started_at": started_at,
        "elapsed_seconds": round(elapsed, 3),
        "input_directory": relative_path(input_dir),
        "input_fingerprint": fingerprint,
        "input_records_by_domain": input_counts,
        "model_name": MODEL_LABEL,
        "model_path": model_path,
        "model_weight_format": MODEL_WEIGHT_FORMAT,
        "language_codes": {"source": SOURCE_LANGUAGE, "target": TARGET_LANGUAGE},
        "settings": {
            "batch_size": args.batch_size,
            "save_every": args.save_every,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "skip_qwen_successes": not args.include_qwen_translated,
            "retry_failed": args.retry_failed,
        },
        "prior_successful_translations": {
            "unique_source_ids_skipped": len(prior_ids),
            "files": prior_files,
            "records_read_by_file": prior_file_counts,
        },
        "summary": {
            "total_chinese_source_records": total,
            "nllb_successful_records": counts["success"],
            "nllb_failed_records": counts["failed"],
            "nllb_processed_unique_records": len(latest),
            "nllb_events_all_attempts": len(all_events),
            "attempted_in_this_run": run_attempted,
            "covered_by_qwen_or_nllb_success": len(covered),
            "remaining_without_success": total - len(covered),
        },
        "nllb_by_domain": domain_counts(latest),
        "quality_flag_counts": dict(quality_flags.most_common()),
        "latest_failure_reasons": dict(failures.most_common()),
    }


def write_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    total: int,
    prior_ids: set[str],
    latest: dict[str, dict[str, Any]],
    chunk_number_value: int,
    model_path: str,
) -> None:
    counts = latest_counts(latest)
    nllb_successes = {item for item, event in latest.items() if event["status"] == "success"}
    covered = prior_ids | nllb_successes
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "input_fingerprint": fingerprint,
            "total_input_records": total,
            "prior_successful_source_ids_skipped": len(prior_ids),
            "nllb_processed_unique_records": len(latest),
            "nllb_successful_records": counts["success"],
            "nllb_failed_records": counts["failed"],
            "covered_by_qwen_or_nllb_success": len(covered),
            "remaining_without_success": total - len(covered),
            "last_chunk_number": chunk_number_value,
            "model_name": MODEL_LABEL,
            "model_path": model_path,
            "model_weight_format": MODEL_WEIGHT_FORMAT,
            "source_language": SOURCE_LANGUAGE,
            "target_language": TARGET_LANGUAGE,
        },
    )


def consolidate(output_dir: Path, latest: dict[str, dict[str, Any]], order: list[str]) -> None:
    accepted = [latest[item] for item in order if latest.get(item, {}).get("status") == "success"]
    rejected = [latest[item] for item in order if latest.get(item, {}).get("status") == "failed"]
    atomic_write_jsonl(output_dir / "accepted.jsonl", accepted)
    atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("translate_zh_th_nllb")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        logs_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "save_every", "max_input_tokens", "max_new_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_records < 0:
        raise ValueError("--max-records cannot be negative")


def print_plan(
    input_counts: dict[str, int], prior_count: int, latest: dict[str, dict[str, Any]],
    pending_count: int, planned: int, args: argparse.Namespace, output_dir: Path
) -> None:
    counts = latest_counts(latest)
    print("Chinese source records:")
    for domain in DOMAINS:
        print(f"  {domain}: {input_counts[domain]:,}")
    print(f"  total: {sum(input_counts.values()):,}")
    print(f"Qwen successful source IDs skipped: {prior_count:,}")
    print(f"NLLB already successful: {counts['success']:,}")
    print(f"NLLB already failed: {counts['failed']:,}")
    print(f"pending under current policy: {pending_count:,}")
    print(f"planned this run: {planned:,}")
    print(f"batch size: {args.batch_size}; save every: {args.save_every}")
    print(f"output directory: {output_dir}")


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    prior_root = resolve_path(args.prior_translations_dir)
    chunks_dir = output_dir / "chunks"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "translation_report.json"
    started_at = utc_now()
    started_clock = time.monotonic()

    try:
        validate_args(args)
        sources, fingerprint, input_counts = load_sources(input_dir)
        source_ids = {record["id"] for record in sources}
        if args.include_qwen_translated:
            prior_ids, prior_file_counts, prior_files = set(), {}, []
        else:
            prior_ids, prior_file_counts, prior_files = load_prior_successes(prior_root)
            prior_ids &= source_ids
        all_events, latest, attempts, maximum_chunk = load_nllb_events(chunks_dir)
        validate_resume(source_ids, latest, checkpoint_path, fingerprint)

        pending: list[dict[str, Any]] = []
        for record in sources:
            record_id = record["id"]
            if record_id in prior_ids:
                continue
            previous = latest.get(record_id)
            if previous is None or (args.retry_failed and previous["status"] == "failed"):
                pending.append(record)
        planned = len(pending) if not args.max_records else min(len(pending), args.max_records)
        print_plan(input_counts, len(prior_ids), latest, len(pending), planned, args, output_dir)
        if args.dry_run:
            print("dry-run: model was not loaded and no files were written")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logging(output_dir)
        if args.max_records:
            pending = pending[: args.max_records]
        model_path_for_report = str(resolve_path(args.model_path))

        if not pending:
            consolidate(output_dir, latest, [record["id"] for record in sources])
            report = build_report(
                input_dir=input_dir, input_counts=input_counts, fingerprint=fingerprint,
                prior_ids=prior_ids, prior_files=prior_files, prior_file_counts=prior_file_counts,
                latest=latest, all_events=all_events, args=args, model_path=model_path_for_report,
                started_at=started_at, elapsed=time.monotonic() - started_clock,
                run_attempted=0, stage="completed",
            )
            atomic_write_json(report_path, report)
            logger.info("No records require NLLB translation")
            return 0

        torch, tokenizer, model, model_path_for_report = load_model(args, logger)
        buffer: list[dict[str, Any]] = []
        run_attempted = 0

        def save_progress(stage: str = "running") -> None:
            nonlocal buffer, maximum_chunk, all_events
            if buffer:
                maximum_chunk += 1
                atomic_write_jsonl(chunks_dir / f"chunk_{maximum_chunk:06d}.jsonl", buffer)
                all_events.extend(buffer)
                for event in buffer:
                    latest[event["source_record_id"]] = event
                buffer = []
            write_checkpoint(
                checkpoint_path, fingerprint=fingerprint, total=len(sources), prior_ids=prior_ids,
                latest=latest, chunk_number_value=maximum_chunk, model_path=model_path_for_report,
            )
            report = build_report(
                input_dir=input_dir, input_counts=input_counts, fingerprint=fingerprint,
                prior_ids=prior_ids, prior_files=prior_files, prior_file_counts=prior_file_counts,
                latest=latest, all_events=all_events, args=args, model_path=model_path_for_report,
                started_at=started_at, elapsed=time.monotonic() - started_clock,
                run_attempted=run_attempted, stage=stage,
            )
            atomic_write_json(report_path, report)
            counts = latest_counts(latest)
            logger.info(
                "Saved: NLLB processed=%d success=%d failed=%d; total covered=%d/%d",
                len(latest), counts["success"], counts["failed"],
                len(prior_ids | {item for item, event in latest.items() if event["status"] == "success"}),
                len(sources),
            )

        try:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                events = translate_resilient(
                    torch, tokenizer, model, batch, args, model_path_for_report, attempts, logger
                )
                buffer.extend(events)
                run_attempted += len(events)
                if len(buffer) >= args.save_every:
                    save_progress()
        except KeyboardInterrupt:
            logger.warning("Interrupted; saving completed in-memory translations")
            save_progress("interrupted")
            consolidate(output_dir, latest, [record["id"] for record in sources])
            logger.info("Progress saved. Run the same command to continue.")
            return 130

        save_progress("partial" if args.max_records and pending else "completed")
        consolidate(output_dir, latest, [record["id"] for record in sources])
        logger.info("NLLB invocation finished; report: %s", report_path)
        return 0
    except Exception as error:
        print(f"NLLB Chinese-Thai translation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
