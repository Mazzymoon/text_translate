#!/usr/bin/env python3
"""Batch-translate the extracted Chinese corpus into Thai with local Qwen3.

The long-running job is resumable.  Translation events are saved as immutable,
atomically-written JSONL chunks.  On restart, the script scans those chunks and
skips completed source IDs.  A failed batch is retried record by record, so one
bad record does not terminate the whole run.
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
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "zh_source.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "dataset" / "crawled" / "zh-th" / "translations" / "qwen3_4b_4bit"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "models"
    / "Qwen--Qwen3-4B-Instruct-2507"
    / "snapshots"
    / "master"
)
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT_VERSION = "zh_th_translation_v1"
QUANTIZATION_METHOD = "bitsandbytes_4bit_nf4"
THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
HAN_RE = re.compile(
    "[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    "\U00020000-\U0002FA1F\U00030000-\U000323AF]"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PREFIX_RE = re.compile(
    r"^(?:泰语译文|泰文译文|翻译|译文|คำแปลภาษาไทย|คำแปล)\s*[:：]\s*",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
FENCE_RE = re.compile(r"^```(?:thai|th)?\s*\n?(.*?)\n?```$", re.IGNORECASE | re.DOTALL)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate the extracted zh-CN candidates into Thai with local Qwen3."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Hugging Face/ModelScope model snapshot directory.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts passed to model.generate at once (default: 4).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Atomically save a progress chunk after this many attempted records (default: 50).",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum records attempted in this invocation; 0 means all pending records.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="4-bit computation dtype; model weights remain 4-bit (default: float16).",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry IDs whose latest saved event is failed; successful IDs are always skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and resume state without loading the model or writing files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, text)


def load_input(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid input JSON in {path}: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError("Input must be a JSON object containing a records array")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    fingerprint = hashlib.sha256()
    for index, record in enumerate(document["records"], start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Input record {index} is not an object")
        record_id = str(record.get("id", "")).strip()
        source_text = record.get("source_text")
        if not record_id:
            raise ValueError(f"Input record {index} has no id")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate input id: {record_id}")
        if record.get("source_lang") != "zh-CN" or record.get("target_lang") != "th":
            raise ValueError(f"Unexpected language direction for {record_id}")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"Empty Chinese source text for {record_id}")
        seen_ids.add(record_id)
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        fingerprint.update(record_id.encode("utf-8"))
        fingerprint.update(b"\x1f")
        fingerprint.update(source_hash.encode("ascii"))
        fingerprint.update(b"\n")
        records.append(record)
    return records, fingerprint.hexdigest()


def chunk_number(path: Path) -> int:
    match = re.fullmatch(r"chunk_(\d+)\.jsonl", path.name)
    if not match:
        raise ValueError(f"Invalid progress chunk filename: {path.name}")
    return int(match.group(1))


def load_events(
    chunks_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str], int]:
    events: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    attempt_counts: Counter[str] = Counter()
    maximum_chunk = 0
    if not chunks_dir.exists():
        return events, latest, attempt_counts, maximum_chunk

    for path in sorted(chunks_dir.glob("chunk_*.jsonl"), key=chunk_number):
        maximum_chunk = max(maximum_chunk, chunk_number(path))
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
                if not isinstance(event, dict) or not str(event.get("source_record_id", "")):
                    raise ValueError(f"Invalid translation event at {path}:{line_number}")
                record_id = str(event["source_record_id"])
                status = event.get("status")
                if status not in {"success", "failed"}:
                    raise ValueError(f"Invalid event status at {path}:{line_number}: {status!r}")
                events.append(event)
                latest[record_id] = event
                attempt_counts[record_id] += 1
    return events, latest, attempt_counts, maximum_chunk


def validate_resume_state(
    records: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
    checkpoint_path: Path,
    fingerprint: str,
) -> None:
    input_ids = {str(record["id"]) for record in records}
    unknown_ids = set(latest) - input_ids
    if unknown_ids:
        raise ValueError(f"Saved chunks contain IDs absent from current input: {sorted(unknown_ids)[:5]}")
    if checkpoint_path.is_file():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid checkpoint JSON: {error}") from error
        saved_fingerprint = checkpoint.get("input_fingerprint")
        if saved_fingerprint and saved_fingerprint != fingerprint:
            raise ValueError(
                "Input candidates changed since the saved checkpoint. Use a new output directory "
                "or restore the original input before resuming."
            )


def build_prompt(source_text: str) -> str:
    return f"""请将下面的中文准确翻译成自然、流畅的泰语。

要求：
1. 忠实保留原文全部含义，不得遗漏或增加信息。
2. 数字、日期、单位和专有名词应准确保留。
3. 保持原文语气和段落关系。
4. 只输出泰语译文，不要解释，不要添加标题或“翻译如下”等内容。

中文原文：
{source_text}"""


def clean_model_output(value: str) -> tuple[str, list[str]]:
    text = CONTROL_RE.sub("", str(value)).strip()
    flags: list[str] = []
    without_thinking = THINK_RE.sub("", text).strip()
    if without_thinking != text:
        flags.append("removed_thinking_block")
    text = without_thinking
    fence_match = FENCE_RE.fullmatch(text)
    if fence_match:
        text = fence_match.group(1).strip()
        flags.append("removed_markdown_fence")
    without_prefix = PREFIX_RE.sub("", text, count=1).strip()
    if without_prefix != text:
        flags.append("removed_translation_prefix")
    return without_prefix, flags


def output_quality(source_text: str, target_text: str) -> tuple[list[str], dict[str, Any]]:
    thai_count = len(THAI_RE.findall(target_text))
    han_count = len(HAN_RE.findall(target_text))
    visible_letters = sum(character.isalpha() for character in target_text)
    thai_ratio = thai_count / max(visible_letters, 1)
    length_ratio = len(target_text) / max(len(source_text), 1)
    flags: list[str] = []
    if thai_ratio < 0.60:
        flags.append("low_thai_letter_ratio")
    if han_count:
        flags.append("contains_han_characters")
    if length_ratio < 0.25:
        flags.append("target_much_shorter_than_source")
    elif length_ratio > 4.0:
        flags.append("target_much_longer_than_source")
    metrics = {
        "thai_character_count": thai_count,
        "target_han_character_count": han_count,
        "thai_letter_ratio": round(thai_ratio, 6),
        "target_to_source_codepoint_ratio": round(length_ratio, 6),
    }
    return flags, metrics


def successful_event(
    record: dict[str, Any], raw_output: str, model_name: str, attempt: int
) -> dict[str, Any]:
    target_text, cleanup_flags = clean_model_output(raw_output)
    if not target_text:
        raise ValueError("model returned empty text")
    if target_text == record["source_text"].strip():
        raise ValueError("model returned the unchanged Chinese source text")
    if not THAI_RE.search(target_text):
        raise ValueError("model output contains no Thai characters")
    quality_flags, metrics = output_quality(record["source_text"], target_text)
    quality_flags = list(dict.fromkeys(cleanup_flags + quality_flags))
    provenance = dict(record.get("provenance") or {})
    provenance["translation_input_id"] = record["id"]
    return {
        "id": f"qwen3_zh_th_{record['id']}",
        "source_record_id": record["id"],
        "language_pair": "zh_th",
        "source_lang": "zh-CN",
        "target_lang": "th",
        "source_text": record["source_text"],
        "target_text": target_text,
        "zh_char_count": record.get("zh_char_count"),
        "domain": record.get("domain"),
        "translation_method": "llm_mt",
        "model_name": model_name,
        "model_quantization": QUANTIZATION_METHOD,
        "prompt_version": PROMPT_VERSION,
        "status": "success",
        "attempt": attempt,
        "generated_at": utc_now(),
        "quality_flags": quality_flags,
        "quality_metrics": metrics,
        "provenance": provenance,
    }


def failed_event(
    record: dict[str, Any], error: BaseException, model_name: str, attempt: int
) -> dict[str, Any]:
    return {
        "id": f"qwen3_zh_th_failed_{record['id']}_attempt_{attempt}",
        "source_record_id": record["id"],
        "source_lang": "zh-CN",
        "target_lang": "th",
        "source_text": record["source_text"],
        "zh_char_count": record.get("zh_char_count"),
        "domain": record.get("domain"),
        "translation_method": "llm_mt",
        "model_name": model_name,
        "model_quantization": QUANTIZATION_METHOD,
        "prompt_version": PROMPT_VERSION,
        "status": "failed",
        "attempt": attempt,
        "failed_at": utc_now(),
        "error_type": type(error).__name__,
        "error_message": str(error)[:2000],
        "provenance": {
            "translation_input_id": record["id"],
            "source_clean_file": (record.get("provenance") or {}).get("source_clean_file"),
            "source_clean_record_id": (record.get("provenance") or {}).get(
                "source_clean_record_id"
            ),
            "source_url": (record.get("provenance") or {}).get("source_url"),
        },
    }


def import_ml_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import bitsandbytes  # noqa: F401 - validates that the 4-bit backend is installed
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Missing local Qwen runtime dependency. Activate the environment that contains "
            "torch, transformers, accelerate and bitsandbytes before running this 4-bit script."
        ) from error
    return torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, gc


def dtype_value(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace, logger: logging.Logger) -> tuple[Any, Any, Any, str]:
    torch, auto_tokenizer, auto_model, bits_and_bytes_config, _ = import_ml_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bitsandbytes 4-bit Qwen inference requires a supported accelerator, but CUDA is "
            "not available in the active Python environment."
        )
    model_path = resolve_path(args.model_path)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Local model snapshot not found at {model_path}. Pass --model-path with the "
            "directory containing config.json and model safetensors."
        )
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = auto_tokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = dtype_value(torch, args.dtype)
    quantization_config = bits_and_bytes_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    logger.info(
        "Loading model from %s (4-bit NF4, double_quant=true, compute_dtype=%s, device_map=%s)",
        model_path,
        args.dtype,
        args.device_map,
    )
    model = auto_model.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return torch, tokenizer, model, str(model_path)


def chat_prompt(tokenizer: Any, source_text: str) -> str:
    messages = [{"role": "user", "content": build_prompt(source_text)}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def model_input_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_batch(
    torch: Any,
    tokenizer: Any,
    model: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    prompts = [chat_prompt(tokenizer, record["source_text"]) for record in records]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_width = inputs["input_ids"].shape[1]
    inputs = {key: value.to(model_input_device(model)) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[:, input_width:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def clear_accelerator_cache(torch: Any) -> None:
    gc.collect()
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def translate_resilient(
    torch: Any,
    tokenizer: Any,
    model: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    model_name: str,
    attempt_counts: Counter[str],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    try:
        outputs = generate_batch(torch, tokenizer, model, records, args)
        if len(outputs) != len(records):
            raise RuntimeError(
                f"Batch output count mismatch: expected {len(records)}, got {len(outputs)}"
            )
        events: list[dict[str, Any]] = []
        for record, output in zip(records, outputs, strict=True):
            record_id = str(record["id"])
            attempt = attempt_counts[record_id] + 1
            try:
                event = successful_event(record, output, model_name, attempt)
            except Exception as error:  # validation failure is isolated to this record
                event = failed_event(record, error, model_name, attempt)
                logger.warning("Record %s failed output validation: %s", record_id, error)
            attempt_counts[record_id] = attempt
            events.append(event)
        return events
    except Exception as batch_error:
        clear_accelerator_cache(torch)
        if len(records) == 1:
            record = records[0]
            record_id = str(record["id"])
            attempt = attempt_counts[record_id] + 1
            attempt_counts[record_id] = attempt
            logger.exception("Record %s generation failed", record_id)
            return [failed_event(record, batch_error, model_name, attempt)]

        logger.warning(
            "Batch of %d failed (%s); retrying each record individually",
            len(records),
            batch_error,
        )
        events = []
        for record in records:
            events.extend(
                translate_resilient(
                    torch,
                    tokenizer,
                    model,
                    [record],
                    args,
                    model_name,
                    attempt_counts,
                    logger,
                )
            )
        return events


def status_counts(latest: dict[str, dict[str, Any]]) -> Counter[str]:
    return Counter(str(event.get("status")) for event in latest.values())


def domain_status_counts(latest: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    domains: dict[str, Counter[str]] = {}
    for event in latest.values():
        domain = str(event.get("domain") or "unknown")
        domains.setdefault(domain, Counter())[str(event.get("status"))] += 1
    return {
        domain: {
            "success": counts["success"],
            "failed": counts["failed"],
            "processed": counts["success"] + counts["failed"],
        }
        for domain, counts in sorted(domains.items())
    }


def create_report(
    *,
    input_path: Path,
    fingerprint: str,
    total_input: int,
    latest: dict[str, dict[str, Any]],
    all_events: list[dict[str, Any]],
    model_name: str,
    args: argparse.Namespace,
    started_at: str,
    elapsed_seconds: float,
    run_attempted: int,
    stage: str,
) -> dict[str, Any]:
    counts = status_counts(latest)
    processed = len(latest)
    failure_reasons = Counter(
        f"{event.get('error_type', 'Error')}: {event.get('error_message', '')}"
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
        "elapsed_seconds": round(elapsed_seconds, 3),
        "input_file": input_path.as_posix(),
        "input_fingerprint": fingerprint,
        "model_name": model_name,
        "model_quantization": QUANTIZATION_METHOD,
        "prompt_version": PROMPT_VERSION,
        "settings": {
            "batch_size": args.batch_size,
            "save_every": args.save_every,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": args.dtype,
            "device_map": args.device_map,
            "retry_failed": args.retry_failed,
        },
        "summary": {
            "total_input_records": total_input,
            "latest_successful_records": counts["success"],
            "latest_failed_records": counts["failed"],
            "processed_unique_records": processed,
            "pending_records": total_input - processed,
            "translation_events_all_attempts": len(all_events),
            "attempted_in_this_run": run_attempted,
        },
        "by_domain": domain_status_counts(latest),
        "quality_flag_counts": dict(quality_flags.most_common()),
        "latest_failure_reasons": dict(failure_reasons.most_common()),
    }


def write_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    total_input: int,
    latest: dict[str, dict[str, Any]],
    maximum_chunk: int,
    model_name: str,
) -> None:
    counts = status_counts(latest)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "input_fingerprint": fingerprint,
            "total_input_records": total_input,
            "processed_unique_records": len(latest),
            "successful_records": counts["success"],
            "failed_records": counts["failed"],
            "pending_records": total_input - len(latest),
            "last_chunk_number": maximum_chunk,
            "model_name": model_name,
            "model_quantization": QUANTIZATION_METHOD,
            "prompt_version": PROMPT_VERSION,
        },
    )


def consolidate_outputs(
    output_dir: Path, latest: dict[str, dict[str, Any]], input_order: list[str]
) -> None:
    accepted = [latest[item] for item in input_order if latest.get(item, {}).get("status") == "success"]
    rejected = [latest[item] for item in input_order if latest.get(item, {}).get("status") == "failed"]
    atomic_write_jsonl(output_dir / "accepted.jsonl", accepted)
    atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)


def configure_logging(output_dir: Path, dry_run: bool) -> logging.Logger:
    logger = logging.getLogger("translate_zh_th_qwen3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if not dry_run:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
        file_handler = logging.FileHandler(logs_dir / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.max_records < 0:
        raise ValueError("--max-records cannot be negative")


def print_plan(
    total: int,
    latest: dict[str, dict[str, Any]],
    pending: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    counts = status_counts(latest)
    planned = len(pending) if args.max_records == 0 else min(len(pending), args.max_records)
    print(f"input records: {total:,}")
    print(f"already successful: {counts['success']:,}")
    print(f"already failed: {counts['failed']:,}")
    print(f"pending under current retry policy: {len(pending):,}")
    print(f"planned this run: {planned:,}")
    print(f"batch size: {args.batch_size}; save every: {args.save_every}")
    print(f"output directory: {output_dir}")


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "translation_report.json"
    started_at = utc_now()
    started_clock = time.monotonic()

    try:
        validate_args(args)
        records, fingerprint = load_input(input_path)
        all_events, latest, attempt_counts, maximum_chunk = load_events(chunks_dir)
        validate_resume_state(records, latest, checkpoint_path, fingerprint)

        pending = []
        for record in records:
            previous = latest.get(str(record["id"]))
            if previous is None or (args.retry_failed and previous.get("status") == "failed"):
                pending.append(record)
        print_plan(len(records), latest, pending, args, output_dir)
        if args.dry_run:
            print("dry-run: model was not loaded and no files were written")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logging(output_dir, dry_run=False)
        if args.max_records:
            pending = pending[: args.max_records]
        if not pending:
            logger.info("No records need translation under the current retry policy")
            consolidate_outputs(output_dir, latest, [str(record["id"]) for record in records])
            report = create_report(
                input_path=input_path,
                fingerprint=fingerprint,
                total_input=len(records),
                latest=latest,
                all_events=all_events,
                model_name=args.model_id,
                args=args,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started_clock,
                run_attempted=0,
                stage="completed" if len(latest) == len(records) else "partial",
            )
            atomic_write_json(report_path, report)
            return 0

        torch, tokenizer, model, model_name = load_model(args, logger)
        buffer: list[dict[str, Any]] = []
        run_attempted = 0

        def save_progress() -> None:
            nonlocal buffer, maximum_chunk, all_events
            if not buffer:
                return
            maximum_chunk += 1
            chunk_path = chunks_dir / f"chunk_{maximum_chunk:06d}.jsonl"
            atomic_write_jsonl(chunk_path, buffer)
            all_events.extend(buffer)
            for event in buffer:
                latest[str(event["source_record_id"])] = event
            buffer = []
            write_checkpoint(
                checkpoint_path,
                fingerprint=fingerprint,
                total_input=len(records),
                latest=latest,
                maximum_chunk=maximum_chunk,
                model_name=model_name,
            )
            interim_report = create_report(
                input_path=input_path,
                fingerprint=fingerprint,
                total_input=len(records),
                latest=latest,
                all_events=all_events,
                model_name=model_name,
                args=args,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started_clock,
                run_attempted=run_attempted,
                stage="running",
            )
            atomic_write_json(report_path, interim_report)
            counts = status_counts(latest)
            logger.info(
                "Saved chunk %06d: processed=%d/%d success=%d failed=%d",
                maximum_chunk,
                len(latest),
                len(records),
                counts["success"],
                counts["failed"],
            )

        try:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                events = translate_resilient(
                    torch,
                    tokenizer,
                    model,
                    batch,
                    args,
                    model_name,
                    attempt_counts,
                    logger,
                )
                buffer.extend(events)
                run_attempted += len(events)
                if len(buffer) >= args.save_every:
                    save_progress()
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received; saving completed in-memory records before exit")
            save_progress()
            consolidate_outputs(output_dir, latest, [str(record["id"]) for record in records])
            report = create_report(
                input_path=input_path,
                fingerprint=fingerprint,
                total_input=len(records),
                latest=latest,
                all_events=all_events,
                model_name=model_name,
                args=args,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started_clock,
                run_attempted=run_attempted,
                stage="interrupted",
            )
            atomic_write_json(report_path, report)
            logger.info("Progress saved. Run the same command to resume.")
            return 130

        save_progress()
        consolidate_outputs(output_dir, latest, [str(record["id"]) for record in records])
        stage = "completed" if len(latest) == len(records) else "partial"
        report = create_report(
            input_path=input_path,
            fingerprint=fingerprint,
            total_input=len(records),
            latest=latest,
            all_events=all_events,
            model_name=model_name,
            args=args,
            started_at=started_at,
            elapsed_seconds=time.monotonic() - started_clock,
            run_attempted=run_attempted,
            stage=stage,
        )
        atomic_write_json(report_path, report)
        write_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            total_input=len(records),
            latest=latest,
            maximum_chunk=maximum_chunk,
            model_name=model_name,
        )
        counts = status_counts(latest)
        logger.info(
            "Run finished: stage=%s success=%d failed=%d pending=%d report=%s",
            stage,
            counts["success"],
            counts["failed"],
            len(records) - len(latest),
            report_path,
        )
        return 0
    except Exception as error:
        print(f"Qwen3 Chinese-Thai translation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
