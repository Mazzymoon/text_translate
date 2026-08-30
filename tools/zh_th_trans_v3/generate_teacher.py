#!/usr/bin/env python3
"""Generate Chinese-to-Thai teacher translations with local BF16 Qwen3-8B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    from .common import (
        DEFAULT_CANDIDATES,
        DEFAULT_CONFIG,
        DEFAULT_RAW_GENERATIONS,
        PIPELINE_VERSION,
        PROMPT_TEMPLATE_SHA256,
        PROMPT_TEMPLATE_VERSION,
        append_jsonl_rows,
        build_teacher_messages,
        count_thai,
        load_completed_candidate_ids,
        load_config,
        read_jsonl,
        resolve_path,
        utc_now,
    )
except ImportError:
    from common import (
        DEFAULT_CANDIDATES,
        DEFAULT_CONFIG,
        DEFAULT_RAW_GENERATIONS,
        PIPELINE_VERSION,
        PROMPT_TEMPLATE_SHA256,
        PROMPT_TEMPLATE_VERSION,
        append_jsonl_rows,
        build_teacher_messages,
        count_thai,
        load_completed_candidate_ids,
        load_config,
        read_jsonl,
        resolve_path,
        utc_now,
    )


REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "original_id",
    "domain",
    "source_lang",
    "target_lang",
    "source_text",
    "zh_char_count",
    "source_file",
    "source_row",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_RAW_GENERATIONS)
    parser.add_argument("--errors-file", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all pending records")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_input(path: Path) -> tuple[int, set[str]]:
    seen: set[str] = set()
    total = 0
    for line_number, row in read_jsonl(path):
        missing = REQUIRED_CANDIDATE_FIELDS - set(row)
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        candidate_id = str(row["candidate_id"]).strip()
        if not candidate_id:
            raise ValueError(f"{path}:{line_number} has empty candidate_id")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id {candidate_id!r} in input")
        if row["source_lang"] != "zh-CN" or row["target_lang"] != "th":
            raise ValueError(f"{path}:{line_number} is not zh-CN->th")
        seen.add(candidate_id)
        total += 1
    return total, seen


def pending_batches(
    path: Path,
    completed_ids: set[str],
    batch_size: int,
    max_records: int,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    yielded = 0
    for _, row in read_jsonl(path):
        if row["candidate_id"] in completed_ids:
            continue
        if max_records and yielded >= max_records:
            break
        batch.append(row)
        yielded += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def serializable_token_id(value: Any) -> int | list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return int(value)


def repair_trailing_partial_jsonl(path: Path) -> bool:
    """Discard only an unterminated last line left by a hard interruption."""

    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False
        position = handle.tell()
        truncate_at = 0
        chunk_size = 64 * 1024
        while position > 0:
            start = max(0, position - chunk_size)
            handle.seek(start)
            chunk = handle.read(position - start)
            offset = chunk.rfind(b"\n")
            if offset >= 0:
                truncate_at = start + offset + 1
                break
            position = start
        handle.truncate(truncate_at)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def generation_metadata(
    *, max_input_length: int, max_new_tokens: int, eos_token_id: Any, pad_token_id: int
) -> dict[str, Any]:
    return {
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "dtype": "bfloat16",
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": serializable_token_id(eos_token_id),
        "pad_token_id": int(pad_token_id),
    }


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0 or args.max_records < 0 or args.max_consecutive_errors <= 0:
        raise ValueError("batch-size/max-consecutive-errors must be positive; max-records cannot be negative")

    input_path = resolve_path(args.input_file)
    output_path = resolve_path(args.output_file)
    errors_path = resolve_path(args.errors_file) if args.errors_file else output_path.with_name(
        "raw_teacher_generation_errors.jsonl"
    )
    config = load_config(resolve_path(args.config))
    generation = config["generation"]
    max_input_length = int(args.max_input_length or generation["max_input_length"])
    max_new_tokens = int(args.max_new_tokens or generation["max_new_tokens"])
    if max_input_length <= 0 or max_new_tokens <= 0:
        raise ValueError("max-input-length and max-new-tokens must be positive")
    total_input, input_ids = validate_input(input_path)

    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
            errors_path.unlink(missing_ok=True)
        elif not args.resume:
            raise FileExistsError("Output exists; pass --resume or --overwrite explicitly")
    elif args.overwrite:
        errors_path.unlink(missing_ok=True)
    if repair_trailing_partial_jsonl(output_path):
        print(f"Removed an interrupted partial final JSONL line from {output_path}")
    completed_ids = load_completed_candidate_ids(output_path)
    unknown_ids = completed_ids - input_ids
    if unknown_ids:
        raise ValueError(f"Output contains candidate IDs absent from input: {len(unknown_ids)}")
    pending_total = total_input - len(completed_ids)
    planned = min(pending_total, args.max_records) if args.max_records else pending_total
    print(
        json.dumps(
            {
                "input_records": total_input,
                "already_completed": len(completed_ids),
                "pending": pending_total,
                "planned_this_run": planned,
                "batch_size": args.batch_size,
                "model_name_or_path": args.model_name_or_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if planned == 0:
        print("Nothing to generate.")
        return 0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install torch and transformers on the inference server") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-8B teacher generation")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16; no silent FP16 fallback is allowed")
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_model_download,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=not args.allow_model_download,
        trust_remote_code=False,
    )
    model.eval()
    device_map = getattr(model, "hf_device_map", None) or {"": str(next(model.parameters()).device)}
    if any(str(device).casefold() in {"cpu", "disk", "meta"} for device in device_map.values()):
        raise RuntimeError(f"CPU/disk/meta offload is not allowed: {device_map}")

    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("Neither model generation_config nor tokenizer defines eos_token_id")
    pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        fallback_eos = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
        pad_token_id = int(fallback_eos)
    metadata = generation_metadata(
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=int(pad_token_id),
    )
    generation_args = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "pad_token_id": int(pad_token_id),
        "use_cache": True,
    }

    def generate_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [
            tokenizer.apply_chat_template(
                build_teacher_messages(row["source_text"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        ).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_args)
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        translations = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        created_at = utc_now()
        output: list[dict[str, Any]] = []
        for row, translation in zip(batch, translations):
            target_text = translation.strip()
            output.append(
                {
                    **row,
                    "source_lang": "zh-CN",
                    "target_lang": "th",
                    "target_text": target_text,
                    "thai_char_count": count_thai(target_text),
                    "teacher_model": args.model_name_or_path,
                    "translation_method": "qwen3_8b_teacher",
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                    "generation_config": metadata,
                    "generated_at": created_at,
                    "pipeline_version": PIPELINE_VERSION,
                }
            )
        return output

    generated_this_run = 0
    failed_this_run = 0
    consecutive_errors = 0
    for batch in pending_batches(input_path, completed_ids, args.batch_size, args.max_records):
        try:
            output_rows = generate_batch(batch)
            append_jsonl_rows(output_path, output_rows)
            completed_ids.update(row["candidate_id"] for row in output_rows)
            generated_this_run += len(output_rows)
            consecutive_errors = 0
        except Exception as batch_error:
            if len(batch) == 1:
                failures = [(batch[0], batch_error)]
            else:
                failures = []
                for row in batch:
                    try:
                        output_rows = generate_batch([row])
                        append_jsonl_rows(output_path, output_rows)
                        completed_ids.add(row["candidate_id"])
                        generated_this_run += 1
                        consecutive_errors = 0
                    except Exception as single_error:
                        failures.append((row, single_error))
            for row, error in failures:
                failed_this_run += 1
                consecutive_errors += 1
                append_jsonl_rows(
                    errors_path,
                    [
                        {
                            "candidate_id": row["candidate_id"],
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "failed_at": utc_now(),
                        }
                    ],
                )
                print(f"failed {row['candidate_id']}: {type(error).__name__}: {error}", file=sys.stderr)
                if consecutive_errors >= args.max_consecutive_errors:
                    raise RuntimeError(
                        f"Stopped after {consecutive_errors} consecutive generation failures; "
                        "successful rows are checkpointed"
                    ) from error
        print(
            f"generated this run={generated_this_run:,}; failed={failed_this_run:,}; "
            f"covered={len(completed_ids):,}/{total_input:,}",
            flush=True,
        )

    verified = load_completed_candidate_ids(output_path)
    if verified != completed_ids:
        raise RuntimeError("Post-write candidate_id verification differs from in-memory state")
    result = {
        "input_records": total_input,
        "completed_records": len(verified),
        "generated_this_run": generated_this_run,
        "failed_this_run": failed_this_run,
        "unique_candidate_ids": len(verified),
        "complete": len(verified) == total_input,
        "output_file": str(output_path),
        "errors_file": str(errors_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if failed_this_run == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed batches remain flushed and can be resumed.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Teacher generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
