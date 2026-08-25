#!/usr/bin/env python3
"""Run deterministic Qwen3 base or LoRA translation inference on the internal test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST = ROOT / "outputs/sft_data/test.jsonl"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-name-or-path", help="Omit for base-model inference")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    user = PROMPT_TEMPLATE_SPEC["user"].format(
        source_language=LANGUAGE_NAMES[record["source_lang"]],
        target_language=LANGUAGE_NAMES[record["target_lang"]],
        source_text=record["source_text"],
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("batch-size and save-every must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if args.adapter_name_or_path:
            from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Install torch, transformers, peft and bitsandbytes on AutoDL") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    torch.cuda.set_device(0)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_model_download,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=not args.allow_model_download,
        trust_remote_code=False,
    )
    variant = "base"
    if args.adapter_name_or_path:
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, is_trainable=False)
        variant = "lora"
    model.eval()

    input_path = args.input_file.resolve()
    manifest_path = input_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing internal-test manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256:
        raise RuntimeError("Prompt template hash differs from prepared internal test")
    rows = read_jsonl(input_path)
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("inference_transformers.py only accepts the internal test split")
        if row.get("prompt_messages") != build_messages(row):
            raise ValueError(f"{row.get('record_id')}: stored prompt differs from inference template")

    output_path = args.output_file.resolve()
    completed: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        completed = read_jsonl(output_path)
        completed_ids = {row["record_id"] for row in completed if row.get("prediction")}
    pending = [row for row in rows if row["record_id"] not in completed_ids]
    generation_parameters = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(row),
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
            max_length=args.max_input_length,
            add_special_tokens=False,
        ).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_parameters)
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, prediction in zip(batch, predictions):
            completed.append(
                {
                    "record_id": row["record_id"],
                    "pair_group_id": row["pair_group_id"],
                    "split": "test",
                    "direction": row["direction"],
                    "source_lang": row["source_lang"],
                    "target_lang": row["target_lang"],
                    "source_text": row["source_text"],
                    "reference": row["target_text"],
                    "prediction": prediction.strip(),
                    "model_variant": variant,
                    "model_name_or_path": args.model_name_or_path,
                    "adapter_name_or_path": args.adapter_name_or_path,
                    "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                    "decoding": {"enable_thinking": False, **generation_parameters},
                }
            )
        if len(completed) % args.save_every < len(batch):
            atomic_write_jsonl(output_path, completed)
            print(f"Saved {len(completed)}/{len(rows)} predictions")
    atomic_write_jsonl(output_path, completed)
    print(json.dumps({"variant": variant, "total": len(rows), "completed": len(completed)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; the most recent checkpointed predictions remain usable.", file=sys.stderr)
        raise SystemExit(130)
