#!/usr/bin/env python3
"""Single-GPU 4-bit QLoRA training for Qwen3-8B translation SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/qwen3_8b_qlora_full.json"
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4))
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", help="auto, none, or a checkpoint path")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--check-config", action="store_true", help="Validate configuration without loading CUDA/model")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config.resolve())
    overrides = {
        "model_name_or_path": args.model_name_or_path,
        "train_file": str(args.train_file) if args.train_file else None,
        "validation_file": str(args.validation_file) if args.validation_file else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.allow_model_download:
        config["local_files_only"] = False
    required = ("model_name_or_path", "train_file", "validation_file", "output_dir")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    batch = int(config.get("per_device_train_batch_size", 2))
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    allowed = {(1, 16), (2, 8), (4, 4)}
    if (batch, accumulation) not in allowed:
        raise ValueError(
            f"Unsupported batch/gradient accumulation combination {(batch, accumulation)}; "
            f"choose one of {sorted(allowed)} to keep effective batch 16"
        )
    config["effective_batch_size"] = batch * accumulation
    if int(config.get("max_length", 1024)) <= 0:
        raise ValueError("max_length must be positive")
    return config


def package_version_tuple(name: str) -> tuple[int, ...]:
    raw = metadata.version(name)
    match = re.match(r"(\d+(?:\.\d+)+)", raw)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def stable_limit(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit or limit <= 0 or len(rows) <= limit:
        return rows
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    source_lang = record["source_lang"]
    target_lang = record["target_lang"]
    user = PROMPT_TEMPLATE_SPEC["user"].format(
        source_language=LANGUAGE_NAMES[source_lang],
        target_language=LANGUAGE_NAMES[target_lang],
        source_text=record["source_text"],
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def tokenize_record(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, Any]:
    messages = build_messages(record)
    if record.get("prompt_messages") != messages:
        raise ValueError(f"{record.get('record_id')}: stored prompt_messages do not match template")
    if record.get("completion") != record.get("target_text"):
        raise ValueError(f"{record.get('record_id')}: completion differs from target_text")
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(record["target_text"], add_special_tokens=False)["input_ids"]
    eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids + eos_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    valid_target_tokens = min(len(target_ids), max(0, max_length - len(prompt_ids)))
    if valid_target_tokens <= 0 or not any(label != -100 for label in labels):
        raise ValueError(f"{record.get('record_id')}: no completion supervision survives truncation")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": prompt_length,
        "record_id": record["record_id"],
    }


def assert_single_gpu_placement(model: Any) -> dict[str, Any]:
    allowed_map_values = {0, "0", "cuda", "cuda:0"}
    device_map = getattr(model, "hf_device_map", None) or {"": "cuda:0"}
    invalid_map = {name: str(device) for name, device in device_map.items() if device not in allowed_map_values}
    invalid_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda" or parameter.device.index not in (None, 0):
            invalid_parameters.append(f"{name}:{parameter.device}")
            if len(invalid_parameters) >= 10:
                break
    invalid_buffers = []
    for name, buffer in model.named_buffers():
        if buffer.device.type not in ("cuda",):
            invalid_buffers.append(f"{name}:{buffer.device}")
            if len(invalid_buffers) >= 10:
                break
    if invalid_map or invalid_parameters or invalid_buffers:
        raise RuntimeError(
            "CPU/disk offload detected; refusing to train. "
            f"invalid_device_map={invalid_map}, invalid_parameters={invalid_parameters}, "
            f"invalid_buffers={invalid_buffers}"
        )
    return {str(name): str(device) for name, device in device_map.items()}


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir() and path.name.split("-")[-1].isdigit():
            candidates.append((int(path.name.split("-")[-1]), path))
    return max(candidates, default=(0, None))[1]


def resolve_resume(value: str | None, output_dir: Path) -> str | None:
    if value is None or value.lower() == "auto":
        checkpoint = latest_checkpoint(output_dir)
        return str(checkpoint) if checkpoint else None
    if value.lower() in {"none", "false", "no"}:
        return None
    path = resolve_project_path(value)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return str(path)


def main() -> int:
    args = parse_args()
    config = build_config(args)
    printable = dict(config)
    print(json.dumps({"validated_config": printable}, ensure_ascii=False, indent=2))
    if args.check_config:
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install torch, transformers>=4.51, datasets, "
            "trl, peft, accelerate and bitsandbytes on AutoDL."
        ) from exc

    if package_version_tuple("transformers") < (4, 51, 0):
        raise RuntimeError("Qwen3 requires transformers>=4.51.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU training and silent fallback are disabled")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    gpu_report = {
        "gpu_name": properties.name,
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 3),
        "cuda_version": torch.version.cuda,
        "bf16_supported": bf16_supported,
        "compute_dtype": str(compute_dtype),
        "requested_device_map": {"": 0},
    }
    print(json.dumps({"gpu_preflight": gpu_report}, ensure_ascii=False, indent=2))

    train_file = resolve_project_path(config["train_file"])
    validation_file = resolve_project_path(config["validation_file"])
    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = train_file.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SFT manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256:
        raise RuntimeError("Prompt template hash differs from prepared SFT data")
    if not manifest.get("tokenizer_validation_performed"):
        raise RuntimeError("SFT data was not formally prepared with tokenizer validation")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=False,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer has no eos_token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    final_device_map = assert_single_gpu_placement(model)
    gpu_report["final_device_map"] = final_device_map
    model.print_trainable_parameters()
    print(json.dumps({"gpu_placement": gpu_report}, ensure_ascii=False, indent=2))

    seed = int(config.get("seed", 20260824))
    max_length = int(config.get("max_length", 1024))
    train_rows = stable_limit(read_jsonl(train_file), config.get("max_train_samples"), seed)
    validation_rows = stable_limit(read_jsonl(validation_file), config.get("max_eval_samples"), seed + 1)
    train_features = [tokenize_record(tokenizer, row, max_length) for row in train_rows]
    validation_features = [tokenize_record(tokenizer, row, max_length) for row in validation_rows]

    import torch as torch_module

    class CompletionOnlyCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            maximum = max(len(feature["input_ids"]) for feature in features)
            input_ids, attention_masks, labels = [], [], []
            for feature in features:
                padding = maximum - len(feature["input_ids"])
                input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
                attention_masks.append(feature["attention_mask"] + [0] * padding)
                labels.append(feature["labels"] + [-100] * padding)
            return {
                "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
                "attention_mask": torch_module.tensor(attention_masks, dtype=torch_module.long),
                "labels": torch_module.tensor(labels, dtype=torch_module.long),
            }

    collator = CompletionOnlyCollator()
    audit_rows = train_features[: min(3, len(train_features))]
    if not audit_rows:
        raise RuntimeError("Training split is empty")
    audit_batch = collator(audit_rows)
    audit_report = []
    for index, feature in enumerate(audit_rows):
        labels = audit_batch["labels"][index].tolist()[: len(feature["input_ids"])]
        prompt_length = feature["prompt_length"]
        prompt_masked = all(label == -100 for label in labels[:prompt_length])
        completion_count = sum(label != -100 for label in labels[prompt_length:])
        all_masked = all(label == -100 for label in labels)
        audit_report.append(
            {
                "record_id": feature["record_id"],
                "prompt_tokens": prompt_length,
                "completion_label_tokens": completion_count,
                "prompt_all_minus_100": prompt_masked,
                "all_labels_minus_100": all_masked,
            }
        )
        if not prompt_masked or completion_count <= 0 or all_masked:
            raise RuntimeError(f"Completion-only label audit failed: {audit_report[-1]}")
    print(json.dumps({"completion_only_label_audit": audit_report}, ensure_ascii=False, indent=2))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 2)),
        max_steps=int(config.get("max_steps", -1)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        warmup_steps=float(config.get("warmup_ratio", 0.03)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 0.3)),
        optim=str(config.get("optim", "paged_adamw_8bit")),
        logging_steps=int(config.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(config.get("eval_steps", 500)),
        save_strategy="steps",
        save_steps=int(config.get("save_steps", 500)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        tf32=bool(config.get("tf32", True)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 4)),
        report_to=[],
        remove_unused_columns=False,
        max_length=max_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_features),
        eval_dataset=Dataset.from_list(validation_features),
        data_collator=collator,
        processing_class=tokenizer,
    )
    resume = resolve_resume(args.resume_from_checkpoint, output_dir)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.time()
    run_summary: dict[str, Any] = {
        "status": "running",
        "config": config,
        "gpu": gpu_report,
        "completion_only_label_audit": audit_report,
        "resume_from_checkpoint": resume,
    }
    atomic_write_json(output_dir / "run_summary.json", run_summary)
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        metrics = dict(result.metrics)
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(output_dir / "adapter_final"))
        tokenizer.save_pretrained(str(output_dir / "adapter_final"))
        trainer.save_state()
        trainer.save_metrics("train", metrics)
        trainer.save_metrics("eval", eval_metrics)
        run_summary.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.time() - started, 3),
                "peak_gpu_memory_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
                "peak_gpu_memory_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 1024**3, 3),
                "samples_per_second": metrics.get("train_samples_per_second"),
                "steps_per_second": metrics.get("train_steps_per_second"),
                "train_metrics": metrics,
                "eval_metrics": eval_metrics,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        run_summary.update(
            {
                "status": "cuda_oom",
                "elapsed_seconds": round(time.time() - started, 3),
                "peak_gpu_memory_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
                "peak_gpu_memory_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 1024**3, 3),
                "error": str(exc),
            }
        )
        atomic_write_json(output_dir / "run_summary.json", run_summary)
        raise RuntimeError("CUDA OOM; no CPU fallback was attempted") from exc
    atomic_write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
