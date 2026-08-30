#!/usr/bin/env python3
"""Shared constants and reproducible I/O helpers for the zh-th v3 pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.zh_th_trans.quality_rules_v2 import (  # noqa: E402
    QUALITY_RULE_VERSION,
    assess_pair,
    count_han,
    exact_pair_key,
    is_thai,
    normalize_text,
    normalized_pair_key,
    normalized_side_key,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "zh_th_qwen3_8b_v3.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "zh_th_qwen3_8b_v3"
DEFAULT_SOURCE_CSV = PROJECT_ROOT / "dataset" / "final" / "zh_en" / "zh_en.csv"
DEFAULT_CANDIDATES = DEFAULT_OUTPUT_DIR / "candidates_24000.jsonl"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_OUTPUT_DIR / "candidates_manifest.json"
DEFAULT_RAW_GENERATIONS = DEFAULT_OUTPUT_DIR / "raw_teacher_generations.jsonl"
DEFAULT_AUDIT_ALL = DEFAULT_OUTPUT_DIR / "audit_all.jsonl"
DEFAULT_ACCEPTED = DEFAULT_OUTPUT_DIR / "accepted.jsonl"
DEFAULT_REJECTED = DEFAULT_OUTPUT_DIR / "rejected.jsonl"
DEFAULT_AUDIT_SUMMARY = DEFAULT_OUTPUT_DIR / "audit_summary.json"
DEFAULT_FINAL_CSV = PROJECT_ROOT / "dataset" / "final" / "zh_th" / "zh_th_qwen3_8b_v3.csv"
DEFAULT_FINAL_MANIFEST = (
    PROJECT_ROOT / "dataset" / "final" / "zh_th" / "manifest_qwen3_8b_v3.json"
)

DOMAINS = ("education", "technology", "finance")
SYSTEM_PROMPT = (
    "You are a professional translator. Translate accurately and output only the "
    "translation, without explanations or additional text."
)
PROMPT_TEMPLATE_VERSION = "qwen3_8b_teacher_zh_th_non_thinking_v1"
PROMPT_TEMPLATE = {
    "version": PROMPT_TEMPLATE_VERSION,
    "system": SYSTEM_PROMPT,
    "user": "Translate the following text from Chinese to Thai:\n\n{source_text}",
    "enable_thinking": False,
}
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    json.dumps(PROMPT_TEMPLATE, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()
REPEAT_NGRAM_SIZE = 20
PIPELINE_VERSION = "zh_th_qwen3_8b_teacher_v3.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield line_number, value


@contextlib.contextmanager
def atomic_text_writer(path: Path, *, encoding: str = "utf-8", newline: str = "\n") -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    with atomic_text_writer(path) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def load_config(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    value = read_json(resolved)
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be a JSON object: {resolved}")
    required = {
        "seed",
        "candidate_count",
        "final_pair_count",
        "candidate_domain_targets",
        "final_direction_targets",
        "generation",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Config is missing fields: {sorted(missing)}")
    if set(value["candidate_domain_targets"]) != set(DOMAINS):
        raise ValueError("candidate_domain_targets must contain education, technology and finance")
    if sum(int(value["candidate_domain_targets"][domain]) for domain in DOMAINS) != int(
        value["candidate_count"]
    ):
        raise ValueError("candidate domain targets do not sum to candidate_count")
    directions = value["final_direction_targets"]
    if set(directions) != {"zh-CN->th", "th->zh-CN"}:
        raise ValueError("final_direction_targets must contain both zh-th directions")
    if sum(int(count) for count in directions.values()) != int(value["final_pair_count"]):
        raise ValueError("direction targets do not sum to final_pair_count")
    generation = value["generation"]
    fixed_generation = {
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "dtype": "bfloat16",
    }
    conflicts = {
        key: {"actual": generation.get(key), "required": expected}
        for key, expected in fixed_generation.items()
        if generation.get(key) != expected
    }
    if conflicts:
        raise ValueError(f"Config conflicts with deterministic BF16 teacher generation: {conflicts}")
    return value


def build_teacher_messages(source_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE["user"].format(source_text=source_text),
        },
    ]


def repeat_score(text: str, n: int = REPEAT_NGRAM_SIZE) -> float:
    compact = "".join(normalize_text(text).split())
    if len(compact) < n:
        return 0.0
    total = len(compact) - n + 1
    grams = {compact[index : index + n] for index in range(total)}
    return 1.0 - len(grams) / total


def count_thai(text: str) -> int:
    return sum(is_thai(char) for char in text)


def canonical_pair_group_id(zh_text: str, th_text: str) -> str:
    texts = {
        "zh-CN": normalize_text(zh_text).casefold(),
        "th": normalize_text(th_text).casefold(),
    }
    payload = "\x1f".join(f"{lang}\x1e{texts[lang]}" for lang in sorted(texts))
    return "pair_" + sha256_text(payload)


def stable_derived_seed(seed: int, label: str) -> int:
    return seed ^ int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)


def load_completed_candidate_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for line_number, row in read_jsonl(path):
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError(f"Missing candidate_id at {path}:{line_number}")
        if candidate_id in completed:
            raise ValueError(f"Duplicate candidate_id {candidate_id!r} in {path}")
        completed.add(candidate_id)
    return completed


def ensure_new_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(float(ordered[index]), 6)


__all__ = [
    "DEFAULT_ACCEPTED",
    "DEFAULT_AUDIT_ALL",
    "DEFAULT_AUDIT_SUMMARY",
    "DEFAULT_CANDIDATES",
    "DEFAULT_CANDIDATE_MANIFEST",
    "DEFAULT_CONFIG",
    "DEFAULT_FINAL_CSV",
    "DEFAULT_FINAL_MANIFEST",
    "DEFAULT_RAW_GENERATIONS",
    "DEFAULT_REJECTED",
    "DEFAULT_SOURCE_CSV",
    "DOMAINS",
    "PIPELINE_VERSION",
    "PROMPT_TEMPLATE",
    "PROMPT_TEMPLATE_SHA256",
    "PROMPT_TEMPLATE_VERSION",
    "QUALITY_RULE_VERSION",
    "SYSTEM_PROMPT",
    "append_jsonl_rows",
    "assess_pair",
    "atomic_text_writer",
    "atomic_write_json",
    "atomic_write_jsonl",
    "build_teacher_messages",
    "canonical_pair_group_id",
    "count_han",
    "count_thai",
    "ensure_new_outputs",
    "exact_pair_key",
    "load_completed_candidate_ids",
    "load_config",
    "normalize_text",
    "normalized_pair_key",
    "normalized_side_key",
    "percentile",
    "read_json",
    "read_jsonl",
    "relative_path",
    "repeat_score",
    "resolve_path",
    "sha256_file",
    "sha256_text",
    "stable_derived_seed",
    "utc_now",
]
