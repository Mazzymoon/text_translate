#!/usr/bin/env python3

"""Clean technology-domain Chinese-English pairs from WikiMatrix.

WikiMatrix is a line-aligned, automatically mined corpus.  It does not provide
reliable document boundaries, so this cleaner deliberately never concatenates
adjacent lines.  It streams both language files with ``zip_longest`` and keeps
only individual Chinese lines within the configured length range.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import random
import re
import statistics
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ZIP = PROJECT_ROOT / "dataset" / "external" / "wikimatrix" / "raw" / "en-zh.txt.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "external" / "wikimatrix"
DATASET_NAME = "WikiMatrix"
DOMAIN = "technology"
ID_PREFIX = "wikimatrix_technology_"

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
LATIN_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,]\d+)*(?:%|％)?")

ENGLISH_KEYWORDS = (
    "technology",
    "technical",
    "science",
    "scientific",
    "computer",
    "software",
    "hardware",
    "internet",
    "network",
    "communication",
    "telecommunications",
    "artificial intelligence",
    "machine learning",
    "algorithm",
    "data",
    "robot",
    "automation",
    "semiconductor",
    "chip",
    "electronics",
    "digital",
    "cybersecurity",
    "biotechnology",
    "engineering",
    "energy",
    "aerospace",
)
CHINESE_KEYWORDS = (
    "科技",
    "技术",
    "科学",
    "计算机",
    "软件",
    "硬件",
    "互联网",
    "网络",
    "通信",
    "人工智能",
    "机器学习",
    "算法",
    "数据",
    "机器人",
    "自动化",
    "半导体",
    "芯片",
    "电子",
    "数字化",
    "网络安全",
    "生物技术",
    "工程",
    "能源",
    "航天",
)

# Common English plurals are recognized but stored under their canonical
# keyword.  Word boundaries prevent fragments such as "networking" from being
# silently treated as the exact "network" keyword.
ENGLISH_PATTERN_TEXT = {
    "technology": r"\btechnolog(?:y|ies)\b",
    "technical": r"\btechnical\b",
    "science": r"\bscience\b",
    "scientific": r"\bscientific\b",
    "computer": r"\bcomputers?\b",
    "software": r"\bsoftware\b",
    "hardware": r"\bhardware\b",
    "internet": r"\binternet\b",
    "network": r"\bnetworks?\b",
    "communication": r"\bcommunications?\b",
    "telecommunications": r"\btelecommunications?\b",
    "artificial intelligence": r"\bartificial\s+intelligence\b",
    "machine learning": r"\bmachine\s+learning\b",
    "algorithm": r"\balgorithms?\b",
    "data": r"\bdata\b",
    "robot": r"\brobots?\b",
    "automation": r"\bautomation\b",
    "semiconductor": r"\bsemiconductors?\b",
    "chip": r"\bchips?\b",
    "electronics": r"\belectronics?\b",
    "digital": r"\bdigital\b",
    "cybersecurity": r"\bcybersecurity\b",
    "biotechnology": r"\bbiotechnology\b",
    "engineering": r"\bengineering\b",
    "energy": r"\benergy\b",
    "aerospace": r"\baerospace\b",
}
ENGLISH_PREFILTERS = {
    **{keyword: keyword.casefold() for keyword in ENGLISH_KEYWORDS},
    "technology": "technolog",
    "computer": "computer",
    "network": "network",
    "communication": "communication",
    "telecommunications": "telecommunication",
    "algorithm": "algorithm",
    "robot": "robot",
    "semiconductor": "semiconductor",
    "chip": "chip",
    "electronics": "electronic",
}
ENGLISH_KEYWORD_PATTERNS = {
    keyword: re.compile(ENGLISH_PATTERN_TEXT[keyword], re.IGNORECASE)
    for keyword in ENGLISH_KEYWORDS
}
KEYWORD_ORDER = ENGLISH_KEYWORDS + CHINESE_KEYWORDS

SUBDOMAIN_KEYWORDS = {
    "ai_computing": {
        "computer",
        "software",
        "hardware",
        "artificial intelligence",
        "machine learning",
        "algorithm",
        "data",
        "robot",
        "automation",
        "计算机",
        "软件",
        "硬件",
        "人工智能",
        "机器学习",
        "算法",
        "数据",
        "机器人",
        "自动化",
    },
    "communication_network": {
        "internet",
        "network",
        "communication",
        "telecommunications",
        "cybersecurity",
        "互联网",
        "网络",
        "通信",
        "网络安全",
    },
    "electronics_semiconductor": {
        "semiconductor",
        "chip",
        "electronics",
        "digital",
        "半导体",
        "芯片",
        "电子",
        "数字化",
    },
    "biotechnology": {"biotechnology", "生物技术"},
    "engineering_manufacturing": {"technical", "engineering", "技术", "工程"},
    "energy_aerospace": {"energy", "aerospace", "能源", "航天"},
    "technology_general": {"technology", "science", "scientific", "科技", "科学"},
}
SUBDOMAIN_PRIORITY = tuple(SUBDOMAIN_KEYWORDS)

LANGUAGE_ERROR_REASONS = {
    "empty_text",
    "text_corruption",
    "identical_bilingual_text",
    "chinese_language_error",
    "english_language_error",
}


@dataclass(frozen=True)
class FileInspection:
    path: Path
    size_bytes: int
    samples: tuple[str, ...]
    english_score: float
    chinese_score: float


class AtomicJsonArrayWriter:
    def __init__(self, path: Path, metadata: dict) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.metadata = metadata
        self.handle: TextIO | None = None
        self.first_record = True
        self.finished = False

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temporary_path.open("w", encoding="utf-8", newline="\n")
        self.handle.write("{\n")
        for key, value in self.metadata.items():
            self.handle.write(f"  {json.dumps(key)}: ")
            self.handle.write(json.dumps(value, ensure_ascii=False))
            self.handle.write(",\n")
        self.handle.write('  "records": [')

    def write(self, record: dict) -> None:
        if self.handle is None:
            raise RuntimeError("JSON writer is not open")
        self.handle.write("\n" if self.first_record else ",\n")
        self.handle.write("    ")
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self.first_record = False

    def finish(self) -> None:
        if self.handle is None:
            raise RuntimeError("JSON writer is not open")
        if self.first_record:
            self.handle.write("]\n}\n")
        else:
            self.handle.write("\n  ]\n}\n")
        self.handle.close()
        self.handle = None
        self.finished = True

    def commit(self) -> None:
        if not self.finished:
            raise RuntimeError("JSON writer must be finished before commit")
        os.replace(self.temporary_path, self.path)

    def abort(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.temporary_path.exists():
            self.temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract technology-domain pairs from WikiMatrix."
    )
    parser.add_argument("--input-zip", type=Path, default=DEFAULT_INPUT_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-zh-chars", type=int, default=100)
    parser.add_argument("--max-zh-chars", type=int, default=220)
    parser.add_argument("--max-records", type=int, default=12000)
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("min_zh_chars", "max_zh_chars", "max_records", "sample"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.min_zh_chars < 1 or args.min_zh_chars > args.max_zh_chars:
        raise ValueError("Expected 1 <= min-zh-chars <= max-zh-chars")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value).replace("\ufeff", "")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def count_han(text: str) -> int:
    return len(HAN_RE.findall(text))


def count_english_words(text: str) -> int:
    return len(ENGLISH_WORD_RE.findall(text))


def looks_corrupt(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    markers = ("Ã", "Â", "â€", "锟", "�")
    return sum(text.count(marker) for marker in markers) >= 2


def validate_pair(en_text: str, zh_text: str) -> str | None:
    if not en_text or not zh_text:
        return "empty_text"
    if looks_corrupt(en_text) or looks_corrupt(zh_text):
        return "text_corruption"
    if en_text.casefold() == zh_text.casefold():
        return "identical_bilingual_text"
    zh_visible = max(1, sum(not char.isspace() for char in zh_text))
    if count_han(zh_text) < 2 or count_han(zh_text) / zh_visible < 0.10:
        return "chinese_language_error"
    en_visible = max(1, sum(not char.isspace() for char in en_text))
    if count_english_words(en_text) < 2 or len(LATIN_RE.findall(en_text)) / en_visible < 0.20:
        return "english_language_error"
    return None


def match_keywords(en_text: str, zh_text: str) -> tuple[str, ...]:
    lowered = en_text.casefold()
    matched = {
        keyword
        for keyword, pattern in ENGLISH_KEYWORD_PATTERNS.items()
        if ENGLISH_PREFILTERS[keyword] in lowered and pattern.search(en_text)
    }
    matched.update(keyword for keyword in CHINESE_KEYWORDS if keyword in zh_text)
    return tuple(keyword for keyword in KEYWORD_ORDER if keyword in matched)


def select_subdomain(keywords: Iterable[str]) -> str:
    keyword_set = set(keywords)
    scores = {
        name: len(keyword_set & values)
        for name, values in SUBDOMAIN_KEYWORDS.items()
    }
    return max(
        SUBDOMAIN_PRIORITY,
        key=lambda name: (scores[name], -SUBDOMAIN_PRIORITY.index(name)),
    )


def extract_numbers(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return {
        value.replace(",", "").replace("％", "%")
        for value in NUMBER_RE.findall(normalized)
    }


def quality_flags(en_text: str, zh_text: str) -> list[str]:
    flags: list[str] = []
    if extract_numbers(en_text) != extract_numbers(zh_text):
        flags.append("number_mismatch")
    ratio = count_han(zh_text) / max(1, count_english_words(en_text))
    if ratio < 0.50 or ratio > 4.00:
        flags.append("length_ratio_outlier")
    return flags


def pair_digest(en_text: str, zh_text: str) -> str:
    return hashlib.sha256(f"{zh_text}\x1f{en_text.casefold()}".encode("utf-8")).hexdigest()


def read_sample_lines(path: Path, limit: int = 5) -> tuple[str, ...]:
    result: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        for line in handle:
            text = line.rstrip("\r\n")
            if text:
                result.append(text)
            if len(result) >= limit:
                break
    return tuple(result)


def inspect_file(path: Path) -> FileInspection | None:
    try:
        samples = read_sample_lines(path)
    except (OSError, UnicodeDecodeError):
        return None
    if not samples or samples[0].lstrip().startswith(("<?xml", "<!DOCTYPE", "<cesAlign")):
        return None
    joined = " ".join(samples)
    visible = max(1, sum(not char.isspace() for char in joined))
    words = count_english_words(joined)
    latin = len(LATIN_RE.findall(joined))
    return FileInspection(
        path=path,
        size_bytes=path.stat().st_size,
        samples=samples,
        english_score=(words + latin / 10) / visible,
        chinese_score=count_han(joined) / visible,
    )


def choose_file(candidates: list[FileInspection], role: str) -> FileInspection:
    if not candidates:
        raise RuntimeError(f"Could not identify the {role} file")
    ordered = sorted(candidates, key=lambda item: item.size_bytes, reverse=True)
    if len(ordered) > 1 and ordered[1].size_bytes >= ordered[0].size_bytes * 0.80:
        names = ", ".join(str(item.path) for item in ordered[:3])
        raise RuntimeError(f"Ambiguous {role} files: {names}")
    return ordered[0]


def safe_extract_zip(zip_path: Path) -> Path:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Input zip does not exist: {zip_path}")
    extraction_dir = zip_path.parent / zip_path.stem
    if extraction_dir.name.endswith(".txt"):
        extraction_dir = extraction_dir.with_name(extraction_dir.name[:-4])
    extraction_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(extraction_dir.rglob("*"))
    if existing_files:
        logging.info("Using existing extraction directory: %s", extraction_dir)
        return extraction_dir
    root = extraction_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            destination = (extraction_dir / info.filename).resolve()
            if os.path.commonpath((str(root), str(destination))) != str(root):
                raise RuntimeError(f"Unsafe ZIP member path: {info.filename}")
        archive.extractall(extraction_dir)
    logging.info("Extracted %s to %s", zip_path, extraction_dir)
    return extraction_dir


def identify_input_files(input_dir: Path) -> tuple[FileInspection, FileInspection, list[FileInspection]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    inspections = [
        inspected
        for path in sorted(input_dir.rglob("*"))
        if path.is_file()
        for inspected in [inspect_file(path)]
        if inspected is not None
    ]
    english = choose_file(
        [item for item in inspections if item.english_score >= 0.15 and item.chinese_score < 0.05],
        "English text",
    )
    chinese = choose_file(
        [
            item
            for item in inspections
            if item.path != english.path and item.chinese_score >= 0.20
        ],
        "Chinese text",
    )
    return english, chinese, inspections


def file_metadata(inspection: FileInspection, role: str) -> dict:
    return {
        "role": role,
        "name": inspection.path.name,
        "path": inspection.path.resolve().as_posix(),
        "size_bytes": inspection.size_bytes,
        "first_five_lines": list(inspection.samples[:5]),
    }


def summarize(values: list[int]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
    }


class TechnologyCleaner:
    def __init__(self, args: argparse.Namespace, cleaned_writer: AtomicJsonArrayWriter, rejected_writer: AtomicJsonArrayWriter) -> None:
        self.args = args
        self.cleaned_writer = cleaned_writer
        self.rejected_writer = rejected_writer
        self.stats: Counter[str] = Counter()
        self.keyword_counts: Counter[str] = Counter()
        self.subdomain_candidate_counts: Counter[str] = Counter()
        self.final_subdomain_counts: Counter[str] = Counter()
        self.rejection_reasons: Counter[str] = Counter()
        self.seen_hashes: set[str] = set()
        self.zh_lengths: list[int] = []
        self.en_lengths: list[int] = []
        self.random_samples: list[dict] = []
        self.random = random.Random(20260805)

    @property
    def output_limit_reached(self) -> bool:
        return bool(self.args.max_records and self.stats["final_valid_records"] >= self.args.max_records)

    def write_rejection(self, reason: str, line_number: int, en_text: str, zh_text: str, details: str | None = None) -> None:
        record = {
            "rejection_reason": reason,
            "source_line": line_number,
            "zh_text": zh_text,
            "en_text": en_text,
        }
        if details:
            record["details"] = details
        self.rejected_writer.write(record)
        self.rejection_reasons[reason] += 1
        self.stats["rejected_records"] += 1

    def process_pair(self, line_number: int, en_text: str, zh_text: str) -> None:
        keywords = match_keywords(en_text, zh_text)
        if not keywords:
            return
        self.stats["technology_candidates"] += 1
        for keyword in keywords:
            self.keyword_counts[keyword] += 1
        subdomain = select_subdomain(keywords)
        self.subdomain_candidate_counts[subdomain] += 1
        if self.output_limit_reached:
            self.stats["candidates_skipped_by_max_records"] += 1
            return

        severe_error = validate_pair(en_text, zh_text)
        if severe_error:
            if severe_error in LANGUAGE_ERROR_REASONS:
                self.stats["language_or_text_errors"] += 1
            self.write_rejection(severe_error, line_number, en_text, zh_text)
            return

        zh_count = count_han(zh_text)
        en_count = count_english_words(en_text)
        if zh_count < self.args.min_zh_chars:
            self.stats["chinese_below_minimum"] += 1
            self.write_rejection("zh_chars_below_minimum", line_number, en_text, zh_text, f"zh_chars={zh_count}")
            return
        if zh_count > self.args.max_zh_chars:
            self.stats["chinese_above_maximum"] += 1
            self.write_rejection("zh_chars_above_maximum", line_number, en_text, zh_text, f"zh_chars={zh_count}")
            return

        digest = pair_digest(en_text, zh_text)
        if digest in self.seen_hashes:
            self.stats["exact_duplicates_removed"] += 1
            self.write_rejection("duplicate_parallel_pair", line_number, en_text, zh_text)
            return

        flags = quality_flags(en_text, zh_text)
        if "number_mismatch" in flags:
            self.stats["number_mismatches"] += 1
        if "length_ratio_outlier" in flags:
            self.stats["length_ratio_outliers"] += 1
        record = {
            "id": f"{ID_PREFIX}{digest[:20]}",
            "zh_text": zh_text,
            "en_text": en_text,
            "zh_char_count": zh_count,
            "en_word_count": en_count,
            "domain": DOMAIN,
            "subdomain": subdomain,
            "matched_keywords": list(keywords),
            "data_origin": "public_parallel_auto_aligned",
            "dataset_name": DATASET_NAME,
            "source_line": line_number,
            "quality_status": "passed",
            "quality_flags": flags,
            "pair_sha256": digest,
        }
        self.cleaned_writer.write(record)
        self.seen_hashes.add(digest)
        self.stats["final_valid_records"] += 1
        self.final_subdomain_counts[subdomain] += 1
        self.zh_lengths.append(zh_count)
        self.en_lengths.append(en_count)
        self._reservoir_sample(record)

    def _reservoir_sample(self, record: dict) -> None:
        count = self.stats["final_valid_records"]
        if len(self.random_samples) < 5:
            self.random_samples.append(record)
            return
        index = self.random.randrange(count)
        if index < 5:
            self.random_samples[index] = record


def process_corpus(args: argparse.Namespace, english: FileInspection, chinese: FileInspection, cleaner: TechnologyCleaner) -> None:
    sentinel = object()
    with english.path.open("r", encoding="utf-8-sig", errors="strict", newline="") as en_handle, chinese.path.open(
        "r", encoding="utf-8-sig", errors="strict", newline=""
    ) as zh_handle:
        aligned: Iterator[tuple[str | object, str | object]] = itertools.zip_longest(
            en_handle, zh_handle, fillvalue=sentinel
        )
        if args.sample:
            aligned = itertools.islice(aligned, args.sample)
        for line_number, (en_line, zh_line) in enumerate(aligned, start=1):
            if en_line is sentinel or zh_line is sentinel:
                cleaner.stats["line_count_mismatches"] += 1
                raise RuntimeError(f"English/Chinese line counts differ at row {line_number}")
            cleaner.stats["raw_aligned_pairs"] += 1
            en_text = normalize_text(str(en_line).rstrip("\r\n"))
            zh_text = normalize_text(str(zh_line).rstrip("\r\n"))
            cleaner.process_pair(line_number, en_text, zh_text)
            if line_number % 500_000 == 0:
                logging.info(
                    "Processed %s pairs; candidates=%s; accepted=%s; rejected=%s",
                    f"{line_number:,}",
                    f"{cleaner.stats['technology_candidates']:,}",
                    f"{cleaner.stats['final_valid_records']:,}",
                    f"{cleaner.stats['rejected_records']:,}",
                )


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_report(
    args: argparse.Namespace,
    english: FileInspection,
    chinese: FileInspection,
    inspections: list[FileInspection],
    input_dir: Path,
    cleaner: TechnologyCleaner,
    generated_at: str,
) -> dict:
    stats = cleaner.stats
    return {
        "schema_version": 1,
        "language_pair": "zh_en",
        "dataset_name": DATASET_NAME,
        "domain": DOMAIN,
        "generated_at": generated_at,
        "input_files": [
            file_metadata(english, "english_text"),
            file_metadata(chinese, "chinese_text"),
        ],
        "discovered_files": [
            file_metadata(item, "discovered_text_candidate")
            for item in inspections
        ],
        "extracted_files": [
            {
                "name": path.relative_to(input_dir).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(input_dir.rglob("*"))
            if path.is_file()
        ],
        "parameters": {
            "min_zh_chars": args.min_zh_chars,
            "max_zh_chars": args.max_zh_chars,
            "max_records": args.max_records,
            "sample": args.sample,
            "line_count_check_scope": f"first_{args.sample}_rows" if args.sample else "full_files",
        },
        "statistics": {
            "raw_aligned_pairs": stats["raw_aligned_pairs"],
            "line_count_mismatches": stats["line_count_mismatches"],
            "technology_candidates": stats["technology_candidates"],
            "keyword_hit_counts": dict(cleaner.keyword_counts),
            "subdomain_candidate_counts": dict(cleaner.subdomain_candidate_counts),
            "chinese_below_100": stats["chinese_below_minimum"],
            "chinese_above_220": stats["chinese_above_maximum"],
            "language_or_text_errors": stats["language_or_text_errors"],
            "exact_duplicates_removed": stats["exact_duplicates_removed"],
            "number_mismatches": stats["number_mismatches"],
            "length_ratio_outliers": stats["length_ratio_outliers"],
            "final_valid_records": stats["final_valid_records"],
            "final_records_by_subdomain": dict(cleaner.final_subdomain_counts),
            "rejected_records": stats["rejected_records"],
            "rejection_reasons": dict(cleaner.rejection_reasons),
            "candidates_skipped_by_max_records": stats["candidates_skipped_by_max_records"],
        },
        "length_statistics": {
            "zh_char_count": summarize(cleaner.zh_lengths),
            "en_word_count": summarize(cleaner.en_lengths),
        },
        "random_samples": cleaner.random_samples,
    }


def log_file(role: str, inspection: FileInspection) -> None:
    logging.info("Identified %s: %s (%s bytes)", role, inspection.path.name, f"{inspection.size_bytes:,}")
    for index, sample in enumerate(inspection.samples[:5], start=1):
        logging.info("  sample %d: %s", index, sample[:300])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        validate_args(args)
        zip_path = args.input_zip if args.input_zip.is_absolute() else PROJECT_ROOT / args.input_zip
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
        input_dir = safe_extract_zip(zip_path.resolve())
        english, chinese, inspections = identify_input_files(input_dir)
        log_file("English text", english)
        log_file("Chinese text", chinese)
        for inspection in inspections:
            if inspection.path not in {english.path, chinese.path}:
                logging.info(
                    "Discovered auxiliary text candidate: %s (%s bytes)",
                    inspection.path.name,
                    f"{inspection.size_bytes:,}",
                )
        generated_at = utc_now()
        metadata = {
            "schema_version": 1,
            "language_pair": "zh_en",
            "dataset_name": DATASET_NAME,
            "domain": DOMAIN,
            "generated_at": generated_at,
        }
        cleaned_writer = AtomicJsonArrayWriter(output_dir / "cleaned" / "technology_pairs.json", metadata)
        rejected_writer = AtomicJsonArrayWriter(output_dir / "rejected" / "technology_rejected.json", metadata)
        cleaned_writer.open()
        rejected_writer.open()
        cleaner = TechnologyCleaner(args, cleaned_writer, rejected_writer)
        try:
            process_corpus(args, english, chinese, cleaner)
            cleaned_writer.finish()
            rejected_writer.finish()
            report = build_report(
                args,
                english,
                chinese,
                inspections,
                input_dir,
                cleaner,
                generated_at,
            )
            write_json_atomic(output_dir / "technology_cleaning_report.json", report)
            cleaned_writer.commit()
            rejected_writer.commit()
        except Exception:
            cleaned_writer.abort()
            rejected_writer.abort()
            raise
        logging.info(
            "Completed: raw=%s candidates=%s accepted=%s rejected=%s",
            f"{cleaner.stats['raw_aligned_pairs']:,}",
            f"{cleaner.stats['technology_candidates']:,}",
            f"{cleaner.stats['final_valid_records']:,}",
            f"{cleaner.stats['rejected_records']:,}",
        )
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        logging.error("WikiMatrix cleaning failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
