#!/usr/bin/env python3

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
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "external" / "unpc" / "raw" / "en-zh"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "external" / "unpc"
DATASET_NAME = "UN Parallel Corpus v1.0"
DOMAIN = "finance"
ID_PREFIX = "unpc_finance_"
ANCHOR_STAT_KEY = "finance_anchor_sentences"
CLEANED_FILENAME = "finance_pairs.json"
REJECTED_FILENAME = "finance_rejected.json"
REPORT_FILENAME = "cleaning_report.json"
DEFAULT_MAX_RECORDS = 12000
REQUIRED_MIN_RECORDS = 0

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
LATIN_RE = re.compile(r"[A-Za-z]")
ID_REFERENCE_RE = re.compile(r"(?:en|zh):\d+:\d+")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,]\d+)*(?:%|\uff05)?")

ENGLISH_KEYWORDS = (
    "finance",
    "financial",
    "economic",
    "economy",
    "budget",
    "tax",
    "taxation",
    "debt",
    "investment",
    "investor",
    "trade",
    "banking",
    "bank",
    "currency",
    "monetary",
    "fiscal",
)
CHINESE_KEYWORDS = (
    "财政",
    "金融",
    "预算",
    "税收",
    "税务",
    "债务",
    "投资",
    "投资者",
    "贸易",
    "银行",
    "货币",
    "汇率",
    "经济",
)
ENGLISH_KEYWORD_PATTERNS = {
    keyword: re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in ENGLISH_KEYWORDS
}
ENGLISH_KEYWORD_PREFILTERS = {
    keyword: keyword.casefold() for keyword in ENGLISH_KEYWORDS
}
KEYWORD_ORDER = ENGLISH_KEYWORDS + CHINESE_KEYWORDS

SUBDOMAIN_KEYWORDS = {
    "fiscal_budget": {"fiscal", "budget", "财政", "预算"},
    "tax": {"tax", "taxation", "税收", "税务"},
    "debt": {"debt", "债务"},
    "investment": {"investment", "investor", "投资", "投资者"},
    "trade": {"trade", "贸易"},
    "banking_currency": {
        "bank",
        "banking",
        "currency",
        "monetary",
        "银行",
        "货币",
        "汇率",
    },
    "macroeconomy": {"economic", "economy", "经济"},
    "finance_general": {"finance", "financial", "金融"},
}
SUBDOMAIN_PRIORITY = tuple(SUBDOMAIN_KEYWORDS)

LANGUAGE_ERROR_REASONS = {
    "chinese_language_error",
    "english_language_error",
    "text_corruption",
    "identical_bilingual_text",
    "empty_text",
}


@dataclass(frozen=True)
class ParsedId:
    document_id: str
    sentence_id: str
    english_references: tuple[str, ...]
    chinese_references: tuple[str, ...]


@dataclass(frozen=True)
class AlignedPair:
    line_number: int
    document_id: str
    sentence_id: str
    en_text: str
    zh_text: str
    matched_keywords: tuple[str, ...]
    severe_error: str | None


@dataclass(frozen=True)
class FileInspection:
    path: Path
    size_bytes: int
    samples: tuple[str, ...]
    id_like: bool
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
        for index, (key, value) in enumerate(self.metadata.items()):
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
        self.handle.write("\n  ]\n}\n" if not self.first_record else "]\n}\n")
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
        description=f"Extract aligned {DOMAIN} text blocks from the UN Parallel Corpus."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-zh-chars", type=int, default=100)
    parser.add_argument("--target-zh-chars", type=int, default=150)
    parser.add_argument("--max-zh-chars", type=int, default=220)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="Maximum accepted records; 0 means unlimited.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Process only the first N aligned rows; 0 means the full corpus.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "min_zh_chars",
        "target_zh_chars",
        "max_zh_chars",
        "context_window",
        "max_records",
        "sample",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.min_zh_chars < 1:
        raise ValueError("--min-zh-chars must be positive")
    if not args.min_zh_chars <= args.target_zh_chars <= args.max_zh_chars:
        raise ValueError("Expected min-zh-chars <= target-zh-chars <= max-zh-chars")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def count_han(text: str) -> int:
    return len(HAN_RE.findall(text))


def count_english_words(text: str) -> int:
    return len(ENGLISH_WORD_RE.findall(text))


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value).replace("\ufeff", "")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def parse_id_line(value: str) -> ParsedId:
    parts = value.strip().split()
    if len(parts) < 3:
        raise ValueError("ID line must contain a document ID and both language references")
    document_id = parts[0]
    references = parts[1:]
    if not document_id or any(ID_REFERENCE_RE.fullmatch(item) is None for item in references):
        raise ValueError("ID references do not match '<en|zh>:<segment>:<part>'")
    english_references = tuple(item for item in references if item.startswith("en:"))
    chinese_references = tuple(item for item in references if item.startswith("zh:"))
    if not english_references or not chinese_references:
        raise ValueError("ID line must contain at least one en and one zh reference")
    return ParsedId(
        document_id=document_id,
        sentence_id=" ".join(references),
        english_references=english_references,
        chinese_references=chinese_references,
    )


def read_sample_lines(path: Path, limit: int = 5) -> tuple[str, ...]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        for line in handle:
            text = line.rstrip("\r\n")
            if text:
                lines.append(text)
            if len(lines) >= limit:
                break
    return tuple(lines)


def inspect_file(path: Path) -> FileInspection | None:
    try:
        samples = read_sample_lines(path)
    except (UnicodeDecodeError, OSError):
        return None
    if not samples:
        return None
    id_like = all(_can_parse_id(item) for item in samples)
    joined = " ".join(samples)
    han = count_han(joined)
    words = count_english_words(joined)
    latin = len(LATIN_RE.findall(joined))
    visible = max(1, sum(not char.isspace() for char in joined))
    return FileInspection(
        path=path,
        size_bytes=path.stat().st_size,
        samples=samples,
        id_like=id_like,
        english_score=(words + latin / 10) / visible,
        chinese_score=han / visible,
    )


def _can_parse_id(value: str) -> bool:
    try:
        parse_id_line(value)
        return True
    except ValueError:
        return False


def choose_largest_unambiguous(
    candidates: list[FileInspection], role: str
) -> FileInspection:
    if not candidates:
        raise RuntimeError(f"Could not identify the {role} file from content")
    ordered = sorted(candidates, key=lambda item: item.size_bytes, reverse=True)
    if len(ordered) > 1 and ordered[1].size_bytes >= ordered[0].size_bytes * 0.80:
        names = ", ".join(str(item.path) for item in ordered[:3])
        raise RuntimeError(f"Ambiguous {role} files: {names}")
    return ordered[0]


def identify_input_files(input_dir: Path) -> tuple[FileInspection, FileInspection, FileInspection]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    inspections = [
        inspected
        for path in sorted(input_dir.rglob("*"))
        if path.is_file()
        for inspected in [inspect_file(path)]
        if inspected is not None
    ]
    id_file = choose_largest_unambiguous(
        [item for item in inspections if item.id_like], "alignment ID"
    )
    remaining = [item for item in inspections if item.path != id_file.path]
    chinese_file = choose_largest_unambiguous(
        [item for item in remaining if item.chinese_score >= 0.20], "Chinese text"
    )
    english_file = choose_largest_unambiguous(
        [
            item
            for item in remaining
            if item.path != chinese_file.path
            and item.english_score >= 0.15
            and item.chinese_score < 0.05
        ],
        "English text",
    )
    return english_file, chinese_file, id_file


def match_keywords(en_text: str, zh_text: str) -> tuple[str, ...]:
    lowered_english = en_text.casefold()
    matched = {
        keyword
        for keyword, pattern in ENGLISH_KEYWORD_PATTERNS.items()
        if ENGLISH_KEYWORD_PREFILTERS[keyword] in lowered_english
        and pattern.search(en_text)
    }
    matched.update(keyword for keyword in CHINESE_KEYWORDS if keyword in zh_text)
    return tuple(keyword for keyword in KEYWORD_ORDER if keyword in matched)


def select_subdomain(keywords: Iterable[str]) -> str:
    keyword_set = set(keywords)
    scores = {
        subdomain: len(keyword_set & subdomain_keywords)
        for subdomain, subdomain_keywords in SUBDOMAIN_KEYWORDS.items()
    }
    return max(SUBDOMAIN_PRIORITY, key=lambda name: (scores[name], -SUBDOMAIN_PRIORITY.index(name)))


def looks_corrupt(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    markers = ("\u93c1\u6b0f\u7b00", "\u9225", "\u951b", "\u9286", "Ã", "Â", "â€")
    return sum(text.count(marker) for marker in markers) >= 2


def validate_sentence_pair(en_text: str, zh_text: str) -> str | None:
    if not en_text or not zh_text:
        return "empty_text"
    if looks_corrupt(en_text) or looks_corrupt(zh_text):
        return "text_corruption"
    if en_text.casefold() == zh_text.casefold():
        return "identical_bilingual_text"
    zh_han = count_han(zh_text)
    zh_visible = max(1, sum(not char.isspace() for char in zh_text))
    if zh_han < 2 or zh_han / zh_visible < 0.10:
        return "chinese_language_error"
    en_words = count_english_words(en_text)
    en_visible = max(1, sum(not char.isspace() for char in en_text))
    if en_words < 2 or len(LATIN_RE.findall(en_text)) / en_visible < 0.20:
        return "english_language_error"
    return None


def extract_numbers(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    numbers = set()
    for match in NUMBER_RE.findall(normalized):
        value = match.replace(",", "").replace("％", "%")
        numbers.add(value)
    return numbers


def quality_flags(en_text: str, zh_text: str) -> list[str]:
    flags: list[str] = []
    if extract_numbers(en_text) != extract_numbers(zh_text):
        flags.append("number_mismatch")
    en_words = max(1, count_english_words(en_text))
    ratio = count_han(zh_text) / en_words
    if ratio < 0.50 or ratio > 4.00:
        flags.append("length_ratio_outlier")
    return flags


def pair_digest(en_text: str, zh_text: str) -> str:
    return hashlib.sha256(f"{zh_text}\x1f{en_text.casefold()}".encode("utf-8")).hexdigest()


def join_rows(rows: list[AlignedPair]) -> tuple[str, str]:
    return (
        " ".join(row.en_text for row in rows),
        " ".join(row.zh_text for row in rows),
    )


class UnpcFinanceCleaner:
    def __init__(
        self,
        args: argparse.Namespace,
        cleaned_writer: AtomicJsonArrayWriter,
        rejected_writer: AtomicJsonArrayWriter,
    ) -> None:
        self.args = args
        self.cleaned_writer = cleaned_writer
        self.rejected_writer = rejected_writer
        self.stats: Counter[str] = Counter()
        self.keyword_counts: Counter[str] = Counter()
        self.subdomain_candidate_counts: Counter[str] = Counter()
        self.final_subdomain_counts: Counter[str] = Counter()
        self.rejection_reasons: Counter[str] = Counter()
        self.records_per_document: Counter[str] = Counter()
        self.used_lines: set[int] = set()
        self.seen_pair_hashes: set[str] = set()
        self.zh_lengths: list[int] = []
        self.en_lengths: list[int] = []
        self.random_samples: list[dict] = []
        self.random = random.Random(20260805)

    @property
    def output_limit_reached(self) -> bool:
        return bool(
            self.args.max_records
            and self.stats["final_valid_records"] >= self.args.max_records
        )

    def note_anchor(self, keywords: tuple[str, ...]) -> None:
        self.stats[ANCHOR_STAT_KEY] += 1
        for keyword in keywords:
            self.keyword_counts[keyword] += 1
        self.subdomain_candidate_counts[select_subdomain(keywords)] += 1

    def write_rejection(
        self,
        reason: str,
        rows: list[AlignedPair],
        *,
        document_id: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        en_text: str | None = None,
        zh_text: str | None = None,
        details: str | None = None,
    ) -> None:
        if rows:
            en_text, zh_text = join_rows(rows)
            document_id = rows[0].document_id
            start_line = rows[0].line_number
            end_line = rows[-1].line_number
        record = {
            "rejection_reason": reason,
            "document_id": document_id,
            "start_line": start_line,
            "end_line": end_line,
            "zh_text": zh_text or "",
            "en_text": en_text or "",
        }
        if details:
            record["details"] = details
        self.rejected_writer.write(record)
        self.rejection_reasons[reason] += 1
        self.stats["rejected_records"] += 1

    def _available_bounds(self, rows: list[AlignedPair], anchor_index: int) -> tuple[int, int]:
        lower = anchor_index
        upper = anchor_index
        for candidate in range(anchor_index - 1, max(-1, anchor_index - self.args.context_window - 1), -1):
            current = rows[candidate]
            following = rows[candidate + 1]
            if (
                current.severe_error
                or current.line_number in self.used_lines
                or current.line_number + 1 != following.line_number
            ):
                break
            lower = candidate
        for candidate in range(anchor_index + 1, min(len(rows), anchor_index + self.args.context_window + 1)):
            current = rows[candidate]
            previous = rows[candidate - 1]
            if (
                current.severe_error
                or current.line_number in self.used_lines
                or previous.line_number + 1 != current.line_number
            ):
                break
            upper = candidate
        return lower, upper

    def _select_interval(
        self, rows: list[AlignedPair], anchor_index: int, lower: int, upper: int
    ) -> list[AlignedPair] | None:
        choices: list[tuple[tuple[int, int, int, int], list[AlignedPair]]] = []
        for start in range(lower, anchor_index + 1):
            for end in range(anchor_index, upper + 1):
                selected = rows[start : end + 1]
                en_text, zh_text = join_rows(selected)
                zh_count = count_han(zh_text)
                en_count = count_english_words(en_text)
                if not self.args.min_zh_chars <= zh_count <= self.args.max_zh_chars:
                    continue
                if en_count < 20:
                    continue
                score = (
                    abs(zh_count - self.args.target_zh_chars),
                    abs((anchor_index - start) - (end - anchor_index)),
                    len(selected),
                    start,
                )
                choices.append((score, selected))
        return min(choices, key=lambda item: item[0])[1] if choices else None

    def process_document(self, rows: list[AlignedPair]) -> None:
        for anchor_index, anchor in enumerate(rows):
            if not anchor.matched_keywords:
                continue
            if anchor.line_number in self.used_lines:
                self.stats["anchors_covered_by_previous_record"] += 1
                continue
            if self.output_limit_reached:
                self.stats["anchors_skipped_by_max_records"] += 1
                continue
            self.stats["merged_candidates"] += 1
            if anchor.severe_error:
                self.write_rejection(anchor.severe_error, [anchor])
                continue

            lower, upper = self._available_bounds(rows, anchor_index)
            selected = self._select_interval(rows, anchor_index, lower, upper)
            if selected is None:
                available = rows[lower : upper + 1]
                en_text, zh_text = join_rows(available)
                zh_count = count_han(zh_text)
                en_count = count_english_words(en_text)
                if en_count < 20:
                    reason = "english_words_below_minimum"
                elif zh_count < self.args.min_zh_chars:
                    reason = "zh_chars_below_minimum_after_context"
                    self.stats["chinese_below_100"] += 1
                elif count_han(anchor.zh_text) > self.args.max_zh_chars:
                    reason = "zh_chars_over_maximum"
                    self.stats["chinese_above_220"] += 1
                else:
                    reason = "cannot_fit_complete_sentences_within_maximum"
                    self.stats["chinese_above_220"] += 1
                self.write_rejection(
                    reason,
                    available,
                    details=f"zh_chars={zh_count}, en_words={en_count}",
                )
                continue

            en_text, zh_text = join_rows(selected)
            digest = pair_digest(en_text, zh_text)
            if digest in self.seen_pair_hashes:
                self.stats["exact_duplicates_removed"] += 1
                self.write_rejection("duplicate_parallel_pair", selected)
                self.used_lines.update(row.line_number for row in selected)
                continue

            matched = tuple(
                keyword
                for keyword in KEYWORD_ORDER
                if any(keyword in row.matched_keywords for row in selected)
            )
            subdomain = select_subdomain(matched)
            flags = quality_flags(en_text, zh_text)
            if "number_mismatch" in flags:
                self.stats["number_mismatches"] += 1
            if "length_ratio_outlier" in flags:
                self.stats["length_ratio_outliers"] += 1
            zh_count = count_han(zh_text)
            en_count = count_english_words(en_text)
            record = {
                "id": f"{ID_PREFIX}{digest[:20]}",
                "zh_text": zh_text,
                "en_text": en_text,
                "zh_char_count": zh_count,
                "en_word_count": en_count,
                "domain": DOMAIN,
                "subdomain": subdomain,
                "matched_keywords": list(matched),
                "data_origin": "public_parallel",
                "dataset_name": DATASET_NAME,
                "document_id": selected[0].document_id,
                "start_sentence_id": selected[0].sentence_id,
                "end_sentence_id": selected[-1].sentence_id,
                "start_line": selected[0].line_number,
                "end_line": selected[-1].line_number,
                "source_pair_count": len(selected),
                "quality_status": "passed",
                "quality_flags": flags,
                "pair_sha256": digest,
            }
            self.cleaned_writer.write(record)
            self.seen_pair_hashes.add(digest)
            self.used_lines.update(row.line_number for row in selected)
            self.stats["final_valid_records"] += 1
            self.final_subdomain_counts[subdomain] += 1
            self.records_per_document[selected[0].document_id] += 1
            self.zh_lengths.append(zh_count)
            self.en_lengths.append(en_count)
            self._reservoir_sample(record)

    def _reservoir_sample(self, record: dict) -> None:
        accepted = self.stats["final_valid_records"]
        if len(self.random_samples) < 5:
            self.random_samples.append(record)
            return
        replacement = self.random.randrange(accepted)
        if replacement < 5:
            self.random_samples[replacement] = record


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
    }


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


def file_metadata(inspection: FileInspection, role: str) -> dict:
    return {
        "role": role,
        "path": inspection.path.resolve().as_posix(),
        "name": inspection.path.name,
        "size_bytes": inspection.size_bytes,
        "first_five_lines": list(inspection.samples[:5]),
    }


def open_text(path: Path) -> TextIO:
    return path.open("r", encoding="utf-8-sig", errors="strict", newline="")


def process_corpus(
    args: argparse.Namespace,
    english: FileInspection,
    chinese: FileInspection,
    ids: FileInspection,
    cleaner: UnpcFinanceCleaner,
) -> None:
    sentinel = object()
    current_document: str | None = None
    document_rows: list[AlignedPair] = []

    with open_text(english.path) as en_handle, open_text(chinese.path) as zh_handle, open_text(
        ids.path
    ) as id_handle:
        aligned: Iterator[tuple[str | object, str | object, str | object]] = itertools.zip_longest(
            en_handle, zh_handle, id_handle, fillvalue=sentinel
        )
        if args.sample:
            aligned = itertools.islice(aligned, args.sample)

        for line_number, (en_line, zh_line, id_line) in enumerate(aligned, start=1):
            if sentinel in (en_line, zh_line, id_line):
                cleaner.stats["line_count_mismatches"] += 1
                raise RuntimeError(
                    f"Input line counts differ at aligned row {line_number}; processing stopped"
                )
            cleaner.stats["raw_aligned_pairs"] += 1
            en_text = normalize_text(str(en_line).rstrip("\r\n"))
            zh_text = normalize_text(str(zh_line).rstrip("\r\n"))
            try:
                parsed_id = parse_id_line(str(id_line).rstrip("\r\n"))
                cleaner.stats["successfully_parsed_ids"] += 1
            except ValueError as error:
                if document_rows:
                    cleaner.process_document(document_rows)
                    document_rows = []
                    current_document = None
                cleaner.stats["id_parse_errors"] += 1
                cleaner.write_rejection(
                    "alignment_id_error",
                    [],
                    start_line=line_number,
                    end_line=line_number,
                    en_text=en_text,
                    zh_text=zh_text,
                    details=str(error),
                )
                continue

            keywords = match_keywords(en_text, zh_text)
            if keywords:
                cleaner.note_anchor(keywords)
            severe_error = validate_sentence_pair(en_text, zh_text)
            if severe_error in LANGUAGE_ERROR_REASONS:
                cleaner.stats["language_or_text_errors"] += 1
            pair = AlignedPair(
                line_number=line_number,
                document_id=parsed_id.document_id,
                sentence_id=parsed_id.sentence_id,
                en_text=en_text,
                zh_text=zh_text,
                matched_keywords=keywords,
                severe_error=severe_error,
            )
            if current_document is None:
                current_document = parsed_id.document_id
            if parsed_id.document_id != current_document:
                cleaner.process_document(document_rows)
                document_rows = []
                current_document = parsed_id.document_id
            document_rows.append(pair)

            if line_number % 500_000 == 0:
                logging.info(
                    "Processed %s pairs; anchors=%s; accepted=%s; rejected=%s",
                    f"{line_number:,}",
                    f"{cleaner.stats[ANCHOR_STAT_KEY]:,}",
                    f"{cleaner.stats['final_valid_records']:,}",
                    f"{cleaner.stats['rejected_records']:,}",
                )

    if document_rows:
        cleaner.process_document(document_rows)


def build_report(
    args: argparse.Namespace,
    english: FileInspection,
    chinese: FileInspection,
    ids: FileInspection,
    cleaner: UnpcFinanceCleaner,
    generated_at: str,
) -> dict:
    stats = cleaner.stats
    return {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "language_pair": "zh_en",
        "domain": DOMAIN,
        "generated_at": generated_at,
        "input_files": [
            file_metadata(english, "english_text"),
            file_metadata(chinese, "chinese_text"),
            file_metadata(ids, "alignment_ids"),
        ],
        "id_format": {
            "description": (
                "document_id followed by one or more en:<segment>:<part> and "
                "zh:<segment>:<part> references"
            ),
            "example": ids.samples[0],
        },
        "parameters": {
            "min_zh_chars": args.min_zh_chars,
            "target_zh_chars": args.target_zh_chars,
            "max_zh_chars": args.max_zh_chars,
            "context_window": args.context_window,
            "max_records": args.max_records,
            "sample": args.sample,
            "line_count_check_scope": f"first_{args.sample}_rows" if args.sample else "full_files",
        },
        "statistics": {
            "raw_aligned_pairs": stats["raw_aligned_pairs"],
            "successfully_parsed_ids": stats["successfully_parsed_ids"],
            "id_or_line_errors": stats["id_parse_errors"] + stats["line_count_mismatches"],
            "id_parse_errors": stats["id_parse_errors"],
            "line_count_mismatches": stats["line_count_mismatches"],
            ANCHOR_STAT_KEY: stats[ANCHOR_STAT_KEY],
            "keyword_hit_counts": dict(cleaner.keyword_counts),
            "subdomain_candidate_counts": dict(cleaner.subdomain_candidate_counts),
            "merged_candidates": stats["merged_candidates"],
            "chinese_below_100": stats["chinese_below_100"],
            "chinese_above_220": stats["chinese_above_220"],
            "language_or_text_errors": stats["language_or_text_errors"],
            "exact_duplicates_removed": stats["exact_duplicates_removed"],
            "number_mismatches": stats["number_mismatches"],
            "length_ratio_outliers": stats["length_ratio_outliers"],
            "final_valid_records": stats["final_valid_records"],
            "final_records_by_subdomain": dict(cleaner.final_subdomain_counts),
            "rejected_records": stats["rejected_records"],
            "rejection_reasons": dict(cleaner.rejection_reasons),
            "anchors_covered_by_previous_record": stats["anchors_covered_by_previous_record"],
            "anchors_skipped_by_max_records": stats["anchors_skipped_by_max_records"],
        },
        "length_statistics": {
            "zh_char_count": summarize_lengths(cleaner.zh_lengths),
            "en_word_count": summarize_lengths(cleaner.en_lengths),
        },
        "records_per_document": dict(cleaner.records_per_document),
        "random_samples": cleaner.random_samples,
    }


def log_identified_file(role: str, inspection: FileInspection) -> None:
    logging.info(
        "Identified %s: %s (%s bytes)",
        role,
        inspection.path.name,
        f"{inspection.size_bytes:,}",
    )
    for index, sample in enumerate(inspection.samples[:5], start=1):
        logging.info("  sample %d: %s", index, sample[:300])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        validate_args(args)
        input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
        english, chinese, ids = identify_input_files(input_dir.resolve())
        log_identified_file("English text", english)
        log_identified_file("Chinese text", chinese)
        log_identified_file("alignment IDs", ids)
        parsed_example = parse_id_line(ids.samples[0])
        logging.info(
            "ID format: document_id=%s; sentence_id=%s",
            parsed_example.document_id,
            parsed_example.sentence_id,
        )

        generated_at = utc_now()
        metadata = {
            "schema_version": 1,
            "language_pair": "zh_en",
            "dataset_name": DATASET_NAME,
            "domain": DOMAIN,
            "generated_at": generated_at,
        }
        cleaned_writer = AtomicJsonArrayWriter(
            output_dir / "cleaned" / CLEANED_FILENAME, metadata
        )
        rejected_writer = AtomicJsonArrayWriter(
            output_dir / "rejected" / REJECTED_FILENAME, metadata
        )
        cleaned_writer.open()
        rejected_writer.open()
        cleaner = UnpcFinanceCleaner(args, cleaned_writer, rejected_writer)
        try:
            process_corpus(args, english, chinese, ids, cleaner)
            cleaned_writer.finish()
            rejected_writer.finish()
            report = build_report(args, english, chinese, ids, cleaner, generated_at)
            report_path = output_dir / REPORT_FILENAME
            write_json_atomic(report_path, report)
            cleaned_writer.commit()
            rejected_writer.commit()
        except Exception:
            cleaned_writer.abort()
            rejected_writer.abort()
            raise

        logging.info(
            "Completed: raw=%s anchors=%s accepted=%s rejected=%s",
            f"{cleaner.stats['raw_aligned_pairs']:,}",
            f"{cleaner.stats[ANCHOR_STAT_KEY]:,}",
            f"{cleaner.stats['final_valid_records']:,}",
            f"{cleaner.stats['rejected_records']:,}",
        )
        if (
            not args.sample
            and REQUIRED_MIN_RECORDS
            and cleaner.stats["final_valid_records"] < REQUIRED_MIN_RECORDS
        ):
            logging.error(
                "Minimum output not reached: accepted=%s required=%s",
                f"{cleaner.stats['final_valid_records']:,}",
                f"{REQUIRED_MIN_RECORDS:,}",
            )
            return 2
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        logging.error("UNPC cleaning failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
