#!/usr/bin/env python3
"""Create a deduplicated Thai monolingual corpus from a Thai Wikipedia dump.

The compressed XML is streamed directly; it is never fully decompressed or
loaded into memory.  Cleaning is intentionally lightweight: retain main-space
article paragraphs, remove common MediaWiki markup, reject empty/non-Thai text,
and perform exact SHA-256 deduplication after Unicode/whitespace normalization.
There is no 100-character minimum.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "dataset"
    / "external"
    / "thwiki"
    / "thwiki-latest-pages-articles.xml.bz2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "dataset"
    / "external"
    / "thwiki"
    / "cleaned"
    / "th_monolingual.jsonl"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "dataset"
    / "external"
    / "thwiki"
    / "cleaning_report.json"
)
DATASET_NAME = "Thai Wikipedia"
DATA_ORIGIN = "public_monolingual"
LICENSE_NAME = "CC BY-SA"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SOURCE_BASE_URL = "https://th.wikipedia.org/wiki/"

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>/]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.IGNORECASE | re.DOTALL)
TAG_CONTENT_RE = re.compile(
    r"<(?:gallery|math|score|timeline|syntaxhighlight|source|code|poem)\b[^>]*>.*?"
    r"</(?:gallery|math|score|timeline|syntaxhighlight|source|code|poem)\s*>",
    re.IGNORECASE | re.DOTALL,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
FILE_LINK_RE = re.compile(
    r"\[\[(?:ไฟล์|ภาพ|file|image|หมวดหมู่|category):[^\]]*\]\]",
    re.IGNORECASE,
)
WIKI_LINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]+))?\]", re.IGNORECASE)
BARE_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
MAGIC_WORD_RE = re.compile(r"__(?:NOTOC|TOC|FORCETOC|NOEDITSECTION|NEWSECTIONLINK)__", re.IGNORECASE)
LIST_PREFIX_RE = re.compile(r"^\s*[*#:;]+\s*")
TABLE_LINE_RE = re.compile(r"^\s*(?:\{\||\|\}|\|-|[!|].*)$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
SPACE_RE = re.compile(r"[\t\f\v \u00A0\u2000-\u200B\u202F\u205F\u3000]+")
NEWLINE_RE = re.compile(r"\r\n?|\u2028|\u2029")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
EMPTY_MARKUP_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream, lightly clean, and deduplicate Thai Wikipedia monolingual text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--max-records",
        type=int,
        default=100_000,
        help="Maximum accepted records; 0 means no limit (default: 100000).",
    )
    parser.add_argument(
        "--min-thai-chars",
        type=int,
        default=10,
        help="Minimum Thai-script characters used only to discard trivial fragments; no 100-char rule (default: 10).",
    )
    parser.add_argument(
        "--min-thai-ratio",
        type=float,
        default=0.50,
        help="Minimum Thai-script share among letters (default: 0.50).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Stop after this many XML pages for a small test; 0 means unrestricted.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress after this many scanned pages (default: 10000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full parsing/statistics but do not write output or report.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def descendant_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def has_child(element: ET.Element, name: str) -> bool:
    return any(local_name(child.tag) == name for child in element)


def strip_balanced(text: str, opening: str, closing: str) -> str:
    """Remove nested MediaWiki template/table spans without recursive regex."""
    output: list[str] = []
    cursor = 0
    depth = 0
    while cursor < len(text):
        if text.startswith(opening, cursor):
            depth += 1
            cursor += len(opening)
            continue
        if depth and text.startswith(closing, cursor):
            depth -= 1
            cursor += len(closing)
            continue
        if depth == 0:
            output.append(text[cursor])
        cursor += 1
    return "".join(output)


def replace_wiki_link(match: re.Match[str]) -> str:
    content = match.group(1)
    if content.startswith(("#", ":")):
        return ""
    pieces = content.split("|")
    label = pieces[-1].strip()
    if "#" in label and len(pieces) == 1:
        label = label.split("#", 1)[0]
    return label


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = NEWLINE_RE.sub("\n", text)
    text = CONTROL_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_wikitext(value: str) -> str:
    text = NEWLINE_RE.sub("\n", value)
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = TAG_CONTENT_RE.sub("", text)
    text = strip_balanced(text, "{|", "|}")
    text = strip_balanced(text, "{{", "}}")
    text = FILE_LINK_RE.sub("", text)
    # A few passes resolve ordinary non-nested links after templates are gone.
    for _ in range(3):
        updated = WIKI_LINK_RE.sub(replace_wiki_link, text)
        if updated == text:
            break
        text = updated
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or "", text)
    text = BARE_URL_RE.sub("", text)
    text = MAGIC_WORD_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = html.unescape(text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if TABLE_LINE_RE.match(line):
            continue
        heading = HEADING_RE.match(line)
        if heading:
            # Headings alone are metadata, not monolingual text records.
            cleaned_lines.append("")
            continue
        line = LIST_PREFIX_RE.sub("", line)
        line = line.replace("'''", "").replace("''", "")
        line = re.sub(r"\{\{[^{}]*\}\}", "", line)
        line = re.sub(r"\[\[[^\[\]]*\]\]", "", line)
        line = normalize_text(line)
        cleaned_lines.append(line)
    return normalize_text("\n".join(cleaned_lines))


def paragraphs(cleaned_text: str) -> Iterator[str]:
    for block in re.split(r"\n\s*\n+", cleaned_text):
        # Consecutive nonblank lines normally belong to one Wiki paragraph/list block.
        text = normalize_text(" ".join(line for line in block.splitlines() if line.strip()))
        if text:
            yield text


def quality_reason(text: str, min_thai_chars: int, min_thai_ratio: float) -> tuple[str | None, int, float]:
    if not text:
        return "empty_after_cleaning", 0, 0.0
    if EMPTY_MARKUP_RE.fullmatch(text):
        return "symbols_only", 0, 0.0
    thai_count = len(THAI_RE.findall(text))
    if thai_count < min_thai_chars:
        return "too_few_thai_characters", thai_count, 0.0
    letter_count = len(LETTER_RE.findall(text))
    thai_ratio = thai_count / max(letter_count, 1)
    if thai_ratio < min_thai_ratio:
        return "not_thai_dominant", thai_count, thai_ratio
    if "{{" in text or "}}" in text or "[[" in text or "]]" in text:
        return "residual_wiki_markup", thai_count, thai_ratio
    return None, thai_count, thai_ratio


def iter_pages(input_path: Path) -> Iterator[ET.Element]:
    with bz2.open(input_path, "rb") as compressed:
        for _event, element in ET.iterparse(compressed, events=("end",)):
            if local_name(element.tag) == "page":
                yield element
                element.clear()


def source_url(title: str) -> str:
    return SOURCE_BASE_URL + quote(title.replace(" ", "_"), safe="")


class AtomicJsonlWriter:
    def __init__(self, output_path: Path, enabled: bool) -> None:
        self.output_path = output_path
        self.enabled = enabled
        self.temp_path: Path | None = None
        self.handle: Any = None

    def __enter__(self) -> "AtomicJsonlWriter":
        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self.output_path.name}.", suffix=".tmp", dir=self.output_path.parent
            )
            self.temp_path = Path(name)
            self.handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self.enabled:
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.enabled:
            return
        assert self.temp_path is not None and self.handle is not None
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            if exc_type is None:
                os.replace(self.temp_path, self.output_path)
            else:
                self.temp_path.unlink(missing_ok=True)
        except BaseException:
            self.temp_path.unlink(missing_ok=True)
            raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_record(
    *, title: str, page_id: str, revision_id: str, paragraph_index: int, text: str,
    thai_count: int, thai_ratio: float, text_hash: str
) -> dict[str, Any]:
    return {
        "id": f"thwiki_mono_{text_hash[:20]}",
        "language": "th",
        "text": text,
        "th_char_count": thai_count,
        "dataset_name": DATASET_NAME,
        "data_origin": DATA_ORIGIN,
        "title": title,
        "page_id": page_id,
        "revision_id": revision_id,
        "paragraph_index": paragraph_index,
        "source_url": source_url(title),
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "text_sha256": text_hash,
        "quality_flags": [],
        "quality_metrics": {"thai_letter_ratio": round(thai_ratio, 6)},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dump does not exist: {input_path}")
    if args.max_records < 0 or args.max_pages < 0:
        raise ValueError("--max-records and --max-pages cannot be negative")
    if args.min_thai_chars < 1:
        raise ValueError("--min-thai-chars must be at least 1")
    if not 0.0 <= args.min_thai_ratio <= 1.0:
        raise ValueError("--min-thai-ratio must be between 0 and 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")

    started_at = utc_now()
    counters: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    stopped_reason = "end_of_dump"

    with AtomicJsonlWriter(output_path, enabled=not args.dry_run) as writer:
        for page in iter_pages(input_path):
            counters["xml_pages_scanned"] += 1
            title = child_text(page, "title").strip()
            namespace = child_text(page, "ns").strip()
            page_id = child_text(page, "id").strip()
            revision_id = ""
            for child in page:
                if local_name(child.tag) == "revision":
                    revision_id = child_text(child, "id").strip()
                    break

            if namespace != "0":
                counters["non_main_namespace_pages"] += 1
            elif has_child(page, "redirect"):
                counters["redirect_pages"] += 1
            else:
                raw_text = descendant_text(page, "text")
                if not raw_text.strip():
                    counters["pages_without_text"] += 1
                else:
                    counters["main_article_pages"] += 1
                    cleaned = clean_wikitext(raw_text)
                    article_candidates = 0
                    for paragraph_index, text in enumerate(paragraphs(cleaned), start=1):
                        article_candidates += 1
                        counters["paragraph_candidates"] += 1
                        reason, thai_count, thai_ratio = quality_reason(
                            text, args.min_thai_chars, args.min_thai_ratio
                        )
                        if reason:
                            rejection_reasons[reason] += 1
                            continue
                        normalized = normalize_text(text)
                        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                        if text_hash in seen_hashes:
                            counters["duplicates_removed"] += 1
                            continue
                        seen_hashes.add(text_hash)
                        writer.write(
                            build_record(
                                title=title,
                                page_id=page_id,
                                revision_id=revision_id,
                                paragraph_index=paragraph_index,
                                text=normalized,
                                thai_count=thai_count,
                                thai_ratio=thai_ratio,
                                text_hash=text_hash,
                            )
                        )
                        counters["accepted_records"] += 1
                        if args.max_records and counters["accepted_records"] >= args.max_records:
                            stopped_reason = "max_records_reached"
                            break
                    if article_candidates == 0:
                        counters["pages_without_paragraphs"] += 1

            if counters["xml_pages_scanned"] % args.progress_every == 0:
                print(
                    f"scanned pages={counters['xml_pages_scanned']:,} "
                    f"articles={counters['main_article_pages']:,} "
                    f"accepted={counters['accepted_records']:,} "
                    f"duplicates={counters['duplicates_removed']:,}",
                    flush=True,
                )
            if stopped_reason == "max_records_reached":
                break
            if args.max_pages and counters["xml_pages_scanned"] >= args.max_pages:
                stopped_reason = "max_pages_reached"
                break

    report = {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "stage": "thwiki_monolingual_cleaning",
        "generated_at": utc_now(),
        "started_at": started_at,
        "input_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "dry_run": args.dry_run,
        "settings": {
            "max_records": args.max_records,
            "min_thai_chars": args.min_thai_chars,
            "min_thai_ratio": args.min_thai_ratio,
            "max_pages": args.max_pages,
            "requires_100_characters": False,
            "deduplication": "SHA256 of NFC and whitespace-normalized exact text",
        },
        "stopped_reason": stopped_reason,
        "counts": dict(counters),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "license": {"name": LICENSE_NAME, "url": LICENSE_URL},
    }
    if not args.dry_run:
        atomic_write_json(report_path, report)
    return report


def print_report(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print("\nThai Wikipedia cleaning summary")
    print(f"XML pages scanned: {counts.get('xml_pages_scanned', 0):,}")
    print(f"main article pages: {counts.get('main_article_pages', 0):,}")
    print(f"paragraph candidates: {counts.get('paragraph_candidates', 0):,}")
    print(f"accepted records: {counts.get('accepted_records', 0):,}")
    print(f"duplicates removed: {counts.get('duplicates_removed', 0):,}")
    print(f"stopped reason: {report['stopped_reason']}")
    print("rejection reasons:")
    for reason, count in report["rejection_reasons"].items():
        print(f"  {reason}: {count:,}")
    if report["dry_run"]:
        print("dry-run: no output or report file was written")
    else:
        print(f"output: {report['output_file']}")


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
        print_report(report)
        return 0
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"Thai Wikipedia cleaning failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
