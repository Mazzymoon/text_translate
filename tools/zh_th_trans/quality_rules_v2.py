#!/usr/bin/env python3
"""Shared quality rules for the Chinese--Thai v2 corpus.

The rules are deliberately source-agnostic. They classify a pair as accept,
review, or reject and expose stable exact/normalized keys for merge and audit.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib
from collections import Counter
from typing import Any


QUALITY_RULE_VERSION = "zh_th_quality_v2.1"
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
SPACE_RE = re.compile(r"[\s\u00A0\u202F\u3000]+")
LOOP_RE = re.compile(r"(.{2,40}?)\1{4,}")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
ENGLISH_BLOCK_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9'’_-]*(?:[ \t,.;:()/+&-]+|$)){5,}"
)
MOJIBAKE_RE = re.compile(
    r"(?:\ufffd|\u951f\u65a4\u62f7|\u93c1\u6b0f\u7b00|\u9225[?？]"
    r"|\u00c3.|\u00c2.|\u00e2(?:\u20ac|\u2122|\u0153|\u017e)"
    r"|\u00e0(?:\u00b8|\u00b9)|\u00ef\u00bf\u00bd)"
)


def is_han(char: str) -> bool:
    value = ord(char)
    return (
        0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
        or 0x20000 <= value <= 0x323AF
    )


def is_thai(char: str) -> bool:
    return 0x0E00 <= ord(char) <= 0x0E7F


def is_latin_letter(char: str) -> bool:
    return char.isalpha() and "LATIN" in unicodedata.name(char, "")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = CONTROL_RE.sub("", text)
    text = ZERO_WIDTH_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def normalized_side_key(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return "".join(
        char for char in normalized if char.isalnum() or is_han(char) or is_thai(char)
    )


def exact_pair_key(zh_text: str, th_text: str) -> str:
    payload = normalize_text(zh_text) + "\x1f" + normalize_text(th_text).casefold()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_pair_key(zh_text: str, th_text: str) -> str:
    payload = normalized_side_key(zh_text) + "\x1f" + normalized_side_key(th_text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_han(text: str) -> int:
    return sum(is_han(char) for char in text)


def _has_private_use(text: str) -> bool:
    return any(unicodedata.category(char) == "Co" for char in text)


def _foreign_letter_marks(text: str, *, side: str) -> Counter[str]:
    found: Counter[str] = Counter()
    for char in text:
        category = unicodedata.category(char)
        if category[0] not in {"L", "M"}:
            continue
        allowed = (
            (side == "zh" and (is_han(char) or is_latin_letter(char)))
            or (side == "th" and (is_thai(char) or is_latin_letter(char)))
        )
        if not allowed:
            name = unicodedata.name(char, "UNKNOWN")
            script = name.split(" ", 1)[0] if name else "UNKNOWN"
            found[script] += 1
    return found


def _script_metrics(text: str) -> dict[str, Any]:
    letter_marks = [char for char in text if unicodedata.category(char)[0] in {"L", "M"}]
    han = sum(is_han(char) for char in letter_marks)
    thai = sum(is_thai(char) for char in letter_marks)
    latin = sum(is_latin_letter(char) for char in letter_marks)
    denominator = max(len(letter_marks), 1)
    return {
        "letter_mark_count": len(letter_marks),
        "han_count": han,
        "thai_count": thai,
        "latin_count": latin,
        "han_ratio": round(han / denominator, 6),
        "thai_ratio": round(thai / denominator, 6),
        "latin_ratio": round(latin / denominator, 6),
    }


def _repetition_metrics(text: str) -> dict[str, Any]:
    compact = normalize_text(text)
    loop = LOOP_RE.search(compact)
    raw = compact.encode("utf-8")
    compression_ratio = len(zlib.compress(raw, 9)) / max(len(raw), 1)
    gram_text = SPACE_RE.sub("", compact)
    gram_count = max(0, len(gram_text) - 7)
    unique_ratio = (
        len({gram_text[index : index + 8] for index in range(gram_count)})
        / max(gram_count, 1)
    )
    return {
        "continuous_repeat_span": len(loop.group(0)) if loop else 0,
        "continuous_repeat_unit": loop.group(1) if loop else "",
        "unique_8gram_ratio": round(unique_ratio, 6),
        "char_8gram_repeat_score": round(1.0 - unique_ratio, 6),
        "compression_ratio": round(compression_ratio, 6),
    }


def _english_metrics(text: str) -> dict[str, Any]:
    words = [word.casefold() for word in ENGLISH_WORD_RE.findall(text)]
    blocks = ENGLISH_BLOCK_RE.findall(text)
    latin_letters = sum(is_latin_letter(char) for char in text)
    return {
        "latin_letter_count": latin_letters,
        "english_word_count": len(words),
        "longest_english_block": max((len(block) for block in blocks), default=0),
        "english_words": words,
    }


def assess_pair(zh_value: Any, th_value: Any) -> dict[str, Any]:
    """Return normalized text, decision, reasons, flags, and reproducible metrics."""

    raw_zh = str(zh_value or "")
    raw_th = str(th_value or "")
    zh_text = normalize_text(raw_zh)
    th_text = normalize_text(raw_th)
    reject: list[str] = []
    review: list[str] = []
    flags: list[str] = []

    if not zh_text:
        reject.append("empty_chinese")
    if not th_text:
        reject.append("empty_thai")
    if CONTROL_RE.search(raw_zh) or CONTROL_RE.search(raw_th):
        reject.append("control_character")
    if ZERO_WIDTH_RE.search(raw_zh) or ZERO_WIDTH_RE.search(raw_th):
        reject.append("invisible_format_character")
    if MOJIBAKE_RE.search(raw_zh) or MOJIBAKE_RE.search(raw_th):
        reject.append("mojibake")
    if _has_private_use(raw_zh) or _has_private_use(raw_th):
        reject.append("private_use_character")

    zh_script = _script_metrics(zh_text)
    th_script = _script_metrics(th_text)
    zh_foreign = _foreign_letter_marks(zh_text, side="zh")
    th_foreign = _foreign_letter_marks(th_text, side="th")
    if zh_text and zh_script["han_count"] == 0:
        reject.append("chinese_side_has_no_han")
    elif zh_script["han_ratio"] < 0.50:
        review.append("chinese_side_not_han_dominant")
    if any(is_thai(char) for char in zh_text):
        reject.append("chinese_side_contains_thai")
    if zh_foreign:
        review.append("chinese_side_foreign_letter_mark")
    if th_text and th_script["thai_count"] == 0:
        reject.append("thai_side_has_no_thai")
    elif th_script["thai_ratio"] < 0.50:
        reject.append("thai_side_not_thai_dominant")
    elif th_script["thai_ratio"] < 0.80:
        review.append("thai_script_ratio_below_0_80")
    if any(is_han(char) for char in th_text):
        reject.append("thai_side_contains_han")
    if th_foreign:
        reject.append("thai_side_foreign_letter_mark")

    repetition = _repetition_metrics(th_text)
    if repetition["continuous_repeat_span"] >= 40:
        reject.append("continuous_repetition")
    if (
        len(th_text) >= 100
        and repetition["unique_8gram_ratio"] < 0.20
        and repetition["compression_ratio"] < 0.20
    ):
        reject.append("severe_ngram_collapse")

    zh_english = _english_metrics(zh_text)
    th_english = _english_metrics(th_text)
    large_target_english = (
        th_english["latin_letter_count"] >= 100
        or th_english["english_word_count"] >= 20
        or th_english["longest_english_block"] >= 100
    )
    source_words = set(zh_english["english_words"])
    target_words = th_english["english_words"]
    supported_words = sum(word in source_words for word in target_words)
    support_ratio = supported_words / max(len(target_words), 1)
    source_supported_english = (
        large_target_english
        and zh_english["latin_letter_count"] >= 20
        and support_ratio >= 0.60
    )
    if large_target_english and not source_supported_english:
        review.append("large_abnormal_english")
    elif source_supported_english:
        flags.append("source_supported_large_english")
    if (
        zh_english["latin_letter_count"] >= 100
        or zh_english["english_word_count"] >= 20
        or zh_english["longest_english_block"] >= 100
    ):
        flags.append("chinese_source_contains_large_english")

    if zh_text and th_text and zh_text.casefold() == th_text.casefold():
        reject.append("source_target_identical")
    length_ratio = len(th_text) / max(len(zh_text), 1)
    if zh_text and th_text and (length_ratio < 0.10 or length_ratio > 8.0):
        reject.append("extreme_length_ratio")
    elif zh_text and th_text and (length_ratio < 0.25 or length_ratio > 4.0):
        flags.append("unusual_length_ratio")

    reject = list(dict.fromkeys(reject))
    review = list(dict.fromkeys(review))
    flags = list(dict.fromkeys(flags))
    decision = "reject" if reject else "review" if review else "accept"
    return {
        "rule_version": QUALITY_RULE_VERSION,
        "decision": decision,
        "reject_reasons": reject,
        "review_reasons": review,
        "quality_flags": flags,
        "zh_text": zh_text,
        "th_text": th_text,
        "pair_sha256": exact_pair_key(zh_text, th_text),
        "normalized_pair_sha256": normalized_pair_key(zh_text, th_text),
        "normalized_zh_key": normalized_side_key(zh_text),
        "normalized_th_key": normalized_side_key(th_text),
        "metrics": {
            "zh_script": zh_script,
            "th_script": th_script,
            "zh_foreign_letter_marks": dict(zh_foreign),
            "th_foreign_letter_marks": dict(th_foreign),
            "repetition": repetition,
            "zh_english": {key: value for key, value in zh_english.items() if key != "english_words"},
            "th_english": {key: value for key, value in th_english.items() if key != "english_words"},
            "target_english_source_support_ratio": round(support_ratio, 6),
            "target_to_source_codepoint_ratio": round(length_ratio, 6),
        },
    }
