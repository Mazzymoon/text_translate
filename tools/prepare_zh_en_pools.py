#!/usr/bin/env python3

"""Prepare the merged Chinese-English pools for later final selection.

This stage deliberately stops before quantity selection, final-domain assignment,
translation-direction assignment, and CSV export.  It validates and normalizes
the three domain pools, scores every hard-valid pair against all three domains,
and creates one SHA256-keyed global pool while retaining every distinct source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "final" / "zh_en" / "pools"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "final" / "zh_en" / "intermediate"
DOMAINS = ("education", "technology", "finance")
TARGET_PER_DOMAIN = 10_000

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,]\d+)*(?:%|％)?")
WHITESPACE_RE = re.compile(r"\s+")

ZERO_WIDTH_CHARACTERS = {
    "\ufeff",  # BOM / zero-width no-break space
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
}

# These tokens are evidence of UTF-8/GBK or UTF-8/Windows-1252 mojibake.  A
# replacement character or NUL is always severe; weaker tokens require at
# least two hits so legitimate isolated CJK characters are not rejected.
STRONG_CORRUPTION_MARKERS = ("\ufffd", "\x00", "锟斤拷", "鏁欏笀")
WEAK_CORRUPTION_MARKERS = (
    "鈥?",
    "鈥",
    "銆?",
    "锛?",
    "涓€",
    "脙",
    "脗",
    "芒鈧",
)


def english_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


EDUCATION_CONCEPTS: dict[str, dict[str, Any]] = {
    "education": {
        "en": english_pattern(r"\beducation\b"),
        "zh": ("教育",),
        "surfaces": ("education", "教育"),
    },
    "educational": {
        "en": english_pattern(r"\beducational\b"),
        "zh": (),
        "surfaces": ("educational",),
    },
    "school": {
        "en": english_pattern(r"\bschools?\b"),
        "zh": ("学校",),
        "surfaces": ("school", "学校"),
    },
    "student": {
        "en": english_pattern(r"\bstudents?\b"),
        "zh": ("学生",),
        "surfaces": ("student", "学生"),
    },
    "teacher": {
        "en": english_pattern(r"\bteachers?\b"),
        "zh": ("教师",),
        "surfaces": ("teacher", "教师"),
    },
    "teaching": {
        "en": english_pattern(r"\bteaching\b"),
        "zh": ("教学",),
        "surfaces": ("teaching", "教学"),
    },
    "university": {
        "en": english_pattern(r"\b(?:university|universities)\b"),
        "zh": ("大学",),
        "surfaces": ("university", "大学"),
    },
    "college": {
        "en": english_pattern(r"\bcolleges?\b"),
        "zh": ("学院",),
        "surfaces": ("college", "学院"),
    },
    "curriculum": {
        "en": english_pattern(r"\b(?:curriculum|curricula)\b"),
        "zh": ("课程",),
        "surfaces": ("curriculum", "课程"),
    },
    "classroom": {
        "en": english_pattern(r"\bclassrooms?\b"),
        "zh": ("课堂",),
        "surfaces": ("classroom", "课堂"),
    },
    "literacy": {
        "en": english_pattern(r"\bliteracy\b"),
        "zh": ("扫盲",),
        "surfaces": ("literacy", "扫盲"),
    },
    "higher_education": {
        "en": english_pattern(r"\bhigher\s+education\b"),
        "zh": ("高等教育",),
        "surfaces": ("higher education", "高等教育"),
    },
    "primary_education": {
        "en": english_pattern(r"\bprimary\s+education\b"),
        "zh": ("基础教育", "初等教育"),
        "surfaces": ("primary education", "基础教育", "初等教育"),
    },
    "secondary_education": {
        "en": english_pattern(r"\bsecondary\s+education\b"),
        "zh": ("中等教育",),
        "surfaces": ("secondary education", "中等教育"),
    },
    "vocational_education": {
        "en": english_pattern(r"\bvocational\s+education\b"),
        "zh": ("职业教育",),
        "surfaces": ("vocational education", "职业教育"),
    },
    "preschool": {
        "en": english_pattern(r"\bpre-?school\b"),
        "zh": ("学前教育",),
        "surfaces": ("preschool", "学前教育"),
    },
    "enrollment": {
        "en": english_pattern(r"\benrol(?:lment|ment)\b"),
        "zh": ("入学", "招生"),
        "surfaces": ("enrollment", "enrolment", "入学", "招生"),
    },
    "scholarship": {
        "en": english_pattern(r"\bscholarships?\b"),
        "zh": ("奖学金",),
        "surfaces": ("scholarship", "奖学金"),
    },
    "academic_institution": {
        "en": english_pattern(r"\bacademic\s+institutions?\b"),
        "zh": ("教育机构",),
        "surfaces": ("academic institution", "教育机构"),
    },
}


TECHNOLOGY_STRONG_CONCEPTS: dict[str, tuple[re.Pattern[str] | None, tuple[str, ...]]] = {
    "artificial_intelligence": (english_pattern(r"\bartificial\s+intelligence\b"), ("人工智能",)),
    "machine_learning": (english_pattern(r"\bmachine\s+learning\b"), ("机器学习",)),
    "deep_learning": (english_pattern(r"\bdeep\s+learning\b"), ("深度学习",)),
    "computer": (english_pattern(r"\bcomputers?\b"), ("计算机",)),
    "software": (english_pattern(r"\bsoftware\b"), ("软件",)),
    "hardware": (english_pattern(r"\bhardware\b"), ("硬件",)),
    "algorithm": (english_pattern(r"\balgorithms?\b"), ("算法",)),
    "database": (english_pattern(r"\bdatabases?\b"), ("数据库",)),
    "internet": (english_pattern(r"\binternet\b"), ("互联网",)),
    "cybersecurity": (english_pattern(r"\bcybersecurity\b"), ("网络安全",)),
    "semiconductor": (english_pattern(r"\bsemiconductors?\b"), ("半导体",)),
    "microchip": (english_pattern(r"\bmicrochips?\b"), ("芯片",)),
    "robotics": (english_pattern(r"\brobotics?\b"), ("机器人",)),
    "automation": (english_pattern(r"\bautomation\b"), ("自动化",)),
    "biotechnology": (english_pattern(r"\bbiotechnology\b"), ("生物技术",)),
    "telecommunications": (english_pattern(r"\btelecommunications?\b"), ("电信",)),
    "information_technology": (english_pattern(r"\binformation\s+technology\b"), ("信息技术",)),
    "digital_technology": (english_pattern(r"\bdigital\s+technology\b"), ("数字技术",)),
    "aerospace": (english_pattern(r"\baerospace\b"), ("航天",)),
}

TECHNOLOGY_GENERAL_CONCEPTS: dict[str, tuple[re.Pattern[str] | None, tuple[str, ...]]] = {
    "technology": (english_pattern(r"\btechnolog(?:y|ies)\b|\btechnical\b"), ("科技", "技术")),
    "science": (english_pattern(r"\bsciences?\b|\bscientific\b"), ("科学",)),
    "data": (english_pattern(r"\bdata\b"), ("数据",)),
    "network": (english_pattern(r"\bnetworks?\b"), ("网络",)),
    "communication": (english_pattern(r"\bcommunications?\b"), ("通信",)),
    "electronic": (english_pattern(r"\belectronics?\b"), ("电子",)),
    "engineering": (english_pattern(r"\bengineering\b"), ("工程",)),
    "energy": (english_pattern(r"\benergy\b"), ("能源",)),
    "satellite": (english_pattern(r"\bsatellites?\b"), ("卫星",)),
    "research": (english_pattern(r"\bresearch\b"), ("科研", "研究")),
    "innovation": (english_pattern(r"\binnovations?\b"), ("创新",)),
}


FINANCE_STRONG_CONCEPTS: dict[str, tuple[re.Pattern[str] | None, tuple[str, ...]]] = {
    "fiscal": (english_pattern(r"\bfiscal\b"), ("财政",)),
    "budget": (english_pattern(r"\bbudgets?\b"), ("预算",)),
    "tax": (english_pattern(r"\btax(?:es|ation)?\b"), ("税收", "税务")),
    "debt": (english_pattern(r"\bdebts?\b"), ("债务",)),
    "investment": (english_pattern(r"\binvest(?:ment|ments|or|ors)\b"), ("投资", "投资者")),
    "banking": (english_pattern(r"\bbanking\b|\bcentral\s+banks?\b"), ("银行业", "中央银行")),
    "currency": (english_pattern(r"\bcurrenc(?:y|ies)\b|\bmonetary\b"), ("货币", "汇率")),
    "securities": (english_pattern(r"\bsecurit(?:y|ies)\b"), ("证券",)),
    "bond": (english_pattern(r"\bbonds?\b"), ("债券",)),
    "stock": (english_pattern(r"\bstock\s+markets?\b|\bstocks?\b|\bshares?\b"), ("股票", "股市")),
    "credit_loan": (english_pattern(r"\bcredit\b|\bloans?\b|\blending\b"), ("信贷", "贷款", "融资")),
    "insurance": (english_pattern(r"\binsurance\b"), ("保险",)),
    "interest_rate": (english_pattern(r"\binterest\s+rates?\b"), ("利率",)),
    "inflation_gdp": (english_pattern(r"\binflation\b|\bGDP\b|\bgross\s+domestic\s+product\b"), ("通货膨胀", "国内生产总值")),
    "revenue_expenditure": (english_pattern(r"\brevenues?\b|\bexpenditures?\b"), ("财政收入", "财政支出")),
}

FINANCE_GENERAL_CONCEPTS: dict[str, tuple[re.Pattern[str] | None, tuple[str, ...]]] = {
    "finance": (english_pattern(r"\bfinance\b|\bfinancial\b"), ("金融",)),
    "economy": (english_pattern(r"\beconomic\b|\beconom(?:y|ies)\b"), ("经济",)),
    "trade": (english_pattern(r"\btrade\b"), ("贸易",)),
    "bank": (english_pattern(r"\bbanks?\b"), ("银行",)),
    "market": (english_pattern(r"\bmarkets?\b"), ("市场",)),
    "capital_fund": (english_pattern(r"\bcapital\b|\bfunds?\b"), ("资本", "基金", "资金")),
    "payment_income": (english_pattern(r"\bpayments?\b|\bincomes?\b|\bprofits?\b"), ("支付", "收入", "利润")),
    "prices": (english_pattern(r"\bprices?\b"), ("价格", "物价")),
    "accounting_audit": (english_pattern(r"\baccounting\b|\baudits?\b"), ("会计", "审计")),
    "international_trade": (english_pattern(r"\bimports?\b|\bexports?\b|\btariffs?\b|\bcommerce\b"), ("进口", "出口", "关税", "商业")),
}

FINANCE_EXCLUSION_PATTERNS: dict[str, tuple[re.Pattern[str], tuple[str, ...]]] = {
    "trade_union_only": (english_pattern(r"\btrade\s+unions?\b"), ("工会",)),
    "economic_rights_only": (
        english_pattern(r"\beconomic\s*,?\s*social\s+and\s+cultural\s+rights\b"),
        ("经济、社会和文化权利", "经济社会文化权利"),
    ),
    "financial_aid_or_compensation_only": (
        english_pattern(r"\bfinancial\s+(?:aid|assistance|support|compensation|reparations?)\b"),
        ("金融援助", "财政援助", "财政赔偿", "金融赔偿"),
    ),
    "geographical_bank_only": (
        english_pattern(r"\briver\s+banks?\b|\bbanks?\s+of\s+(?:the\s+)?(?:river|lake|canal)\b"),
        ("河岸", "江岸", "湖岸"),
    ),
}

FINANCE_TRADE_WORD_PATTERN = english_pattern(r"\btrade\b")
FINANCE_ECONOMY_WORD_PATTERN = english_pattern(r"\beconomic\b|\beconom(?:y|ies)\b")
FINANCE_WORD_PATTERN = english_pattern(r"\bfinance\b|\bfinancial\b")
FINANCE_BANK_WORD_PATTERN = english_pattern(r"\bbanks?\b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, normalize, score, hash, and deduplicate the three zh-en pools."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-zh-chars", type=int, default=100)
    parser.add_argument("--max-zh-chars", type=int, default=220)
    parser.add_argument("--min-en-words", type=int, default=20)
    parser.add_argument("--target-per-domain", type=int, default=TARGET_PER_DOMAIN)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform the complete analysis and print statistics without writing outputs.",
    )
    args = parser.parse_args(argv)
    if args.min_zh_chars < 1:
        parser.error("--min-zh-chars must be positive")
    if args.max_zh_chars < args.min_zh_chars:
        parser.error("--max-zh-chars must be at least --min-zh-chars")
    if args.min_en_words < 1:
        parser.error("--min-en-words must be positive")
    if args.target_per_domain < 1:
        parser.error("--target-per-domain must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    normalized: list[str] = []
    for character in text:
        if character in ZERO_WIDTH_CHARACTERS:
            continue
        if character in {"\u00a0", "\u3000", "\r", "\n", "\t", "\f", "\v"}:
            normalized.append(" ")
            continue
        category = unicodedata.category(character)
        if category in {"Cc", "Cf"}:
            continue
        if category == "Zs":
            normalized.append(" ")
        else:
            normalized.append(character)
    return WHITESPACE_RE.sub(" ", "".join(normalized)).strip()


def count_han(text: str) -> int:
    return len(HAN_RE.findall(text))


def count_latin(text: str) -> int:
    return len(LATIN_RE.findall(text))


def count_english_words(text: str) -> int:
    return len(ENGLISH_WORD_RE.findall(text))


def extract_numbers(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return {
        match.replace(",", "").replace("％", "%")
        for match in NUMBER_RE.findall(normalized)
    }


def corruption_evidence(text: str) -> list[str]:
    evidence = [marker for marker in STRONG_CORRUPTION_MARKERS if marker in text]
    if evidence:
        return evidence
    weak_hits: list[str] = []
    for marker in WEAK_CORRUPTION_MARKERS:
        weak_hits.extend([marker] * text.count(marker))
    return weak_hits if len(weak_hits) >= 2 else []


def pair_sha256(zh_text: str, en_text: str) -> str:
    return hashlib.sha256(f"{zh_text}\x1f{en_text.casefold()}".encode("utf-8")).hexdigest()


def validate_pair(
    zh_text: str,
    en_text: str,
    *,
    min_zh_chars: int,
    max_zh_chars: int,
    min_en_words: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    reasons: list[str] = []
    flags: list[str] = []
    zh_han = count_han(zh_text)
    zh_latin = count_latin(zh_text)
    en_han = count_han(en_text)
    en_latin = count_latin(en_text)
    en_words = count_english_words(en_text)
    ratio = zh_han / max(1, en_words)

    if not zh_text or not en_text:
        reasons.append("empty_text")
    if zh_text and en_text and zh_text.casefold() == en_text.casefold():
        reasons.append("identical_bilingual_text")
    if zh_text and zh_han / max(1, zh_han + zh_latin) < 0.50:
        reasons.append("chinese_language_error")
    if en_text and en_latin / max(1, en_latin + en_han) < 0.80:
        reasons.append("english_language_error")
    if zh_han < min_zh_chars:
        reasons.append("zh_chars_below_minimum")
    elif zh_han > max_zh_chars:
        reasons.append("zh_chars_above_maximum")
    if en_words < min_en_words:
        reasons.append("english_words_below_minimum")

    corrupt = corruption_evidence(f"{zh_text}\n{en_text}")
    if corrupt:
        reasons.append("severe_mojibake")
    if ratio < 0.25 or ratio > 6.0:
        reasons.append("extreme_length_ratio")
    elif ratio < 0.50 or ratio > 4.0:
        flags.append("length_ratio_outlier")
    if extract_numbers(zh_text) != extract_numbers(en_text):
        flags.append("number_mismatch")

    metrics = {
        "zh_char_count": zh_han,
        "zh_latin_count": zh_latin,
        "en_word_count": en_words,
        "en_han_count": en_han,
        "en_latin_count": en_latin,
        "zh_chars_per_en_word": round(ratio, 6),
        "corruption_evidence": corrupt,
    }
    return list(dict.fromkeys(reasons)), flags, metrics


def concept_matches(
    en_text: str,
    zh_text: str,
    concepts: dict[str, tuple[re.Pattern[str] | None, tuple[str, ...]]],
) -> tuple[set[str], set[str]]:
    matched_concepts: set[str] = set()
    matched_keywords: set[str] = set()
    for concept, (pattern, chinese_terms) in concepts.items():
        english_hit = bool(pattern and pattern.search(en_text))
        chinese_hits = [term for term in chinese_terms if term in zh_text]
        if english_hit or chinese_hits:
            matched_concepts.add(concept)
            if english_hit:
                matched_keywords.add(concept.replace("_", " "))
            matched_keywords.update(chinese_hits)
    return matched_concepts, matched_keywords


def score_education(en_text: str, zh_text: str) -> dict[str, Any]:
    concepts: set[str] = set()
    keywords: set[str] = set()
    for concept, config in EDUCATION_CONCEPTS.items():
        english_hit = bool(config["en"].search(en_text))
        chinese_hits = [term for term in config["zh"] if term in zh_text]
        if english_hit or chinese_hits:
            concepts.add(concept)
            if english_hit:
                keywords.add(config["surfaces"][0])
            keywords.update(chinese_hits)
    score = 3 * len(concepts)
    return {
        "score": score,
        "eligible": bool(concepts),
        "threshold_rule": "at_least_one_explicit_education_concept",
        "matched_keywords": sorted(keywords),
        "matched_concepts": sorted(concepts),
        "strong_concepts": sorted(concepts),
        "general_concepts": [],
        "excluded_matches": [],
    }


def score_technology(en_text: str, zh_text: str) -> dict[str, Any]:
    strong, strong_keywords = concept_matches(
        en_text, zh_text, TECHNOLOGY_STRONG_CONCEPTS
    )
    general, general_keywords = concept_matches(
        en_text, zh_text, TECHNOLOGY_GENERAL_CONCEPTS
    )
    eligible = bool(strong) or len(general) >= 2
    return {
        "score": 3 * len(strong) + len(general),
        "eligible": eligible,
        "threshold_rule": "one_strong_or_two_distinct_general_concepts",
        "matched_keywords": sorted(strong_keywords | general_keywords),
        "matched_concepts": sorted(strong | general),
        "strong_concepts": sorted(strong),
        "general_concepts": sorted(general),
        "excluded_matches": [],
    }


def count_pattern_matches(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


def apply_finance_exclusions(
    en_text: str,
    zh_text: str,
    general: set[str],
    keywords: set[str],
) -> list[str]:
    excluded: list[str] = []

    trade_total = count_pattern_matches(FINANCE_TRADE_WORD_PATTERN, en_text)
    trade_excluded = count_pattern_matches(FINANCE_EXCLUSION_PATTERNS["trade_union_only"][0], en_text)
    chinese_trade = "贸易" in zh_text
    chinese_union_only = "工会" in zh_text and not chinese_trade
    if "trade" in general and trade_total <= trade_excluded and (not chinese_trade or chinese_union_only):
        general.discard("trade")
        keywords.discard("trade")
        excluded.append("trade_union_only")

    economy_total = count_pattern_matches(FINANCE_ECONOMY_WORD_PATTERN, en_text)
    economy_excluded = count_pattern_matches(
        FINANCE_EXCLUSION_PATTERNS["economic_rights_only"][0], en_text
    )
    chinese_economy = "经济" in zh_text
    chinese_rights_only = any(
        phrase in zh_text
        for phrase in FINANCE_EXCLUSION_PATTERNS["economic_rights_only"][1]
    )
    if (
        "economy" in general
        and economy_total <= economy_excluded
        and (not chinese_economy or chinese_rights_only)
    ):
        general.discard("economy")
        keywords.discard("economy")
        excluded.append("economic_rights_only")

    finance_total = count_pattern_matches(FINANCE_WORD_PATTERN, en_text)
    finance_excluded = count_pattern_matches(
        FINANCE_EXCLUSION_PATTERNS["financial_aid_or_compensation_only"][0], en_text
    )
    chinese_finance = "金融" in zh_text
    chinese_aid_only = any(
        phrase in zh_text
        for phrase in FINANCE_EXCLUSION_PATTERNS["financial_aid_or_compensation_only"][1]
    )
    if (
        "finance" in general
        and finance_total <= finance_excluded
        and (not chinese_finance or chinese_aid_only)
    ):
        general.discard("finance")
        keywords.discard("finance")
        excluded.append("financial_aid_or_compensation_only")

    bank_total = count_pattern_matches(FINANCE_BANK_WORD_PATTERN, en_text)
    bank_excluded = count_pattern_matches(
        FINANCE_EXCLUSION_PATTERNS["geographical_bank_only"][0], en_text
    )
    if "bank" in general and bank_total and bank_total <= bank_excluded and "银行" not in zh_text:
        general.discard("bank")
        keywords.discard("bank")
        excluded.append("geographical_bank_only")

    return sorted(excluded)


def score_finance(en_text: str, zh_text: str) -> dict[str, Any]:
    strong, strong_keywords = concept_matches(en_text, zh_text, FINANCE_STRONG_CONCEPTS)
    general, general_keywords = concept_matches(en_text, zh_text, FINANCE_GENERAL_CONCEPTS)
    excluded = apply_finance_exclusions(en_text, zh_text, general, general_keywords)
    eligible = bool(strong) or len(general) >= 2
    return {
        "score": 3 * len(strong) + len(general),
        "eligible": eligible,
        "threshold_rule": "one_strong_or_two_distinct_general_or_context_concepts",
        "matched_keywords": sorted(strong_keywords | general_keywords),
        "matched_concepts": sorted(strong | general),
        "strong_concepts": sorted(strong),
        "general_concepts": sorted(general),
        "excluded_matches": excluded,
    }


DOMAIN_SCORERS = {
    "education": score_education,
    "technology": score_technology,
    "finance": score_finance,
}


def load_pool(path: Path, expected_domain: str) -> tuple[dict[str, Any], list[Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input pool does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse input pool {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise RuntimeError(f"Unrecognized pool structure in {path}: expected an object with records[]")
    if value.get("domain") != expected_domain:
        raise RuntimeError(
            f"Pool domain mismatch in {path}: expected {expected_domain!r}, got {value.get('domain')!r}"
        )
    return value, value["records"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_entry(record: dict[str, Any], source_domain: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "dataset_name": record.get("dataset_name"),
        "data_origin": record.get("data_origin"),
        "pool_record_id": record.get("pool_record_id"),
        "source_domain": source_domain,
    }
    if "provenance" in record:
        entry["provenance"] = deepcopy(record["provenance"])
    return entry


def hard_rejection_record(
    record: Any,
    source_domain: str,
    record_index: int,
    *,
    original_zh: Any = None,
    original_en: Any = None,
    normalized_zh: str = "",
    normalized_en: str = "",
    reasons: Iterable[str],
    flags: Iterable[str] = (),
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_domain": source_domain,
        "source_record_index": record_index,
        "rejection_reasons": list(reasons),
        "quality_flags": list(flags),
        "original_zh_text": original_zh,
        "original_en_text": original_en,
        "normalized_zh_text": normalized_zh,
        "normalized_en_text": normalized_en,
        "metrics": metrics or {},
    }
    if isinstance(record, dict):
        result["pool_record_id"] = record.get("pool_record_id")
        result["dataset_name"] = record.get("dataset_name")
        result["data_origin"] = record.get("data_origin")
        if "provenance" in record:
            result["provenance"] = deepcopy(record["provenance"])
    return result


def initialize_global_record(
    digest: str,
    zh_text: str,
    en_text: str,
    flags: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"zh_en_pair_{digest[:20]}",
        "pair_sha256": digest,
        "zh_text": zh_text,
        "en_text": en_text,
        "zh_char_count": metrics["zh_char_count"],
        "en_word_count": metrics["en_word_count"],
        "quality_flags": list(flags),
        "candidate_domains": [],
        "source_domains": [],
        "domain_scores": {},
        "sources": [],
        "_source_keys": set(),
        "_occurrence_count": 0,
    }


def score_global_records(global_records: dict[str, dict[str, Any]]) -> None:
    for record in global_records.values():
        scores = {
            domain: DOMAIN_SCORERS[domain](record["en_text"], record["zh_text"])
            for domain in DOMAINS
        }
        record["domain_scores"] = scores
        record["candidate_domains"] = [
            domain for domain in DOMAINS if scores[domain]["eligible"]
        ]
        record["source_domains"] = sorted(record["source_domains"], key=DOMAINS.index)
        record["sources"].sort(
            key=lambda item: (
                str(item.get("dataset_name")),
                str(item.get("source_domain")),
                str(item.get("pool_record_id")),
                canonical_json(item.get("provenance")),
            )
        )


def public_global_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def compact_eligible_view(record: dict[str, Any], domain: str) -> dict[str, Any]:
    dataset_names = sorted(
        {source.get("dataset_name") for source in record["sources"] if source.get("dataset_name")}
    )
    score = record["domain_scores"][domain]
    return {
        "pair_sha256": record["pair_sha256"],
        "domain": domain,
        "domain_score": score["score"],
        "threshold_rule": score["threshold_rule"],
        "matched_keywords": score["matched_keywords"],
        "matched_concepts": score["matched_concepts"],
        "quality_flags": record["quality_flags"],
        "source_domains": record["source_domains"],
        "dataset_names": dataset_names,
    }


def soft_review_reasons(score: dict[str, Any], domain: str) -> list[str]:
    strong_count = len(score["strong_concepts"])
    general_count = len(score["general_concepts"])
    concept_count = len(score["matched_concepts"])
    reasons: list[str] = []
    if domain == "education":
        reasons.append("no_explicit_education_concept")
    elif domain == "technology":
        if concept_count == 0:
            reasons.append("no_technology_concept")
        elif strong_count == 0 and general_count == 1:
            reasons.append("only_one_general_technology_concept")
        else:
            reasons.append("technology_threshold_not_met")
    else:
        if concept_count == 0:
            reasons.append("no_finance_concept")
        elif strong_count == 0 and general_count == 1:
            reasons.append("only_one_general_finance_concept")
        else:
            reasons.append("finance_threshold_not_met")
        reasons.extend(f"excluded_{name}" for name in score["excluded_matches"])
    return list(dict.fromkeys(reasons))


def soft_review_record(record: dict[str, Any], domain: str) -> dict[str, Any]:
    score = record["domain_scores"][domain]
    reasons = soft_review_reasons(score, domain)
    return {
        "pair_sha256": record["pair_sha256"],
        "domain": domain,
        "review_reason": reasons[0],
        "review_reasons": reasons,
        "zh_text": record["zh_text"],
        "en_text": record["en_text"],
        "zh_char_count": record["zh_char_count"],
        "en_word_count": record["en_word_count"],
        "quality_flags": record["quality_flags"],
        "domain_score": score,
        "source_domains": record["source_domains"],
        "sources": record["sources"],
    }


def duplicate_statistics(
    valid_hashes_by_domain: dict[str, Counter[str]],
    global_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    within: dict[str, dict[str, int]] = {}
    for domain, counts in valid_hashes_by_domain.items():
        duplicate_counts = [count for count in counts.values() if count > 1]
        within[domain] = {
            "duplicate_groups": len(duplicate_counts),
            "extra_records_merged": sum(count - 1 for count in duplicate_counts),
        }

    cross_source_groups = 0
    cross_source_extra = 0
    cross_domain_groups = 0
    cross_domain_extra = 0
    for record in global_records.values():
        dataset_names = {
            source.get("dataset_name")
            for source in record["sources"]
            if source.get("dataset_name") is not None
        }
        if len(dataset_names) > 1:
            cross_source_groups += 1
            cross_source_extra += len(dataset_names) - 1
        if len(record["source_domains"]) > 1:
            cross_domain_groups += 1
            cross_domain_extra += len(record["source_domains"]) - 1

    return {
        "within_domain": within,
        "cross_source": {
            "duplicate_groups": cross_source_groups,
            "extra_dataset_associations": cross_source_extra,
        },
        "cross_domain": {
            "duplicate_groups": cross_domain_groups,
            "extra_domain_associations": cross_domain_extra,
        },
    }


def score_histogram(records: Iterable[dict[str, Any]], domain: str) -> dict[str, int]:
    counts = Counter(str(record["domain_scores"][domain]["score"]) for record in records)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def verify_prepared_data(
    global_records: dict[str, dict[str, Any]],
    eligible: dict[str, list[dict[str, Any]]],
    soft_review: dict[str, list[dict[str, Any]]],
) -> None:
    hashes = set(global_records)
    if len(hashes) != len(global_records):
        raise RuntimeError("Global unique pool contains duplicate SHA256 keys")
    forbidden = {"final_domain", "source_lang", "target_lang", "source_text", "target_text"}
    for digest, record in global_records.items():
        if pair_sha256(record["zh_text"], record["en_text"]) != digest:
            raise RuntimeError(f"SHA256 verification failed for {digest}")
        if forbidden & set(record):
            raise RuntimeError(f"Final-domain or direction fields leaked into {digest}")
        source_keys = {canonical_json(source) for source in record["sources"]}
        if len(source_keys) != len(record["sources"]):
            raise RuntimeError(f"Duplicate source entries remain for {digest}")
    for domain in DOMAINS:
        eligible_hashes = {item["pair_sha256"] for item in eligible[domain]}
        review_hashes = {item["pair_sha256"] for item in soft_review[domain]}
        if not eligible_hashes <= hashes or not review_hashes <= hashes:
            raise RuntimeError(f"{domain} view contains a SHA256 absent from the global pool")
        if eligible_hashes & review_hashes:
            raise RuntimeError(f"{domain} eligible and soft-review views overlap")


def build_outputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    input_dir = resolve_path(args.input_dir)
    generated_at = utc_now()
    input_hashes: dict[Path, str] = {}
    input_reports: dict[str, dict[str, Any]] = {}
    hard_rejected: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    hard_reasons: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    normalization_reasons: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    valid_hashes_by_domain: dict[str, Counter[str]] = {
        domain: Counter() for domain in DOMAINS
    }
    dataset_source_counts: Counter[str] = Counter()
    dataset_valid_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    global_records: dict[str, dict[str, Any]] = {}

    for domain in DOMAINS:
        path = input_dir / f"{domain}_pool.json"
        input_hashes[path] = file_sha256(path) if path.is_file() else ""
        outer, records = load_pool(path, domain)
        normalized_records = 0
        hard_valid_occurrences = 0

        for record_index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                rejection = hard_rejection_record(
                    record,
                    domain,
                    record_index,
                    reasons=("invalid_record_structure",),
                )
                hard_rejected[domain].append(rejection)
                hard_reasons[domain]["invalid_record_structure"] += 1
                continue

            dataset_name = record.get("dataset_name")
            dataset_source_counts[str(dataset_name) if dataset_name is not None else "<missing>"] += 1
            original_zh = record.get("zh_text")
            original_en = record.get("en_text")
            structure_reasons: list[str] = []
            if not isinstance(original_zh, str):
                structure_reasons.append("missing_or_non_string_zh_text")
            if not isinstance(original_en, str):
                structure_reasons.append("missing_or_non_string_en_text")
            if structure_reasons:
                rejection = hard_rejection_record(
                    record,
                    domain,
                    record_index,
                    original_zh=original_zh,
                    original_en=original_en,
                    reasons=structure_reasons,
                )
                hard_rejected[domain].append(rejection)
                hard_reasons[domain].update(structure_reasons)
                continue

            normalized_zh = normalize_text(original_zh)
            normalized_en = normalize_text(original_en)
            normalization_changed = normalized_zh != original_zh or normalized_en != original_en
            if normalization_changed:
                normalized_records += 1
                if normalized_zh != original_zh:
                    normalization_reasons[domain]["zh_text_changed"] += 1
                if normalized_en != original_en:
                    normalization_reasons[domain]["en_text_changed"] += 1

            reasons, flags, metrics = validate_pair(
                normalized_zh,
                normalized_en,
                min_zh_chars=args.min_zh_chars,
                max_zh_chars=args.max_zh_chars,
                min_en_words=args.min_en_words,
            )
            if reasons:
                rejection = hard_rejection_record(
                    record,
                    domain,
                    record_index,
                    original_zh=original_zh,
                    original_en=original_en,
                    normalized_zh=normalized_zh,
                    normalized_en=normalized_en,
                    reasons=reasons,
                    flags=flags,
                    metrics=metrics,
                )
                if normalized_zh and normalized_en:
                    rejection["pair_sha256"] = pair_sha256(normalized_zh, normalized_en)
                rejection["normalization_changed"] = normalization_changed
                hard_rejected[domain].append(rejection)
                hard_reasons[domain].update(reasons)
                continue

            digest = pair_sha256(normalized_zh, normalized_en)
            valid_hashes_by_domain[domain][digest] += 1
            hard_valid_occurrences += 1
            quality_flag_counts.update(flags)
            if dataset_name is not None:
                dataset_valid_counts[str(dataset_name)] += 1

            global_record = global_records.get(digest)
            if global_record is None:
                global_record = initialize_global_record(
                    digest, normalized_zh, normalized_en, flags, metrics
                )
                global_records[digest] = global_record
            else:
                global_record["quality_flags"] = sorted(
                    set(global_record["quality_flags"]) | set(flags)
                )
            global_record["_occurrence_count"] += 1
            if domain not in global_record["source_domains"]:
                global_record["source_domains"].append(domain)
            source = source_entry(record, domain)
            key = canonical_json(source)
            if key not in global_record["_source_keys"]:
                global_record["_source_keys"].add(key)
                global_record["sources"].append(source)

        input_reports[domain] = {
            "input_file": display_path(path),
            "input_file_sha256": input_hashes[path],
            "pool_schema_version": outer.get("schema_version"),
            "pool_stage": outer.get("stage"),
            "records_read": len(records),
            "normalization_modified_records": normalized_records,
            "normalization_changes": dict(normalization_reasons[domain]),
            "hard_valid_occurrences": hard_valid_occurrences,
            "hard_rejected_occurrences": len(hard_rejected[domain]),
            "hard_rejection_reasons": dict(hard_reasons[domain]),
        }

    score_global_records(global_records)

    eligible: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    soft_review: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    for record in global_records.values():
        for domain in DOMAINS:
            if record["domain_scores"][domain]["eligible"]:
                eligible[domain].append(compact_eligible_view(record, domain))
            elif domain in record["source_domains"]:
                soft_review[domain].append(soft_review_record(record, domain))

    for domain in DOMAINS:
        eligible[domain].sort(key=lambda item: (-item["domain_score"], item["pair_sha256"]))
        soft_review[domain].sort(key=lambda item: item["pair_sha256"])
        hard_rejected[domain].sort(
            key=lambda item: (item["source_record_index"], str(item.get("pool_record_id")))
        )

    verify_prepared_data(global_records, eligible, soft_review)

    duplicates = duplicate_statistics(valid_hashes_by_domain, global_records)
    dataset_unique_counts: Counter[str] = Counter()
    for record in global_records.values():
        names = {
            source.get("dataset_name")
            for source in record["sources"]
            if source.get("dataset_name") is not None
        }
        dataset_unique_counts.update(str(name) for name in names)

    domain_summary: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        soft_reason_counts: Counter[str] = Counter()
        for item in soft_review[domain]:
            soft_reason_counts.update(item["review_reasons"])
        domain_summary[domain] = {
            "eligible_unique_pairs": len(eligible[domain]),
            "soft_review_unique_pairs": len(soft_review[domain]),
            "surface_gap_to_target": max(0, args.target_per_domain - len(eligible[domain])),
            "surface_surplus_over_target": max(0, len(eligible[domain]) - args.target_per_domain),
            "score_histogram": score_histogram(global_records.values(), domain),
            "soft_review_reasons": dict(
                sorted(soft_reason_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }

    report = {
        "schema_version": 1,
        "stage": "prepared_unique_pool",
        "generated_at": generated_at,
        "dry_run": bool(args.dry_run),
        "parameters": {
            "input_dir": display_path(input_dir),
            "output_dir": display_path(resolve_path(args.output_dir)),
            "min_zh_chars": args.min_zh_chars,
            "max_zh_chars": args.max_zh_chars,
            "min_en_words": args.min_en_words,
            "target_per_domain": args.target_per_domain,
            "hard_length_ratio_range": [0.25, 6.0],
            "soft_length_ratio_range": [0.5, 4.0],
        },
        "input_pools": input_reports,
        "domain_summary": domain_summary,
        "duplicates": duplicates,
        "global_unique_pairs": len(global_records),
        "dataset_name_counts": {
            name: {
                "source_records_read": dataset_source_counts[name],
                "hard_valid_source_occurrences": dataset_valid_counts[name],
                "associated_global_unique_pairs": dataset_unique_counts[name],
            }
            for name in sorted(dataset_source_counts)
        },
        "quality_flag_counts": dict(quality_flag_counts),
        "scope_guarantees": {
            "quantity_completion_performed": False,
            "final_domain_assigned": False,
            "translation_direction_assigned": False,
            "final_domain_json_generated": False,
            "csv_generated": False,
            "input_pools_overwritten": False,
            "shortfall_is_failure": False,
        },
    }

    public_records = [
        public_global_record(global_records[digest]) for digest in sorted(global_records)
    ]
    outputs = {
        "unique": {
            "schema_version": 1,
            "language_pair": "zh_en",
            "stage": "prepared_unique_pool",
            "generated_at": generated_at,
            "records": public_records,
        },
        "eligible": {
            domain: {
                "schema_version": 1,
                "language_pair": "zh_en",
                "domain": domain,
                "stage": "domain_eligible_view",
                "generated_at": generated_at,
                "records": eligible[domain],
            }
            for domain in DOMAINS
        },
        "review": {
            domain: {
                "schema_version": 1,
                "language_pair": "zh_en",
                "domain": domain,
                "stage": "domain_soft_review",
                "generated_at": generated_at,
                "records": soft_review[domain],
            }
            for domain in DOMAINS
        },
        "rejected": {
            domain: {
                "schema_version": 1,
                "language_pair": "zh_en",
                "domain": domain,
                "stage": "hard_rejected",
                "generated_at": generated_at,
                "records": hard_rejected[domain],
            }
            for domain in DOMAINS
        },
        "report": report,
    }

    for path, before_hash in input_hashes.items():
        if not path.is_file() or file_sha256(path) != before_hash:
            raise RuntimeError(f"Input pool changed during preparation: {path}")

    runtime = {
        "global_records": global_records,
        "eligible": eligible,
        "soft_review": soft_review,
        "hard_rejected": hard_rejected,
    }
    return outputs, runtime


def write_json(path: Path, value: dict[str, Any]) -> None:
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


def write_outputs(args: argparse.Namespace, outputs: dict[str, Any]) -> None:
    output_dir = resolve_path(args.output_dir)
    destinations: list[tuple[Path, dict[str, Any]]] = [
        (output_dir / "zh_en_unique_pool.json", outputs["unique"]),
        (
            output_dir / "reports" / "zh_en_pool_preparation_report.json",
            outputs["report"],
        ),
    ]
    for domain in DOMAINS:
        destinations.extend(
            [
                (
                    output_dir / "eligible" / f"{domain}_eligible.json",
                    outputs["eligible"][domain],
                ),
                (
                    output_dir / "review" / f"{domain}_soft_review.json",
                    outputs["review"][domain],
                ),
                (
                    output_dir / "rejected" / f"{domain}_hard_rejected.json",
                    outputs["rejected"][domain],
                ),
            ]
        )
    for path, value in destinations:
        write_json(path, value)


def print_summary(outputs: dict[str, Any]) -> None:
    report = outputs["report"]
    print("\nInput pools")
    print(
        f"{'domain':<12} {'read':>8} {'normalized':>11} {'hard_valid':>11} {'hard_rejected':>14}"
    )
    for domain in DOMAINS:
        item = report["input_pools"][domain]
        print(
            f"{domain:<12} {item['records_read']:>8,} "
            f"{item['normalization_modified_records']:>11,} "
            f"{item['hard_valid_occurrences']:>11,} "
            f"{item['hard_rejected_occurrences']:>14,}"
        )

    print("\nDomain candidate views")
    print(f"{'domain':<12} {'eligible':>10} {'soft_review':>12} {'gap_to_10000':>13}")
    for domain in DOMAINS:
        item = report["domain_summary"][domain]
        print(
            f"{domain:<12} {item['eligible_unique_pairs']:>10,} "
            f"{item['soft_review_unique_pairs']:>12,} "
            f"{item['surface_gap_to_target']:>13,}"
        )

    duplicate = report["duplicates"]
    print("\nDuplicate summary")
    for domain in DOMAINS:
        item = duplicate["within_domain"][domain]
        print(
            f"{domain} within-domain: {item['duplicate_groups']:,} groups, "
            f"{item['extra_records_merged']:,} extra records merged"
        )
    print(
        "cross-source: "
        f"{duplicate['cross_source']['duplicate_groups']:,} groups, "
        f"{duplicate['cross_source']['extra_dataset_associations']:,} extra dataset associations"
    )
    print(
        "cross-domain: "
        f"{duplicate['cross_domain']['duplicate_groups']:,} groups, "
        f"{duplicate['cross_domain']['extra_domain_associations']:,} extra domain associations"
    )
    print(f"global unique pairs: {report['global_unique_pairs']:,}")

    print("\nHard-rejection reasons")
    for domain in DOMAINS:
        reasons = report["input_pools"][domain]["hard_rejection_reasons"]
        rendered = ", ".join(
            f"{reason}={count:,}"
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        )
        print(f"{domain}: {rendered or 'none'}")

    print("\nSoft-review reasons")
    for domain in DOMAINS:
        reasons = report["domain_summary"][domain]["soft_review_reasons"]
        rendered = ", ".join(
            f"{reason}={count:,}" for reason, count in reasons.items()
        )
        print(f"{domain}: {rendered or 'none'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs, _runtime = build_outputs(args)
        print_summary(outputs)
        if args.dry_run:
            print("\nDry run complete: no intermediate files were written.")
        else:
            write_outputs(args, outputs)
            print(f"\nPrepared outputs written to {resolve_path(args.output_dir)}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Pool preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
