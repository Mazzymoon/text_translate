#!/usr/bin/env python3

"""Extract technology-domain Chinese-English pairs from UNPC.

This module configures the shared UNPC cleaner used by the finance and
education entry points.  Technology anchors apply a strong/general keyword
policy: one strong concept is sufficient, while general terms require at
least two distinct technology concepts.  English and Chinese translations of
the same term count as one concept, so ``data`` plus ``数据`` alone is not
enough to create an anchor.
"""

from __future__ import annotations

import re

import clean_unpc_finance as core


ENGLISH_STRONG_KEYWORDS = (
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "computer",
    "software",
    "hardware",
    "algorithm",
    "database",
    "internet",
    "cybersecurity",
    "semiconductor",
    "microchip",
    "robotics",
    "automation",
    "biotechnology",
    "telecommunications",
    "information technology",
    "digital technology",
    "aerospace",
)

ENGLISH_GENERAL_KEYWORDS = (
    "technology",
    "technical",
    "science",
    "scientific",
    "data",
    "network",
    "communication",
    "electronic",
    "engineering",
    "energy",
    "satellite",
    "research",
    "innovation",
)

CHINESE_STRONG_KEYWORDS = (
    "人工智能",
    "机器学习",
    "深度学习",
    "计算机",
    "软件",
    "硬件",
    "算法",
    "数据库",
    "互联网",
    "网络安全",
    "半导体",
    "芯片",
    "机器人",
    "自动化",
    "生物技术",
    "信息技术",
    "数字技术",
    "电信",
    "航天",
)

CHINESE_GENERAL_KEYWORDS = (
    "科技",
    "技术",
    "科学",
    "数据",
    "网络",
    "通信",
    "电子",
    "工程",
    "能源",
    "卫星",
    "科研",
    "研究",
    "创新",
)

ENGLISH_KEYWORDS = ENGLISH_STRONG_KEYWORDS + ENGLISH_GENERAL_KEYWORDS
CHINESE_KEYWORDS = CHINESE_STRONG_KEYWORDS + CHINESE_GENERAL_KEYWORDS
KEYWORD_ORDER = ENGLISH_KEYWORDS + CHINESE_KEYWORDS

ENGLISH_PATTERN_TEXT = {
    "artificial intelligence": r"\bartificial\s+intelligence\b",
    "machine learning": r"\bmachine\s+learning\b",
    "deep learning": r"\bdeep\s+learning\b",
    "computer": r"\bcomputers?\b",
    "software": r"\bsoftware\b",
    "hardware": r"\bhardware\b",
    "algorithm": r"\balgorithms?\b",
    "database": r"\bdatabases?\b",
    "internet": r"\binternet\b",
    "cybersecurity": r"\bcybersecurity\b",
    "semiconductor": r"\bsemiconductors?\b",
    "microchip": r"\bmicrochips?\b",
    "robotics": r"\brobotics\b",
    "automation": r"\bautomation\b",
    "biotechnology": r"\bbiotechnology\b",
    "telecommunications": r"\btelecommunications?\b",
    "information technology": r"\binformation\s+technology\b",
    "digital technology": r"\bdigital\s+technology\b",
    "aerospace": r"\baerospace\b",
    "technology": r"\btechnolog(?:y|ies)\b",
    "technical": r"\btechnical\b",
    "science": r"\bsciences?\b",
    "scientific": r"\bscientific\b",
    "data": r"\bdata\b",
    "network": r"\bnetworks?\b",
    "communication": r"\bcommunications?\b",
    "electronic": r"\belectronics?\b",
    "engineering": r"\bengineering\b",
    "energy": r"\benergy\b",
    "satellite": r"\bsatellites?\b",
    "research": r"\bresearch\b",
    "innovation": r"\binnovations?\b",
}

ENGLISH_PREFILTERS = {
    **{keyword: keyword.casefold() for keyword in ENGLISH_KEYWORDS},
    "computer": "computer",
    "algorithm": "algorithm",
    "database": "database",
    "semiconductor": "semiconductor",
    "microchip": "microchip",
    "telecommunications": "telecommunication",
    "technology": "technolog",
    "science": "science",
    "network": "network",
    "communication": "communication",
    "electronic": "electronic",
    "satellite": "satellite",
    "innovation": "innovation",
}

SURFACE_TO_CONCEPT = {
    "artificial intelligence": "artificial_intelligence",
    "人工智能": "artificial_intelligence",
    "machine learning": "machine_learning",
    "机器学习": "machine_learning",
    "deep learning": "deep_learning",
    "深度学习": "deep_learning",
    "computer": "computer",
    "计算机": "computer",
    "software": "software",
    "软件": "software",
    "hardware": "hardware",
    "硬件": "hardware",
    "algorithm": "algorithm",
    "算法": "algorithm",
    "database": "database",
    "数据库": "database",
    "internet": "internet",
    "互联网": "internet",
    "cybersecurity": "cybersecurity",
    "网络安全": "cybersecurity",
    "semiconductor": "semiconductor",
    "半导体": "semiconductor",
    "microchip": "microchip",
    "芯片": "microchip",
    "robotics": "robotics",
    "机器人": "robotics",
    "automation": "automation",
    "自动化": "automation",
    "biotechnology": "biotechnology",
    "生物技术": "biotechnology",
    "telecommunications": "telecommunications",
    "电信": "telecommunications",
    "information technology": "information_technology",
    "信息技术": "information_technology",
    "digital technology": "digital_technology",
    "数字技术": "digital_technology",
    "aerospace": "aerospace",
    "航天": "aerospace",
    "technology": "technology",
    "technical": "technology",
    "科技": "technology",
    "技术": "technology",
    "science": "science",
    "scientific": "science",
    "科学": "science",
    "data": "data",
    "数据": "data",
    "network": "network",
    "网络": "network",
    "communication": "communication",
    "通信": "communication",
    "electronic": "electronic",
    "电子": "electronic",
    "engineering": "engineering",
    "工程": "engineering",
    "energy": "energy",
    "能源": "energy",
    "satellite": "satellite",
    "卫星": "satellite",
    "research": "research",
    "科研": "research",
    "研究": "research",
    "innovation": "innovation",
    "创新": "innovation",
}

STRONG_CONCEPTS = {
    SURFACE_TO_CONCEPT[keyword]
    for keyword in ENGLISH_STRONG_KEYWORDS + CHINESE_STRONG_KEYWORDS
}
GENERAL_CONCEPTS = {
    SURFACE_TO_CONCEPT[keyword]
    for keyword in ENGLISH_GENERAL_KEYWORDS + CHINESE_GENERAL_KEYWORDS
}

SUBDOMAIN_KEYWORDS = {
    "ai_computing": {
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "computer",
        "software",
        "hardware",
        "algorithm",
        "database",
        "information technology",
        "digital technology",
        "人工智能",
        "机器学习",
        "深度学习",
        "计算机",
        "软件",
        "硬件",
        "算法",
        "数据库",
        "信息技术",
        "数字技术",
    },
    "communication_network": {
        "internet",
        "cybersecurity",
        "telecommunications",
        "network",
        "communication",
        "互联网",
        "网络安全",
        "电信",
        "网络",
        "通信",
    },
    "electronics_semiconductor": {
        "semiconductor",
        "microchip",
        "electronic",
        "半导体",
        "芯片",
        "电子",
    },
    "biotechnology": {"biotechnology", "生物技术"},
    "engineering_manufacturing": {
        "robotics",
        "automation",
        "engineering",
        "机器人",
        "自动化",
        "工程",
    },
    "energy_aerospace": {
        "energy",
        "satellite",
        "aerospace",
        "能源",
        "卫星",
        "航天",
    },
    "science_research": {
        "science",
        "scientific",
        "data",
        "research",
        "innovation",
        "科学",
        "数据",
        "科研",
        "研究",
        "创新",
    },
    "technology_general": {"technology", "technical", "科技", "技术"},
}

ENGLISH_KEYWORD_PATTERNS = {
    keyword: re.compile(ENGLISH_PATTERN_TEXT[keyword], re.IGNORECASE)
    for keyword in ENGLISH_KEYWORDS
}


def raw_keyword_matches(en_text: str, zh_text: str) -> tuple[str, ...]:
    lowered = en_text.casefold()
    matched = {
        keyword
        for keyword, pattern in ENGLISH_KEYWORD_PATTERNS.items()
        if ENGLISH_PREFILTERS[keyword] in lowered and pattern.search(en_text)
    }
    matched.update(keyword for keyword in CHINESE_KEYWORDS if keyword in zh_text)
    return tuple(keyword for keyword in KEYWORD_ORDER if keyword in matched)


def match_technology_keywords(en_text: str, zh_text: str) -> tuple[str, ...]:
    """Return an anchor only for one strong or two distinct general concepts."""

    matched = raw_keyword_matches(en_text, zh_text)
    concepts = {SURFACE_TO_CONCEPT[keyword] for keyword in matched}
    if concepts & STRONG_CONCEPTS:
        return matched
    if len(concepts & GENERAL_CONCEPTS) >= 2:
        return matched
    return ()


def configure_shared_cleaner() -> None:
    core.DOMAIN = "technology"
    core.ID_PREFIX = "unpc_technology_"
    core.ANCHOR_STAT_KEY = "technology_anchor_sentences"
    core.CLEANED_FILENAME = "technology_pairs.json"
    core.REJECTED_FILENAME = "technology_rejected.json"
    core.REPORT_FILENAME = "technology_cleaning_report.json"
    core.DEFAULT_MAX_RECORDS = 10000
    core.REQUIRED_MIN_RECORDS = 8581

    core.ENGLISH_KEYWORDS = ENGLISH_KEYWORDS
    core.CHINESE_KEYWORDS = CHINESE_KEYWORDS
    core.ENGLISH_KEYWORD_PATTERNS = ENGLISH_KEYWORD_PATTERNS
    core.ENGLISH_KEYWORD_PREFILTERS = ENGLISH_PREFILTERS
    core.KEYWORD_ORDER = KEYWORD_ORDER
    core.SUBDOMAIN_KEYWORDS = SUBDOMAIN_KEYWORDS
    core.SUBDOMAIN_PRIORITY = tuple(SUBDOMAIN_KEYWORDS)
    core.match_keywords = match_technology_keywords


def main() -> int:
    configure_shared_cleaner()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
