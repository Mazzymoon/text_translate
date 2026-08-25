#!/usr/bin/env python3

"""Extract education-domain Chinese-English pairs from UNPC.

The finance cleaner owns the shared streaming, alignment, merging, validation,
deduplication, atomic-output, and reporting implementation.  This entry point
supplies only the education-domain configuration so both cleaners apply the
same structural and quality rules.
"""

from __future__ import annotations

import re

import clean_unpc_finance as core


ENGLISH_KEYWORDS = (
    "education",
    "educational",
    "school",
    "student",
    "teacher",
    "teaching",
    "university",
    "college",
    "curriculum",
    "classroom",
    "literacy",
    "higher education",
    "primary education",
    "secondary education",
    "vocational education",
    "preschool",
    "enrollment",
    "scholarship",
    "academic institution",
)

CHINESE_KEYWORDS = (
    "教育",
    "学校",
    "学生",
    "教师",
    "教学",
    "大学",
    "学院",
    "课程",
    "课堂",
    "扫盲",
    "高等教育",
    "基础教育",
    "初等教育",
    "中等教育",
    "职业教育",
    "学前教育",
    "入学",
    "招生",
    "奖学金",
    "教育机构",
)

# Common plural forms are accepted while the stored matched keyword remains
# the canonical keyword above.  Deliberately absent: learning, training, and
# academic on its own, because they are too broad for an education anchor.
ENGLISH_PATTERN_TEXT = {
    "education": r"\beducation\b",
    "educational": r"\beducational\b",
    "school": r"\bschools?\b",
    "student": r"\bstudents?\b",
    "teacher": r"\bteachers?\b",
    "teaching": r"\bteaching\b",
    "university": r"\b(?:university|universities)\b",
    "college": r"\bcolleges?\b",
    "curriculum": r"\b(?:curriculum|curricula)\b",
    "classroom": r"\bclassrooms?\b",
    "literacy": r"\bliteracy\b",
    "higher education": r"\bhigher\s+education\b",
    "primary education": r"\bprimary\s+education\b",
    "secondary education": r"\bsecondary\s+education\b",
    "vocational education": r"\bvocational\s+education\b",
    "preschool": r"\bpre-?school\b",
    "enrollment": r"\benrollment\b",
    "scholarship": r"\bscholarships?\b",
    "academic institution": r"\bacademic\s+institutions?\b",
}

ENGLISH_PREFILTERS = {
    **{keyword: keyword.casefold() for keyword in ENGLISH_KEYWORDS},
    "university": "universit",
    "curriculum": "curricul",
    "preschool": "school",
}

SUBDOMAIN_KEYWORDS = {
    "vocational_education": {"vocational education", "职业教育"},
    "preschool": {"preschool", "学前教育"},
    "teacher_education": {"teacher", "教师"},
    "literacy": {"literacy", "扫盲"},
    "curriculum_teaching": {
        "teaching",
        "curriculum",
        "classroom",
        "教学",
        "课程",
        "课堂",
    },
    "higher_education": {
        "higher education",
        "university",
        "college",
        "academic institution",
        "高等教育",
        "大学",
        "学院",
    },
    "basic_education": {
        "school",
        "student",
        "primary education",
        "secondary education",
        "学校",
        "学生",
        "基础教育",
        "初等教育",
        "中等教育",
    },
    "education_general": {
        "education",
        "educational",
        "enrollment",
        "scholarship",
        "教育",
        "入学",
        "招生",
        "奖学金",
        "教育机构",
    },
}


def configure_shared_cleaner() -> None:
    core.DOMAIN = "education"
    core.ID_PREFIX = "unpc_education_"
    core.ANCHOR_STAT_KEY = "education_anchor_sentences"
    core.CLEANED_FILENAME = "education_pairs.json"
    core.REJECTED_FILENAME = "education_rejected.json"
    core.REPORT_FILENAME = "education_cleaning_report.json"
    core.DEFAULT_MAX_RECORDS = 10000
    core.REQUIRED_MIN_RECORDS = 8626

    core.ENGLISH_KEYWORDS = ENGLISH_KEYWORDS
    core.CHINESE_KEYWORDS = CHINESE_KEYWORDS
    core.ENGLISH_KEYWORD_PATTERNS = {
        keyword: re.compile(ENGLISH_PATTERN_TEXT[keyword], re.IGNORECASE)
        for keyword in ENGLISH_KEYWORDS
    }
    core.ENGLISH_KEYWORD_PREFILTERS = ENGLISH_PREFILTERS
    core.KEYWORD_ORDER = ENGLISH_KEYWORDS + CHINESE_KEYWORDS
    core.SUBDOMAIN_KEYWORDS = SUBDOMAIN_KEYWORDS
    core.SUBDOMAIN_PRIORITY = tuple(SUBDOMAIN_KEYWORDS)


def main() -> int:
    configure_shared_cleaner()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
