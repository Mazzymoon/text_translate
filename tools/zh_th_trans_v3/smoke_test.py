#!/usr/bin/env python3
"""Model-free end-to-end smoke test for the zh-th v3 data pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    from .common import (
        DEFAULT_CANDIDATES,
        DEFAULT_CANDIDATE_MANIFEST,
        DEFAULT_CONFIG,
        PROJECT_ROOT,
        PROMPT_TEMPLATE_SHA256,
        PROMPT_TEMPLATE_VERSION,
        count_thai,
        load_completed_candidate_ids,
        read_jsonl,
    )
    from .generate_teacher import pending_batches, repair_trailing_partial_jsonl
except ImportError:
    from common import (
        DEFAULT_CANDIDATES,
        DEFAULT_CANDIDATE_MANIFEST,
        DEFAULT_CONFIG,
        PROJECT_ROOT,
        PROMPT_TEMPLATE_SHA256,
        PROMPT_TEMPLATE_VERSION,
        count_thai,
        load_completed_candidate_ids,
        read_jsonl,
    )
    from generate_teacher import pending_batches, repair_trailing_partial_jsonl

from tools.training.prepare_sft_data import canonical_pair_id as sft_canonical_pair_id


def run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-cardinality",
        action="store_true",
        help="Also use the real 24k candidate file with mock Thai to verify the 20k allocator.",
    )
    parser.add_argument("--candidate-file", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def mock_teacher_row(row: dict[str, object], index: int, generation: dict[str, object]) -> dict[str, object]:
    thai_digits = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
    thai_base = (
        "ประเทศไทยมีประวัติศาสตร์และวัฒนธรรมที่หลากหลาย การศึกษาช่วยพัฒนาความรู้ "
        "การวิจัยสนับสนุนนวัตกรรม และความร่วมมือช่วยให้สังคมเติบโตอย่างมั่นคง หมายเลข "
    )
    target_text = thai_base + f"{index:06d}".translate(thai_digits)
    return {
        **row,
        "target_text": target_text,
        "thai_char_count": count_thai(target_text),
        "teacher_model": "/mock/Qwen3-8B",
        "translation_method": "qwen3_8b_teacher",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "generation_config": generation
        | {"eos_token_id": [151645, 151643], "pad_token_id": 151643},
        "generated_at": "2026-08-30T00:00:00Z",
    }


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="zh_th_v3_smoke_") as name:
        root = Path(name)
        source_csv = root / "zh_en.csv"
        config_path = root / "config.json"
        candidates = root / "candidates.jsonl"
        candidate_manifest = root / "candidate_manifest.json"
        raw = root / "raw.jsonl"
        audit_all = root / "audit_all.jsonl"
        accepted = root / "accepted.jsonl"
        rejected = root / "rejected.jsonl"
        audit_summary = root / "audit_summary.json"
        final_csv = root / "final.csv"
        final_manifest = root / "final_manifest.json"

        config = {
            "seed": 20260830,
            "candidate_count": 30,
            "final_pair_count": 24,
            "candidate_domain_targets": {
                "education": 10,
                "technology": 10,
                "finance": 10,
            },
            "final_direction_targets": {"zh-CN->th": 12, "th->zh-CN": 12},
            "candidate_quality": {
                "min_zh_chars": 100,
                "max_zh_chars": 400,
                "min_han_letter_ratio": 0.5,
            },
            "teacher_output_quality": {"min_thai_chars": 20, "max_repeat_score_20": 0.5},
            "generation": {
                "enable_thinking": False,
                "do_sample": False,
                "num_beams": 1,
                "max_input_length": 1024,
                "max_new_tokens": 768,
                "dtype": "bfloat16",
            },
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        fields = [
            "source_lang",
            "target_lang",
            "source_text",
            "target_text",
            "zh_char_count",
            "domain",
            "translation_method",
        ]
        chinese_base = (
            "这是一段用于验证中泰语料流水线的中文测试材料，内容涵盖公共服务、知识传播、"
            "社会发展、科研合作、人才培养和政策实施。文本需要保持完整、清晰并具有足够长度，"
            "以确认候选提取、教师生成结果审计、领域平衡选择和双向数据分配能够正确衔接。"
            "所有测试记录都只用于程序验证，不代表真实模型生成质量。"
        )
        with source_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for domain in ("education", "technology", "finance"):
                for index in range(12):
                    source_text = f"{chinese_base}领域{domain}测试编号{index:03d}。"
                    writer.writerow(
                        {
                            "source_lang": "zh-CN",
                            "target_lang": "en",
                            "source_text": source_text,
                            "target_text": f"English placeholder {domain} {index}",
                            "zh_char_count": 130,
                            "domain": domain,
                            "translation_method": "human",
                        }
                    )

        run(
            "tools/zh_th_trans_v3/prepare_candidates.py",
            "--input-csv",
            str(source_csv),
            "--output-file",
            str(candidates),
            "--manifest-file",
            str(candidate_manifest),
            "--config",
            str(config_path),
        )
        with raw.open("w", encoding="utf-8", newline="\n") as handle:
            for index, (_, row) in enumerate(read_jsonl(candidates), start=1):
                output = mock_teacher_row(row, index, config["generation"])
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

        run(
            "tools/zh_th_trans_v3/audit_teacher_outputs.py",
            "--input-file",
            str(raw),
            "--audit-all",
            str(audit_all),
            "--accepted-file",
            str(accepted),
            "--rejected-file",
            str(rejected),
            "--summary-file",
            str(audit_summary),
            "--config",
            str(config_path),
        )
        run(
            "tools/zh_th_trans_v3/build_final_dataset.py",
            "--accepted-file",
            str(accepted),
            "--output-csv",
            str(final_csv),
            "--manifest-file",
            str(final_manifest),
            "--candidate-manifest",
            str(candidate_manifest),
            "--audit-summary",
            str(audit_summary),
            "--config",
            str(config_path),
        )

        with final_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 24
        assert Counter(row["direction"] for row in rows) == {"zh-CN->th": 12, "th->zh-CN": 12}
        assert Counter(row["domain"] for row in rows) == {
            "education": 8,
            "technology": 8,
            "finance": 8,
        }
        assert len({row["record_id"] for row in rows}) == 24
        assert len({row["candidate_id"] for row in rows}) == 24
        assert len({row["pair_group_id"] for row in rows}) == 24

        resume_file = root / "resume.jsonl"
        first_three = [row for _, row in list(read_jsonl(raw))[:3]]
        with resume_file.open("w", encoding="utf-8", newline="\n") as handle:
            for row in first_three:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.write('{"candidate_id":"interrupted')
        assert repair_trailing_partial_jsonl(resume_file)
        assert load_completed_candidate_ids(resume_file) == {
            row["candidate_id"] for row in first_three
        }
        remaining_ids = {
            row["candidate_id"]
            for batch in pending_batches(
                candidates,
                {row["candidate_id"] for row in first_three},
                batch_size=4,
                max_records=0,
            )
            for row in batch
        }
        assert not remaining_ids.intersection(row["candidate_id"] for row in first_three)
        assert len(remaining_ids) == 27
        sample_pair = rows[0]
        zh_text = sample_pair["source_text"] if sample_pair["source_lang"] == "zh-CN" else sample_pair["target_text"]
        th_text = sample_pair["target_text"] if sample_pair["target_lang"] == "th" else sample_pair["source_text"]
        assert sample_pair["pair_group_id"] == sft_canonical_pair_id(
            "zh-CN", "th", zh_text, th_text
        )
        assert sample_pair["pair_group_id"] == sft_canonical_pair_id(
            "th", "zh-CN", th_text, zh_text
        )

    result: dict[str, object] = {
                "valid": True,
                "model_loaded": False,
                "candidate_rows": 30,
                "final_rows": 24,
                "direction_counts": {"zh-CN->th": 12, "th->zh-CN": 12},
                "domain_counts": {"education": 8, "technology": 8, "finance": 8},
                "resume_partial_line_repair": True,
                "resume_skips_completed_ids": True,
                "pair_group_matches_sft_logic": True,
                "unique_pair_group_ids": 24,
            }

    if args.full_cardinality:
        candidate_path = args.candidate_file.resolve()
        candidate_manifest_path = args.candidate_manifest.resolve()
        config_path = args.config.resolve()
        if not candidate_path.is_file() or not candidate_manifest_path.is_file():
            raise FileNotFoundError("Full-cardinality smoke test requires prepared real 24k candidates")
        full_config = json.loads(config_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="zh_th_v3_full_smoke_") as full_name:
            full_root = Path(full_name)
            raw = full_root / "raw.jsonl"
            audit_all = full_root / "audit_all.jsonl"
            accepted = full_root / "accepted.jsonl"
            rejected = full_root / "rejected.jsonl"
            summary = full_root / "summary.json"
            final_csv = full_root / "final.csv"
            final_manifest = full_root / "manifest.json"
            with raw.open("w", encoding="utf-8", newline="\n") as handle:
                for index, (_, row) in enumerate(read_jsonl(candidate_path), start=1):
                    handle.write(
                        json.dumps(
                            mock_teacher_row(row, index, full_config["generation"]),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            run(
                "tools/zh_th_trans_v3/audit_teacher_outputs.py",
                "--input-file",
                str(raw),
                "--audit-all",
                str(audit_all),
                "--accepted-file",
                str(accepted),
                "--rejected-file",
                str(rejected),
                "--summary-file",
                str(summary),
                "--config",
                str(config_path),
            )
            run(
                "tools/zh_th_trans_v3/build_final_dataset.py",
                "--accepted-file",
                str(accepted),
                "--output-csv",
                str(final_csv),
                "--manifest-file",
                str(final_manifest),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--audit-summary",
                str(summary),
                "--config",
                str(config_path),
            )
            with final_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                full_rows = list(csv.DictReader(handle))
            full_directions = Counter(row["direction"] for row in full_rows)
            full_domains = Counter(row["domain"] for row in full_rows)
            assert len(full_rows) == 20_000
            assert full_directions == {"zh-CN->th": 10_000, "th->zh-CN": 10_000}
            assert full_domains == {"education": 6_667, "technology": 6_667, "finance": 6_666}
            assert len({row["pair_group_id"] for row in full_rows}) == 20_000
            result["full_cardinality"] = {
                "candidate_rows": 24_000,
                "final_rows": 20_000,
                "direction_counts": dict(full_directions),
                "domain_counts": dict(full_domains),
                "unique_pair_group_ids": 20_000,
            }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
