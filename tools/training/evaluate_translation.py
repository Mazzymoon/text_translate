#!/usr/bin/env python3
"""Evaluate internal-test translations with BLEU, chrF and COMET."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIRECTIONS = ("zh-CN->en", "en->zh-CN", "zh-CN->th", "th->zh-CN")
BLEU_TOKENIZERS = {"en": "13a", "zh-CN": "zh", "th": "flores200"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--lora-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=16)
    parser.add_argument("--skip-comet", action="store_true")
    return parser.parse_args()


def read_predictions(path: Path, expected_variant: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "test":
                raise ValueError(f"{path}:{line_number} is not an internal-test prediction")
            if row.get("model_variant") != expected_variant:
                raise ValueError(f"{path}:{line_number} variant is not {expected_variant}")
            required = ("record_id", "direction", "source_text", "reference", "prediction", "target_lang")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} missing {missing}")
            if row["record_id"] in seen:
                raise ValueError(f"Duplicate record_id in {path}: {row['record_id']}")
            if not str(row["prediction"]).strip():
                raise ValueError(f"Empty prediction in {path}:{line_number}")
            seen.add(row["record_id"])
            rows.append(row)
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "direction", "count", "bleu", "chrf", "comet"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    if not args.base_predictions and not args.lora_predictions:
        raise ValueError("Provide --base-predictions and/or --lora-predictions")
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise RuntimeError("Install sacrebleu to compute BLEU and chrF") from exc

    variants: dict[str, list[dict[str, Any]]] = {}
    if args.base_predictions:
        variants["base"] = read_predictions(args.base_predictions.resolve(), "base")
    if args.lora_predictions:
        variants["lora"] = read_predictions(args.lora_predictions.resolve(), "lora")
    if len(variants) == 2:
        base_ids = {row["record_id"] for row in variants["base"]}
        lora_ids = {row["record_id"] for row in variants["lora"]}
        if base_ids != lora_ids:
            raise ValueError("Base and LoRA prediction files do not contain identical internal-test IDs")

    comet_model = None
    comet_device_count = 0
    if not args.skip_comet:
        try:
            import torch
            from comet import download_model, load_from_checkpoint
        except ImportError as exc:
            raise RuntimeError("Install unbabel-comet to compute COMET, or pass --skip-comet") from exc
        comet_path = Path(args.comet_model)
        checkpoint = str(comet_path) if comet_path.exists() else download_model(args.comet_model)
        comet_model = load_from_checkpoint(checkpoint)
        comet_device_count = 1 if torch.cuda.is_available() else 0

    detailed: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for variant, rows in variants.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["direction"]].append(row)
        missing = [direction for direction in EXPECTED_DIRECTIONS if not grouped.get(direction)]
        if missing:
            raise ValueError(f"{variant} predictions are missing directions: {missing}")
        direction_metrics: dict[str, Any] = {}
        for direction in EXPECTED_DIRECTIONS:
            items = grouped[direction]
            hypotheses = [item["prediction"] for item in items]
            references = [item["reference"] for item in items]
            target_lang = items[0]["target_lang"]
            bleu_metric = BLEU(tokenize=BLEU_TOKENIZERS[target_lang])
            chrf_metric = CHRF(word_order=0)
            bleu = bleu_metric.corpus_score(hypotheses, [references])
            chrf = chrf_metric.corpus_score(hypotheses, [references])
            comet_score = None
            if comet_model is not None:
                comet_data = [
                    {"src": item["source_text"], "mt": item["prediction"], "ref": item["reference"]}
                    for item in items
                ]
                prediction = comet_model.predict(
                    comet_data,
                    batch_size=args.comet_batch_size,
                    gpus=comet_device_count,
                    progress_bar=True,
                )
                comet_score = float(prediction.system_score)
            metrics = {
                "count": len(items),
                "bleu": float(bleu.score),
                "bleu_signature": str(bleu_metric.get_signature()),
                "bleu_tokenizer": BLEU_TOKENIZERS[target_lang],
                "chrf": float(chrf.score),
                "chrf_signature": str(chrf_metric.get_signature()),
                "comet": comet_score,
            }
            direction_metrics[direction] = metrics
            csv_rows.append({"variant": variant, "direction": direction, **{key: metrics[key] for key in ("count", "bleu", "chrf", "comet")}})
        macro = {
            metric: statistics.fmean(direction_metrics[direction][metric] for direction in EXPECTED_DIRECTIONS)
            for metric in ("bleu", "chrf")
        }
        macro["comet"] = (
            statistics.fmean(direction_metrics[direction]["comet"] for direction in EXPECTED_DIRECTIONS)
            if comet_model is not None
            else None
        )
        detailed[variant] = {"directions": direction_metrics, "macro_average": macro}
        csv_rows.append({"variant": variant, "direction": "macro_average", "count": len(rows), **macro})

    comparison = None
    if "base" in detailed and "lora" in detailed:
        comparison = {"directions": {}, "macro_average": {}}
        for direction in EXPECTED_DIRECTIONS:
            comparison["directions"][direction] = {}
            for metric in ("bleu", "chrf", "comet"):
                base_value = detailed["base"]["directions"][direction][metric]
                lora_value = detailed["lora"]["directions"][direction][metric]
                comparison["directions"][direction][metric] = (
                    lora_value - base_value if base_value is not None and lora_value is not None else None
                )
        for metric in ("bleu", "chrf", "comet"):
            base_value = detailed["base"]["macro_average"][metric]
            lora_value = detailed["lora"]["macro_average"][metric]
            comparison["macro_average"][metric] = (
                lora_value - base_value if base_value is not None and lora_value is not None else None
            )

    report = {
        "schema_version": 1,
        "evaluation_scope": "internal_test_only",
        "comet_model": None if args.skip_comet else args.comet_model,
        "variants": detailed,
        "lora_minus_base": comparison,
    }
    output_dir = args.output_dir.resolve()
    atomic_write_json(output_dir / "evaluation_report.json", report)
    atomic_write_csv(output_dir / "evaluation_summary.csv", csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
