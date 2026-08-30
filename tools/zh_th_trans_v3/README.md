# Qwen3-8B Teacher Chinese–Thai v3 pipeline

This directory builds a new Chinese–Thai v3 corpus without modifying the frozen v2 corpus. Qwen3-8B is used only as the teacher that generates Thai text; the planned student is Qwen3-4B and is not trained by this pipeline.

## Why v3 exists

The v2 corpus mixed public ALT pairs and NLLB-600M pseudo-labels. V3 instead starts from the reviewed Chinese side of the frozen 30,000-row Chinese–English corpus, samples three domains evenly, generates Thai once with Qwen3-8B, applies the existing v2 rule gate, and assigns every unique pair to exactly one direction.

The frozen file `dataset/final/zh_th/zh_th_clean_v2.csv` is never overwritten.

## Pipeline

1. `prepare_candidates.py` extracts 8,000 unique Chinese texts per domain using seed `20260830`.
2. `generate_teacher.py` performs deterministic BF16 Chinese-to-Thai generation and flushes every completed batch.
3. `audit_teacher_outputs.py` reuses `tools/zh_th_trans/quality_rules_v2.py`, adds target-short and compatible 20-gram repeat checks, and writes accepted/rejected views.
4. `build_final_dataset.py` selects 20,000 unique accepted pairs, balances the three domains as closely as integer counts allow, and assigns 10,000 pairs to each direction.

Rule filtering detects formal corruption and degeneracy; it is **not** complete semantic translation assurance. Human review or optional semantic QE is still required before treating every teacher output as a gold translation.

## Files

Inputs:

```text
dataset/final/zh_en/zh_en.csv
configs/zh_th_qwen3_8b_v3.json
```

Intermediate outputs:

```text
outputs/zh_th_qwen3_8b_v3/candidates_24000.jsonl
outputs/zh_th_qwen3_8b_v3/candidates_manifest.json
outputs/zh_th_qwen3_8b_v3/raw_teacher_generations.jsonl
outputs/zh_th_qwen3_8b_v3/raw_teacher_generation_errors.jsonl  # only if failures occur
outputs/zh_th_qwen3_8b_v3/audit_all.jsonl
outputs/zh_th_qwen3_8b_v3/accepted.jsonl
outputs/zh_th_qwen3_8b_v3/rejected.jsonl
outputs/zh_th_qwen3_8b_v3/audit_summary.json
```

Final outputs:

```text
dataset/final/zh_th/zh_th_qwen3_8b_v3.csv
dataset/final/zh_th/manifest_qwen3_8b_v3.json
```

## Server commands

Prepare candidates (no model is loaded):

```bash
cd /root/autodl-tmp/text_translate
python tools/zh_th_trans_v3/prepare_candidates.py
```

Generate with a local Qwen3-8B path:

```bash
python tools/zh_th_trans_v3/generate_teacher.py \
  --model-name-or-path /path/to/Qwen3-8B \
  --batch-size 4
```

Resume after an interruption:

```bash
python tools/zh_th_trans_v3/generate_teacher.py \
  --model-name-or-path /path/to/Qwen3-8B \
  --batch-size 4 \
  --resume
```

The generator reads existing `candidate_id` values, skips completed rows, rejects duplicate IDs, writes each batch immediately, and repairs only an interrupted partial final JSONL line. `--overwrite` must be explicitly supplied to start generation from zero.

Audit and build:

```bash
python tools/zh_th_trans_v3/audit_teacher_outputs.py
python tools/zh_th_trans_v3/build_final_dataset.py
```

Optional semantic QE filtering during the final build:

```bash
python tools/zh_th_trans_v3/build_final_dataset.py \
  --qe-score-file /path/to/qe_scores.jsonl \
  --min-qe-score 0.75
```

The QE file must map `candidate_id` to `semantic_qe_score`. No QE dependency is required when these options are omitted.

Run the complete guarded pipeline:

```bash
bash tools/zh_th_trans_v3/run_pipeline.sh /path/to/Qwen3-8B 4
```

Existing intermediate and final files are kept rather than deleted. If an audit must be intentionally rebuilt after raw generation changes, invoke it directly with `--overwrite`. The same applies to candidate and final builders.

## Quality review

Inspect `audit_summary.json` first, including reject reasons, domain availability, length distributions, repeat buckets and duplicate mappings. Then manually sample `accepted.jsonl`, especially long outputs and records carrying quality warnings. The fields `semantic_qe_score=null` and `semantic_status=not_evaluated` explicitly show that semantic QE has not yet been performed.

## Data leakage note

The final `pair_group_id` is computed from canonical Chinese and Thai text and does not include direction. If a pair is ever materialized in both directions later, both records will receive the same group and must stay in one train/validation/test split. In the v3 submission itself, every pair appears in only one direction.

