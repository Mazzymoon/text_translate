#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DATASET_ROOT,
  zhEnCrawledDirectory,
  ZH_EN_DOMAINS,
  parseArguments,
  readJson,
  writeJsonAtomic,
} from "./lib/corpus.mjs";

const TRANSLATIONS_DIR = zhEnCrawledDirectory("translations");
const DEFAULT_EXPECTED_RECORDS = 148;
const CONTAMINATION_REASON = "prompt_leak_or_batch_alignment_contamination";
const PROMPT_LEAK_PATTERNS = [
  /You must retain an equal number/i,
  /Do not omit, summarize, explain/i,
  /only output the translation/i,
  /separators? in the translation/i,
  /<SEP\w*>/i,
];

function acceptedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.tencent_mt.json`);
}

function rejectedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.tencent_mt_rejected.json`);
}

function readArray(filePath) {
  const value = readJson(filePath);
  if (!Array.isArray(value)) throw new Error(`${filePath} 顶层必须是数组`);
  return value;
}

function hasPromptLeak(record) {
  return PROMPT_LEAK_PATTERNS.some((pattern) => pattern.test(record.target_text ?? ""));
}

function positiveInteger(value, name, fallback) {
  const parsed = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${name} 必须是正整数`);
  return parsed;
}

function uniqueIds(records, label) {
  const ids = new Set();
  for (const record of records) {
    if (!record.id || ids.has(record.id)) {
      throw new Error(`${label} 中存在缺失或重复 id：${record.id ?? "unknown"}`);
    }
    ids.add(record.id);
  }
  return ids;
}

function main() {
  const args = parseArguments(process.argv.slice(2));
  const expectedRecords = positiveInteger(
    args["expected-records"],
    "--expected-records",
    DEFAULT_EXPECTED_RECORDS,
  );
  const state = new Map();
  const contaminatedBatchIds = new Set();

  for (const domain of ZH_EN_DOMAINS) {
    const accepted = readArray(acceptedFile(domain));
    const rejected = readArray(rejectedFile(domain));
    uniqueIds(accepted, `${domain} accepted`);
    uniqueIds(rejected, `${domain} rejected`);
    state.set(domain, { accepted, rejected });
    for (const record of [...accepted, ...rejected]) {
      if (hasPromptLeak(record)) contaminatedBatchIds.add(record.provider_batch_id);
    }
  }

  contaminatedBatchIds.delete(undefined);
  contaminatedBatchIds.delete(null);
  const plan = [];
  for (const domain of ZH_EN_DOMAINS) {
    const { accepted, rejected } = state.get(domain);
    const moved = accepted.filter((record) =>
      contaminatedBatchIds.has(record.provider_batch_id),
    );
    const kept = accepted.filter(
      (record) => !contaminatedBatchIds.has(record.provider_batch_id),
    );
    const existingRejectedIds = new Set(rejected.map((record) => record.id));
    const marked = moved.map((record) => ({
      ...record,
      rejection_reason: CONTAMINATION_REASON,
      retranslation_required: true,
      contamination_detected_at: new Date().toISOString(),
    }));
    if (marked.some((record) => existingRejectedIds.has(record.id))) {
      throw new Error(`${domain}: accepted 与 rejected 之间存在重复 id，已停止`);
    }
    plan.push({ domain, accepted, rejected, kept, marked });
  }

  const movedTotal = plan.reduce((sum, item) => sum + item.marked.length, 0);
  console.table(
    plan.map((item) => ({
      domain: item.domain,
      accepted_before: item.accepted.length,
      marked_for_retranslation: item.marked.length,
      accepted_after: item.kept.length,
      rejected_after: item.rejected.length + item.marked.length,
    })),
  );
  console.log(
    `发现 ${contaminatedBatchIds.size} 个污染批次，将标记 ${movedTotal} 条 accepted 记录重新翻译。`,
  );

  if (movedTotal === 0) {
    console.log("没有新的污染记录需要移动；当前状态已经处理完成。 ");
    return;
  }
  if (movedTotal !== expectedRecords) {
    throw new Error(
      `安全检查失败：预计移动 ${expectedRecords} 条，实际为 ${movedTotal} 条；未修改文件`,
    );
  }
  if (!args.execute) {
    console.log("当前为预演模式，没有修改文件；确认后追加 --execute。 ");
    return;
  }

  // Write rejected first so an interrupted run cannot lose any translation.
  // A temporary duplicate is recoverable by rerunning this idempotent tool.
  for (const item of plan) {
    writeJsonAtomic(rejectedFile(item.domain), [...item.rejected, ...item.marked]);
    writeJsonAtomic(acceptedFile(item.domain), item.kept);
  }
  console.log(
    `已把 ${movedTotal} 条记录移入 rejected，并设置 retranslation_required=true；腾讯用量账本未修改。`,
  );
}

const isDirectExecution =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectExecution) {
  try {
    main();
  } catch (error) {
    console.error(`腾讯污染记录标记失败：${error.message}`);
    process.exitCode = 1;
  }
}
