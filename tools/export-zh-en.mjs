#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import {
  DATASET_ROOT,
  ZH_EN_DOMAINS,
  countHanCharacters,
  emptyZhEnDocument,
  normalizeText,
  parseArguments,
  readJson,
  serializeCsv,
  zhEnDataFile,
} from "./lib/corpus.mjs";

const OUTPUT_FILE = path.join(DATASET_ROOT, "final", "zh_en", "zh_en.csv");
const COLUMNS = [
  "source_lang",
  "target_lang",
  "source_text",
  "target_text",
  "zh_char_count",
  "domain",
  "translation_method",
];
const DOMAINS = ["education", "technology", "finance"];
const METHODS = new Set(["human", "google_mt", "llm_mt"]);

function validateReadyRecord(record) {
  const errors = [];
  const expectedTarget = record.source_lang === "zh-CN" ? "en" : "zh-CN";
  if (!["zh-CN", "en"].includes(record.source_lang)) errors.push("source_lang 非法");
  if (record.target_lang !== expectedTarget) errors.push("target_lang 与 source_lang 不匹配");
  if (!normalizeText(record.source_text)) errors.push("source_text 为空");
  if (!normalizeText(record.target_text)) errors.push("target_text 为空");
  if (!DOMAINS.includes(record.domain)) errors.push("domain 非法");
  if (!METHODS.has(record.translation_method)) errors.push("translation_method 非法");
  if (record.provenance?.rights_status === "prohibited") errors.push("来源禁止用于当前用途");

  const chineseText = record.source_lang === "zh-CN" ? record.source_text : record.target_text;
  const count = countHanCharacters(chineseText);
  if (count < 100) errors.push(`中文只有 ${count} 个汉字`);
  return { errors, count };
}

function main() {
  const args = parseArguments(process.argv.slice(2));
  const requireComplete = Boolean(args["require-complete"]);
  const readyRecords = ZH_EN_DOMAINS.flatMap((domain) =>
    readJson(zhEnDataFile("cleaned", domain), emptyZhEnDocument(domain)).records,
  ).filter((record) => record.status === "ready");
  const errors = [];
  const rows = [];
  const seen = new Set();

  for (const record of readyRecords) {
    const validation = validateReadyRecord(record);
    if (validation.errors.length > 0) {
      errors.push(`${record.id}: ${validation.errors.join("；")}`);
      continue;
    }
    const duplicateKey = [
      record.source_lang,
      record.target_lang,
      normalizeText(record.source_text),
      normalizeText(record.target_text),
    ].join("\u001F");
    if (seen.has(duplicateKey)) {
      errors.push(`${record.id}: 双语正文完全重复`);
      continue;
    }
    seen.add(duplicateKey);
    rows.push({
      source_lang: record.source_lang,
      target_lang: record.target_lang,
      source_text: normalizeText(record.source_text),
      target_text: normalizeText(record.target_text),
      zh_char_count: validation.count,
      domain: record.domain,
      translation_method: record.translation_method,
    });
  }

  rows.sort((left, right) =>
    [left.domain, left.source_lang, left.source_text]
      .join("|")
      .localeCompare([right.domain, right.source_lang, right.source_text].join("|")),
  );

  const distribution = new Map();
  for (const row of rows) {
    const key = `${row.domain} / ${row.source_lang}->${row.target_lang}`;
    distribution.set(key, (distribution.get(key) ?? 0) + 1);
  }
  if (requireComplete) {
    for (const domain of DOMAINS) {
      for (const direction of ["zh-CN->en", "en->zh-CN"]) {
        const key = `${domain} / ${direction}`;
        const count = distribution.get(key) ?? 0;
        if (count < 5001) errors.push(`${key} 至少应为 5001 条，实际为 ${count} 条`);
      }
    }
  }
  if (errors.length > 0) {
    throw new Error(`不能导出，共 ${errors.length} 个问题：\n- ${errors.join("\n- ")}`);
  }

  fs.writeFileSync(OUTPUT_FILE, serializeCsv(rows, COLUMNS), "utf8");

  console.table(
    [...distribution].map(([group, count]) => ({ group, rows: count })),
  );
  console.log(`已导出 ${rows.length} 条：${OUTPUT_FILE}`);
}

try {
  main();
} catch (error) {
  console.error(`导出失败：${error.message}`);
  process.exitCode = 1;
}
