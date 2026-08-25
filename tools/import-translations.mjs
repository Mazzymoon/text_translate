#!/usr/bin/env node

import path from "node:path";
import {
  countHanCharacters,
  normalizeText,
  parseArguments,
  readJson,
  requireZhEnDomain,
  writeJson,
  zhEnDataFile,
} from "./lib/corpus.mjs";

const METHODS = new Set(["human", "google_mt", "llm_mt"]);

function main() {
  const args = parseArguments(process.argv.slice(2));
  const domain = requireZhEnDomain(args.domain);
  const cleanedFile = zhEnDataFile("cleaned", domain);
  if (!args.input) throw new Error("必须提供 --input <译文JSON路径>");
  const inputPath = path.resolve(args.input);
  const translations = readJson(inputPath);
  if (!Array.isArray(translations)) throw new Error("译文 JSON 顶层必须是数组");

  const cleaned = readJson(cleanedFile);
  const recordsById = new Map(cleaned.records.map((record) => [record.id, record]));
  const seenIds = new Set();
  const errors = [];
  let updated = 0;

  for (const item of translations) {
    if (!item.id || seenIds.has(item.id)) {
      errors.push(`${item.id ?? "unknown"}: id 缺失或在导入文件中重复`);
      continue;
    }
    seenIds.add(item.id);
    const record = recordsById.get(item.id);
    if (!record) {
      errors.push(`${item.id}: cleaned JSON 中不存在`);
      continue;
    }
    const targetText = normalizeText(item.target_text);
    const method = item.translation_method ?? "llm_mt";
    if (!targetText) errors.push(`${item.id}: target_text 为空`);
    if (!METHODS.has(method)) errors.push(`${item.id}: translation_method 非法`);
    if (record.target_lang === "zh-CN" && countHanCharacters(targetText) < 100) {
      errors.push(`${item.id}: 中文译文不足 100 个汉字`);
    }
    if (errors.some((error) => error.startsWith(`${item.id}:`))) continue;

    record.target_text = targetText;
    record.translation_method = method;
    record.status = "ready";
    const chineseText = record.source_lang === "zh-CN" ? record.source_text : targetText;
    record.zh_char_count = countHanCharacters(chineseText);
    record.quality.translated_at = new Date().toISOString();
    record.quality.review_status = item.review_status ?? "pending";
    record.quality.review_notes = item.review_notes ?? "";
    updated += 1;
  }

  if (errors.length > 0) {
    throw new Error(`导入失败，共 ${errors.length} 个问题：\n- ${errors.join("\n- ")}`);
  }
  cleaned.generated_at = new Date().toISOString();
  writeJson(cleanedFile, cleaned);
  console.log(`已导入 ${updated} 条译文：${inputPath}`);
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
