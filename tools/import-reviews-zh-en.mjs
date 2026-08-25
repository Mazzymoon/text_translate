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

const REQUIRED_FIELDS = [
  "id",
  "target_text",
  "translation_method",
  "review_status",
  "review_notes",
];
const METHODS = new Set(["human", "google_mt", "llm_mt"]);
const REVIEW_STATUSES = new Set(["pending", "approved", "revised", "rejected"]);

function normalizeOptional(value) {
  return value === null || value === undefined ? "" : normalizeText(value);
}

function recomputeZhCount(record) {
  if (record.source_lang === "zh-CN") return countHanCharacters(record.source_text);
  if (record.target_lang === "zh-CN") return countHanCharacters(record.target_text);
  return null;
}

function ensureQuality(record) {
  if (!record.quality || typeof record.quality !== "object") {
    record.quality = {};
  }
  return record.quality;
}

function validateReadyTranslation(record, targetText, method, location) {
  const errors = [];
  if (!targetText) errors.push(`${location}: target_text is empty`);
  if (!METHODS.has(method)) errors.push(`${location}: invalid translation_method=${method}`);
  if (record.target_lang === "zh-CN" && countHanCharacters(targetText) < 100) {
    errors.push(`${location}: Chinese target_text has fewer than 100 Han characters`);
  }
  return errors;
}

function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input) throw new Error("Required: --input <review JSON path>");
  const inputPath = path.resolve(args.input);
  const dryRun = Boolean(args["dry-run"]);
  const reviewDocument = readJson(inputPath);
  const items = Array.isArray(reviewDocument) ? reviewDocument : reviewDocument?.records;
  if (!Array.isArray(items)) {
    throw new Error("Review JSON must be an array or an object with a records array");
  }
  if (items.length === 0) throw new Error("Review JSON is empty");
  const domain = requireZhEnDomain(
    args.domain ?? reviewDocument?.domain ?? items[0]?.domain,
  );
  const cleanedFile = zhEnDataFile("cleaned", domain);

  const cleaned = readJson(cleanedFile);
  const recordsById = new Map(cleaned.records.map((record) => [record.id, record]));
  const seenIds = new Set();
  const errors = [];
  const summary = new Map([
    ["pending", 0],
    ["approved", 0],
    ["revised", 0],
    ["rejected", 0],
  ]);

  for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
    const item = items[itemIndex];
    const location = `${inputPath}#records[${itemIndex}]`;
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      errors.push(`${location}: review item must be an object`);
      continue;
    }
    const missingFields = REQUIRED_FIELDS.filter(
      (field) => !Object.prototype.hasOwnProperty.call(item, field),
    );
    if (missingFields.length > 0) {
      errors.push(`${location}: missing required fields: ${missingFields.join(", ")}`);
      continue;
    }

    const id = normalizeOptional(item.id);
    const reviewStatus = normalizeOptional(item.review_status).toLowerCase() || "pending";

    if (!id) {
      errors.push(`${location}: id is empty`);
      continue;
    }
    if (seenIds.has(id)) {
      errors.push(`${location}: duplicated id=${id}`);
      continue;
    }
    seenIds.add(id);

    const record = recordsById.get(id);
    if (!record) {
      errors.push(`${location}: id does not exist in cleaned JSON: ${id}`);
      continue;
    }
    if (record.language_pair !== "zh_en") {
      errors.push(`${location}: record is not zh_en`);
      continue;
    }
    if (!REVIEW_STATUSES.has(reviewStatus)) {
      errors.push(`${location}: invalid review_status=${reviewStatus}`);
      continue;
    }

    const currentTarget = normalizeText(record.target_text);
    const targetText = normalizeOptional(item.target_text);
    const method = normalizeOptional(item.translation_method) || record.translation_method;
    const notes = normalizeOptional(item.review_notes);

    if (reviewStatus === "pending") {
      if (targetText !== currentTarget) {
        errors.push(`${location}: target_text changed but review_status is still pending`);
      }
      summary.set("pending", summary.get("pending") + 1);
      continue;
    }

    if (reviewStatus === "approved") {
      if (targetText !== currentTarget) {
        errors.push(`${location}: target_text changed; use review_status=revised`);
        continue;
      }
      errors.push(...validateReadyTranslation(record, targetText, method, location));
      if (errors.some((error) => error.startsWith(location))) continue;
    }

    if (reviewStatus === "revised") {
      errors.push(...validateReadyTranslation(record, targetText, method, location));
      if (errors.some((error) => error.startsWith(location))) continue;
    }

    const quality = ensureQuality(record);
    if (reviewStatus === "rejected") {
      record.target_text = "";
      record.translation_method = null;
      record.status = "needs_translation";
      record.zh_char_count =
        record.source_lang === "zh-CN" ? countHanCharacters(record.source_text) : null;
    } else {
      record.target_text = targetText;
      record.translation_method = method;
      record.status = "ready";
      record.zh_char_count = recomputeZhCount(record);
    }

    quality.review_status = reviewStatus;
    quality.review_notes = notes;
    quality.reviewed_at = new Date().toISOString();
    summary.set(reviewStatus, summary.get(reviewStatus) + 1);
  }

  if (errors.length > 0) {
    throw new Error(`Review import failed with ${errors.length} issue(s):\n- ${errors.join("\n- ")}`);
  }

  if (!dryRun) {
    cleaned.generated_at = new Date().toISOString();
    writeJson(cleanedFile, cleaned);
  }

  console.table([...summary].map(([status, rows]) => ({ status, rows })));
  console.log(`${dryRun ? "Checked" : "Imported"} ${seenIds.size} review rows: ${inputPath}`);
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
