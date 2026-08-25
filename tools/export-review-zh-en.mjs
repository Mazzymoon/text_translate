#!/usr/bin/env node

import path from "node:path";
import {
  DATASET_ROOT,
  zhEnCrawledDirectory,
  normalizeText,
  parseArguments,
  readJson,
  requireZhEnDomain,
  sha256,
  writeJson,
  zhEnDataFile,
} from "./lib/corpus.mjs";

const REVIEW_DIR = zhEnCrawledDirectory("review");
const REVIEW_STATUSES = new Set(["pending", "approved", "revised", "rejected"]);

function splitCsvList(value) {
  return String(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function directionOf(record) {
  return `${record.source_lang}->${record.target_lang}`;
}

function defaultOutputPath(domain, samplePerDirection) {
  const domainPart = domain ? `_${domain}` : "";
  const samplePart = samplePerDirection ? "_sample" : "";
  return path.join(REVIEW_DIR, `zh_en${domainPart}${samplePart}_review.json`);
}

function selectSample(records, samplePerDirection, seed) {
  if (!samplePerDirection) return records;
  const groups = new Map();

  for (const record of records) {
    const key = `${record.domain}\u001F${directionOf(record)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(record);
  }

  return [...groups.values()]
    .flatMap((group) =>
      group
        .sort((left, right) =>
          sha256(`${seed}\u001F${left.id}`).localeCompare(sha256(`${seed}\u001F${right.id}`)),
        )
        .slice(0, samplePerDirection),
    )
    .sort((left, right) =>
      [left.domain, directionOf(left), left.id]
        .join("|")
        .localeCompare([right.domain, directionOf(right), right.id].join("|")),
    );
}

function main() {
  const args = parseArguments(process.argv.slice(2));
  const domain = requireZhEnDomain(args.domain);
  const statuses =
    args.status && args.status !== "all" ? new Set(splitCsvList(args.status)) : null;
  const samplePerDirection = args["sample-per-direction"]
    ? Number(args["sample-per-direction"])
    : 0;
  const seed = String(args.seed ?? "review");
  const outputPath = path.resolve(
    args.output ?? defaultOutputPath(domain, samplePerDirection),
  );

  if (samplePerDirection && (!Number.isInteger(samplePerDirection) || samplePerDirection < 1)) {
    throw new Error("--sample-per-direction must be a positive integer");
  }
  if (statuses) {
    for (const status of statuses) {
      if (!REVIEW_STATUSES.has(status)) {
        throw new Error(`Unsupported review status filter: ${status}`);
      }
    }
  }

  const cleaned = readJson(zhEnDataFile("cleaned", domain));
  let records = cleaned.records.filter((record) => record.language_pair === "zh_en");
  if (domain) records = records.filter((record) => record.domain === domain);
  if (statuses) {
    records = records.filter((record) =>
      statuses.has(record.quality?.review_status ?? "pending"),
    );
  }
  records = records.filter((record) => record.status === "ready");
  records = selectSample(records, samplePerDirection, seed);

  const rows = records.map((record) => ({
    id: record.id,
    review_status: record.quality?.review_status ?? "pending",
    review_notes: record.quality?.review_notes ?? "",
    translation_method: record.translation_method,
    source_lang: record.source_lang,
    target_lang: record.target_lang,
    source_text: normalizeText(record.source_text),
    target_text: normalizeText(record.target_text),
    domain: record.domain,
    zh_char_count: record.zh_char_count,
    status: record.status,
    source: {
      url: record.provenance?.source_url ?? "",
      site: record.provenance?.source_site ?? "",
      title: record.provenance?.title ?? "",
      published_at: record.provenance?.published_at ?? "",
    },
  }));

  writeJson(outputPath, {
    schema_version: 1,
    language_pair: "zh_en",
    domain,
    generated_at: new Date().toISOString(),
    editable_fields: [
      "review_status",
      "review_notes",
      "target_text",
      "translation_method",
    ],
    allowed_review_statuses: [...REVIEW_STATUSES],
    records: rows,
  });

  const distribution = new Map();
  for (const row of rows) {
    const key = `${row.domain} / ${row.source_lang}->${row.target_lang}`;
    distribution.set(key, (distribution.get(key) ?? 0) + 1);
  }
  console.table([...distribution].map(([group, rows]) => ({ group, rows })));
  console.log(`Exported ${rows.length} review rows: ${outputPath}`);
}

try {
  main();
} catch (error) {
  console.error(`Review export failed: ${error.message}`);
  process.exitCode = 1;
}
