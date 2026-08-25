#!/usr/bin/env node

import {
  ZH_EN_DOMAINS,
  countEnglishWords,
  countHanCharacters,
  emptyZhEnDocument,
  normalizeText,
  parseArguments,
  readJson,
  requireZhEnDomain,
  sha256,
  writeJsonAtomic,
  zhEnDataFile,
} from "./lib/corpus.mjs";

function splitChineseSentences(text) {
  return normalizeText(text)
    .split(/\n+/)
    .flatMap((paragraph) => paragraph.match(/[^。！？!?；;]+[。！？!?；;]?/g) ?? [])
    .map(normalizeText)
    .filter((sentence) => countHanCharacters(sentence) >= 5);
}

function splitEnglishSentences(text) {
  return normalizeText(text)
    .split(/\n+/)
    .flatMap((paragraph) => paragraph.split(/(?<=[.!?])\s+(?=[A-Z0-9"'])/))
    .map(normalizeText)
    .filter((sentence) => countEnglishWords(sentence) >= 5);
}

function packUnits(units, measure, minimum, maximum) {
  const chunks = [];
  let current = "";

  for (const unit of units) {
    const separator = current ? " " : "";
    const combined = `${current}${separator}${unit}`;
    if (current && measure(current) >= minimum && measure(combined) > maximum) {
      chunks.push(current);
      current = unit;
    } else {
      current = combined;
    }
  }

  if (current) {
    if (measure(current) >= minimum) {
      chunks.push(current);
    } else if (chunks.length > 0 && measure(`${chunks.at(-1)} ${current}`) <= maximum * 1.25) {
      chunks[chunks.length - 1] = `${chunks.at(-1)} ${current}`;
    }
  }
  return chunks.map(normalizeText).filter((chunk) => measure(chunk) >= minimum);
}

function createChunks(record) {
  if (record.source_lang === "zh-CN") {
    return packUnits(splitChineseSentences(record.raw_text), countHanCharacters, 100, 180);
  }
  if (record.source_lang === "en") {
    return packUnits(splitEnglishSentences(record.raw_text), countEnglishWords, 60, 110);
  }
  return [];
}

function sanitizeChunk(value) {
  return normalizeText(value)
    .replace(/\s+>\s*(?=[A-Z])/g, " ")
    .replace(/([.!?])['’](?=\s+[A-Z])/g, "$1")
    .replace(/\bdestiny\s+(?=Education Secretary\b)/g, "destiny. ")
    .replace(/\s+[A-Z][^.!?]{0,120}\bsaid:\s*$/i, "")
    .replace(/[（(]?(?:https?:\/\/|www\.)[^\s）)]+[）)]?/gi, "")
    .replace(/website:\s+(?=Oak National Academy\b)/i, "website. ")
    .replace(/\s+项目办公室\s+.*$/u, "")
    .replace(/\s+附件[:：]\s*.*$/u, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function buildCandidate(rawRecord, text, chunkIndex, existing) {
  const sourceText = sanitizeChunk(text);
  const targetLanguage = rawRecord.source_lang === "zh-CN" ? "en" : "zh-CN";
  const id = `clean_${sha256(
    [rawRecord.domain, rawRecord.source_lang, sourceText].join("\u001F"),
  ).slice(0, 20)}`;
  const prior = existing.get(id);
  const targetText = prior?.target_text ?? "";
  const translationMethod = prior?.translation_method ?? null;
  const chineseText = rawRecord.source_lang === "zh-CN" ? sourceText : targetText;

  return {
    id,
    language_pair: "zh_en",
    source_lang: rawRecord.source_lang,
    target_lang: targetLanguage,
    source_text: sourceText,
    target_text: targetText,
    zh_char_count: chineseText ? countHanCharacters(chineseText) : null,
    domain: rawRecord.domain,
    translation_method: translationMethod,
    status: targetText && translationMethod ? "ready" : "needs_translation",
    provenance: {
      raw_record_id: rawRecord.id,
      source_id: rawRecord.source_id,
      chunk_index: chunkIndex,
      source_url: rawRecord.canonical_url,
      source_site: rawRecord.source_site,
      title: rawRecord.title,
      author: rawRecord.author,
      published_at: rawRecord.published_at,
      crawled_at: rawRecord.crawled_at,
      rights_status: rawRecord.rights_status,
      rights_note: rawRecord.rights_note,
      rights_evidence_url: rawRecord.rights_evidence_url,
      license_name: rawRecord.license_name,
      license_url: rawRecord.license_url,
      attribution: rawRecord.attribution,
      source_content_sha256: rawRecord.content_sha256,
    },
    quality: {
      source_sha256: sha256(sourceText),
      source_han_count: countHanCharacters(sourceText),
      source_word_count: countEnglishWords(sourceText),
      cleaned_at: prior?.quality?.cleaned_at ?? new Date().toISOString(),
      translated_at: prior?.quality?.translated_at,
      review_status: prior?.quality?.review_status ?? "pending",
      review_notes: prior?.quality?.review_notes ?? "",
      reviewed_at: prior?.quality?.reviewed_at,
    },
  };
}

function roundRobinBySource(records) {
  const groups = new Map();
  for (const record of records) {
    const key = record.source_id ?? record.source_site ?? "unknown";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(record);
  }
  for (const group of groups.values()) {
    group.sort((left, right) =>
      [right.published_at, right.canonical_url]
        .join("|")
        .localeCompare([left.published_at, left.canonical_url].join("|")),
    );
  }

  const orderedGroups = [...groups.entries()].sort(([left], [right]) => left.localeCompare(right));
  const ordered = [];
  let index = 0;
  let added = true;
  while (added) {
    added = false;
    for (const [, group] of orderedGroups) {
      if (index < group.length) {
        ordered.push(group[index]);
        added = true;
      }
    }
    index += 1;
  }
  return ordered;
}

function loadOtherDomainTextKeys(domain) {
  const keys = new Set();
  for (const otherDomain of ZH_EN_DOMAINS.filter((candidate) => candidate !== domain)) {
    const document = readJson(
      zhEnDataFile("cleaned", otherDomain),
      emptyZhEnDocument(otherDomain),
    );
    for (const record of document.records) {
      keys.add(
        `${record.source_lang}\u001F${normalizeText(record.source_text).toLowerCase()}`,
      );
    }
  }
  return keys;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  const domain = requireZhEnDomain(args.domain);
  const maximumChunksPerPage = Number(args["max-chunks-per-page"] ?? 10);
  const requiredMinimum = args["require-min-per-direction"]
    ? Number(args["require-min-per-direction"])
    : 0;
  if (!Number.isInteger(maximumChunksPerPage) || maximumChunksPerPage < 1) {
    throw new Error("--max-chunks-per-page 必须是正整数");
  }
  if (!Number.isInteger(requiredMinimum) || requiredMinimum < 0) {
    throw new Error("--require-min-per-direction 必须是非负整数");
  }
  if (args["per-direction"]) {
    console.warn("--per-direction 已停用；现在会保留全部合格清洗记录。");
  }

  const rawFile = zhEnDataFile("raw", domain);
  const cleanedFile = zhEnDataFile("cleaned", domain);
  const raw = readJson(rawFile);
  const cleaned = readJson(cleanedFile, emptyZhEnDocument(domain));
  if (raw.domain !== domain) throw new Error(`${rawFile} 的 domain 与命令参数不一致`);
  if (cleaned.domain && cleaned.domain !== domain) {
    throw new Error(`${cleanedFile} 的 domain 与命令参数不一致`);
  }

  const existingById = new Map(cleaned.records.map((record) => [record.id, record]));
  const seenSourceTexts = loadOtherDomainTextKeys(domain);
  const candidates = [];
  const directionCounts = new Map([
    ["zh-CN->en", 0],
    ["en->zh-CN", 0],
  ]);

  const domainRawRecords = roundRobinBySource(
    raw.records.filter(
      (record) => record.domain === domain && ["zh-CN", "en"].includes(record.source_lang),
    ),
  );

  for (const rawRecord of domainRawRecords) {
    const chunks = createChunks(rawRecord).slice(0, maximumChunksPerPage);
    for (let index = 0; index < chunks.length; index += 1) {
      const sourceText = sanitizeChunk(chunks[index]);
      if (rawRecord.source_lang === "zh-CN" && countHanCharacters(sourceText) < 100) continue;
      if (rawRecord.source_lang === "en" && countEnglishWords(sourceText) < 60) continue;
      if (/\bDr\.\s+(?:It|This|We)\b/.test(sourceText)) continue;
      if (/[\w.-]+@[\w.-]+\.[A-Za-z]{2,}/.test(sourceText)) continue;
      const duplicateKey = `${rawRecord.source_lang}\u001F${sourceText.toLowerCase()}`;
      if (seenSourceTexts.has(duplicateKey)) continue;

      const candidate = buildCandidate(rawRecord, sourceText, index, existingById);
      seenSourceTexts.add(duplicateKey);
      candidates.push(candidate);
      const direction = `${candidate.source_lang}->${candidate.target_lang}`;
      directionCounts.set(direction, (directionCounts.get(direction) ?? 0) + 1);
    }
  }

  cleaned.schema_version = 2;
  cleaned.language_pair = "zh_en";
  cleaned.domain = domain;
  cleaned.generated_at = new Date().toISOString();
  cleaned.cleaning_summary = {
    cleaned_at: cleaned.generated_at,
    max_chunks_per_page: maximumChunksPerPage,
    direction_counts: Object.fromEntries(directionCounts),
  };
  cleaned.records = candidates.sort((left, right) =>
    [left.source_lang, left.provenance.source_id, left.id]
      .join("|")
      .localeCompare([right.source_lang, right.provenance.source_id, right.id].join("|")),
  );
  writeJsonAtomic(cleanedFile, cleaned);

  const summary = [...directionCounts].map(([direction, count]) => ({
    domain,
    direction,
    cleaned_rows: count,
    required_minimum: requiredMinimum || "-",
    missing: requiredMinimum ? Math.max(0, requiredMinimum - count) : 0,
  }));
  console.table(summary);

  if (requiredMinimum && summary.some((row) => row.cleaned_rows < requiredMinimum)) {
    throw new Error(
      `清洗结果已保存，但 ${domain} 尚未达到每方向 ${requiredMinimum} 条，请继续补采。`,
    );
  }
}

main().catch((error) => {
  console.error(`清洗失败：${error.message}`);
  process.exitCode = 1;
});
