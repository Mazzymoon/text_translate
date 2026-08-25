#!/usr/bin/env node

import path from "node:path";
import {
  DATASET_ROOT,
  ZH_EN_DOMAINS,
  countEnglishWords,
  countHanCharacters,
  emptyZhEnDocument,
  isIsoDateInRange,
  normalizeText,
  parseArguments,
  readJson,
  sha256,
  zhEnCrawledDirectory,
  zhEnDataFile,
} from "./lib/corpus.mjs";

const SOURCE_FILE = path.join(zhEnCrawledDirectory("source"), "zh_en_sources.json");
const METHODS = new Set(["human", "google_mt", "llm_mt"]);
const REVIEW_STATUSES = new Set(["pending", "approved", "revised", "rejected"]);
const REVIEWED_STATUSES = new Set(["approved", "revised"]);

function addError(errors, id, message) {
  errors.push(`${id ?? "unknown"}: ${message}`);
}

function loadDocuments(stage) {
  return ZH_EN_DOMAINS.map((domain) => {
    const filePath = zhEnDataFile(stage, domain);
    return { domain, filePath, document: readJson(filePath, emptyZhEnDocument(domain)) };
  });
}

function main() {
  const args = parseArguments(process.argv.slice(2));
  const requiredMinimum = args["require-min-per-direction"]
    ? Number(args["require-min-per-direction"])
    : args["require-complete"]
      ? 5001
      : 0;
  const requireReview = Boolean(args["require-review"]);
  if (!Number.isInteger(requiredMinimum) || requiredMinimum < 0) {
    throw new Error("--require-min-per-direction 必须是非负整数");
  }

  const config = readJson(SOURCE_FILE);
  const rawDocuments = loadDocuments("raw");
  const cleanedDocuments = loadDocuments("cleaned");
  const errors = [];
  const rawIds = new Set();
  const rawUrls = new Set();
  const rawById = new Map();
  const rawDistribution = new Map();

  for (const { domain, filePath, document } of rawDocuments) {
    if (document.domain !== domain) errors.push(`${filePath}: 顶层 domain 应为 ${domain}`);
    for (const record of document.records) {
      if (!record.id) addError(errors, record.id, "缺少 id");
      if (rawIds.has(record.id)) addError(errors, record.id, "raw id 跨文件重复");
      rawIds.add(record.id);
      rawById.set(record.id, record);
      if (rawUrls.has(record.canonical_url)) addError(errors, record.id, "canonical_url 跨文件重复");
      rawUrls.add(record.canonical_url);
      if (record.domain !== domain) addError(errors, record.id, `domain 应为 ${domain}`);
      if (!["zh-CN", "en"].includes(record.source_lang)) addError(errors, record.id, "source_lang 非法");
      if (!record.canonical_url || !record.source_site || !record.source_id) {
        addError(errors, record.id, "来源信息不完整");
      }
      if (!normalizeText(record.title)) addError(errors, record.id, "标题为空");
      if (!record.crawled_at || Number.isNaN(new Date(record.crawled_at).valueOf())) {
        addError(errors, record.id, "抓取时间缺失或格式错误");
      }
      if (!record.rights_status || !record.rights_note) addError(errors, record.id, "权利信息不完整");
      if (!record.rights_evidence_url) addError(errors, record.id, "缺少权利证据页");
      if (
        record.rights_status === "open_government_licence_3" &&
        (!record.license_name || !record.license_url || !record.attribution)
      ) {
        addError(errors, record.id, "开放许可或署名字段不完整");
      }
      if (!normalizeText(record.raw_text)) addError(errors, record.id, "raw_text 为空");
      if (!normalizeText(record.raw_content_html)) addError(errors, record.id, "raw_content_html 为空");
      if (sha256(record.raw_text) !== record.content_sha256) addError(errors, record.id, "正文哈希不匹配");
      if (
        !isIsoDateInRange(
          record.published_at,
          config.collection_window.start_date,
          config.collection_window.end_date,
        )
      ) {
        addError(errors, record.id, "发布日期不在采集区间内");
      }
      const key = `${domain} / ${record.source_lang}`;
      rawDistribution.set(key, (rawDistribution.get(key) ?? 0) + 1);
    }
  }

  const cleanIds = new Set();
  const cleanTexts = new Set();
  const distribution = new Map();
  const readyDistribution = new Map();
  const reviewDistribution = new Map();
  const statuses = new Map();
  for (const { domain, filePath, document } of cleanedDocuments) {
    if (document.domain !== domain) errors.push(`${filePath}: 顶层 domain 应为 ${domain}`);
    for (const record of document.records) {
      if (!record.id) addError(errors, record.id, "缺少 id");
      if (cleanIds.has(record.id)) addError(errors, record.id, "cleaned id 跨文件重复");
      cleanIds.add(record.id);
      const textKey = `${record.source_lang}\u001F${normalizeText(record.source_text).toLowerCase()}`;
      if (cleanTexts.has(textKey)) addError(errors, record.id, "规范化源文本跨文件重复");
      cleanTexts.add(textKey);
      if (record.domain !== domain) addError(errors, record.id, `domain 应为 ${domain}`);
      if (!["zh-CN", "en"].includes(record.source_lang)) addError(errors, record.id, "source_lang 非法");
      const expectedTarget = record.source_lang === "zh-CN" ? "en" : "zh-CN";
      if (record.target_lang !== expectedTarget) addError(errors, record.id, "target_lang 不匹配");
      if (!normalizeText(record.source_text)) addError(errors, record.id, "source_text 为空");
      if (record.source_lang === "zh-CN" && countHanCharacters(record.source_text) < 100) {
        addError(errors, record.id, "中文源文本不足 100 个汉字");
      }
      if (record.source_lang === "en" && countEnglishWords(record.source_text) < 60) {
        addError(errors, record.id, "英文源文本不足 60 个单词");
      }
      if (sha256(record.source_text) !== record.quality?.source_sha256) {
        addError(errors, record.id, "清洗文本哈希不匹配");
      }
      const rawRecord = rawById.get(record.provenance?.raw_record_id);
      if (!rawRecord) {
        addError(errors, record.id, "找不到对应的 raw 记录");
      } else if (rawRecord.domain !== domain) {
        addError(errors, record.id, "关联的 raw 记录属于其他领域");
      }
      if (!record.provenance?.source_url) addError(errors, record.id, "缺少来源 URL");
      if (!record.provenance?.rights_status || !record.provenance?.rights_note) {
        addError(errors, record.id, "缺少来源权利信息");
      }
      if (!record.provenance?.rights_evidence_url) {
        addError(errors, record.id, "缺少来源权利证据页");
      }

      const hasTarget = Boolean(normalizeText(record.target_text));
      const hasMethod = METHODS.has(record.translation_method);
      const reviewStatus = record.quality?.review_status ?? "pending";
      if (!REVIEW_STATUSES.has(reviewStatus)) {
        addError(errors, record.id, `未知人工审核状态：${reviewStatus}`);
      }
      if (record.status === "ready") {
        if (!hasTarget || !hasMethod) addError(errors, record.id, "ready 记录缺少译文或翻译方式");
        const chineseText = record.source_lang === "zh-CN" ? record.source_text : record.target_text;
        const actualCount = countHanCharacters(chineseText);
        if (actualCount < 100) addError(errors, record.id, "中文译文不足 100 个汉字");
        if (record.zh_char_count !== actualCount) addError(errors, record.id, "zh_char_count 不正确");
      } else if (record.status === "needs_translation") {
        if (hasTarget || record.translation_method !== null) {
          addError(errors, record.id, "待翻译记录不应包含译文或翻译方式");
        }
      } else {
        addError(errors, record.id, `未知状态 ${record.status}`);
      }

      const direction = `${record.source_lang}->${record.target_lang}`;
      const key = `${domain} / ${direction}`;
      distribution.set(key, (distribution.get(key) ?? 0) + 1);
      if (record.status === "ready") {
        readyDistribution.set(key, (readyDistribution.get(key) ?? 0) + 1);
        if (REVIEWED_STATUSES.has(reviewStatus)) {
          reviewDistribution.set(key, (reviewDistribution.get(key) ?? 0) + 1);
        }
      }
      statuses.set(record.status, (statuses.get(record.status) ?? 0) + 1);
    }
  }

  if (requiredMinimum) {
    for (const domain of ZH_EN_DOMAINS) {
      for (const direction of ["zh-CN->en", "en->zh-CN"]) {
        const key = `${domain} / ${direction}`;
        const actual = distribution.get(key) ?? 0;
        if (actual < requiredMinimum) {
          errors.push(`${key} 至少应为 ${requiredMinimum} 条，实际为 ${actual} 条`);
        }
      }
    }
  }
  if (requireReview) {
    for (const [key, readyCount] of readyDistribution) {
      const reviewedCount = reviewDistribution.get(key) ?? 0;
      const requiredCount = Math.ceil(readyCount * 0.1);
      if (reviewedCount < requiredCount) {
        errors.push(`${key} 至少应人工审核 ${requiredCount} 条，实际为 ${reviewedCount} 条`);
      }
    }
  }

  console.table([...rawDistribution].map(([group, pages]) => ({ group, raw_pages: pages })));
  console.table([...distribution].map(([group, rows]) => ({ group, cleaned_rows: rows })));
  console.table([...statuses].map(([status, rows]) => ({ status, rows })));
  if (errors.length > 0) {
    throw new Error(`中间数据校验失败，共 ${errors.length} 个问题：\n- ${errors.join("\n- ")}`);
  }
  const rawCount = rawDocuments.reduce((sum, item) => sum + item.document.records.length, 0);
  const cleanCount = cleanedDocuments.reduce((sum, item) => sum + item.document.records.length, 0);
  console.log(`中间数据校验通过：${rawCount} 个原始页面，${cleanCount} 条清洗记录。`);
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
