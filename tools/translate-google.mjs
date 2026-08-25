#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DATASET_ROOT,
  zhEnCrawledDirectory,
  ZH_EN_DOMAINS,
  countEnglishWords,
  countHanCharacters,
  normalizeText,
  parseArguments,
  readJson,
  sha256,
  sleep,
  writeJsonAtomic,
  zhEnDataFile,
} from "./lib/corpus.mjs";

const PROVIDER = "google_cloud_translation_basic_v2";
const GOOGLE_ENDPOINT = "https://translation.googleapis.com/language/translate/v2";
const MONTHLY_FREE_TIER_CODEPOINTS = 500_000;
const DEFAULT_SAFETY_BUDGET_CODEPOINTS = 480_000;
const DEFAULT_BATCH_CODEPOINTS = 4_500;
const DEFAULT_BATCH_RECORDS = 50;
const DEFAULT_RETRIES = 2;
const DEFAULT_TIMEOUT_MS = 60_000;
const TRANSLATIONS_DIR = zhEnCrawledDirectory("translations");
const LOCK_FILE = path.join(TRANSLATIONS_DIR, ".google_translate.lock");

const DIRECTIONS = {
  "zh-en": { source_lang: "zh-CN", target_lang: "en" },
  "en-zh": { source_lang: "en", target_lang: "zh-CN" },
};

export function countCodepoints(value) {
  return [...String(value ?? "")].length;
}

function positiveInteger(value, name, fallback) {
  const parsed = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} 必须是正整数`);
  }
  return parsed;
}

function parseDomains(value) {
  const requested = String(value ?? "all");
  if (requested === "all") return [...ZH_EN_DOMAINS];
  if (!ZH_EN_DOMAINS.includes(requested)) {
    throw new Error(`--domain 必须是 all、${ZH_EN_DOMAINS.join("、")} 之一`);
  }
  return [requested];
}

function parseDirections(value) {
  const requested = String(value ?? "all");
  if (requested === "all") return Object.keys(DIRECTIONS);
  if (!DIRECTIONS[requested]) {
    throw new Error("--direction 必须是 all、zh-en、en-zh 之一");
  }
  return [requested];
}

function billingMonth(value) {
  const googleMonthParts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Los_Angeles",
    year: "numeric",
    month: "2-digit",
  })
    .formatToParts(new Date())
    .reduce((result, part) => ({ ...result, [part.type]: part.value }), {});
  const defaultMonth = `${googleMonthParts.year}-${googleMonthParts.month}`;
  const month = String(value ?? defaultMonth);
  if (!/^20\d{2}-(?:0[1-9]|1[0-2])$/.test(month)) {
    throw new Error("--billing-month 必须使用 YYYY-MM 格式");
  }
  return month;
}

function usageFile(month) {
  return path.join(TRANSLATIONS_DIR, `google_usage_${month}.json`);
}

function acceptedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.google_mt.json`);
}

function rejectedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.google_mt_rejected.json`);
}

function readArray(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const value = readJson(filePath);
  if (!Array.isArray(value)) throw new Error(`${filePath} 顶层必须是数组`);
  return value;
}

function createLedger(month, safetyBudgetCodepoints) {
  return {
    schema_version: 1,
    provider: PROVIDER,
    billing_month: month,
    monthly_free_tier_codepoints: MONTHLY_FREE_TIER_CODEPOINTS,
    safety_budget_codepoints: safetyBudgetCodepoints,
    attempted_codepoints: 0,
    successful_codepoints: 0,
    requests_attempted: 0,
    records_accepted: 0,
    records_rejected: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    requests: [],
  };
}

function loadLedger(month, requestedSafetyBudget) {
  const filePath = usageFile(month);
  const ledger = fs.existsSync(filePath)
    ? readJson(filePath)
    : createLedger(month, requestedSafetyBudget);
  if (ledger.provider !== PROVIDER || ledger.billing_month !== month) {
    throw new Error(`${filePath} 的服务商或账期不匹配`);
  }
  if (requestedSafetyBudget > DEFAULT_SAFETY_BUDGET_CODEPOINTS) {
    throw new Error(
      `安全预算不能超过 ${DEFAULT_SAFETY_BUDGET_CODEPOINTS.toLocaleString()} 个代码点`,
    );
  }
  const existingBudget = Number(
    ledger.safety_budget_codepoints ?? DEFAULT_SAFETY_BUDGET_CODEPOINTS,
  );
  if (requestedSafetyBudget > existingBudget) {
    throw new Error(
      `本月账本安全预算已经锁定为 ${existingBudget.toLocaleString()}，不能在脚本中调高`,
    );
  }
  ledger.safety_budget_codepoints = Math.min(existingBudget, requestedSafetyBudget);
  ledger.monthly_free_tier_codepoints = MONTHLY_FREE_TIER_CODEPOINTS;
  ledger.requests ??= [];
  return { filePath, ledger };
}

function persistLedger(filePath, ledger) {
  ledger.updated_at = new Date().toISOString();
  writeJsonAtomic(filePath, ledger);
}

function decodeHtmlEntities(value) {
  return String(value ?? "")
    .replace(/&#(\d+);/g, (_, number) => String.fromCodePoint(Number(number)))
    .replace(/&#x([0-9a-f]+);/gi, (_, number) =>
      String.fromCodePoint(Number.parseInt(number, 16)),
    )
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&apos;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");
}

function qualityFailure(record, targetText) {
  if (!targetText) return "empty_translation";
  if (record.target_lang === "zh-CN" && countHanCharacters(targetText) < 100) {
    return "target_zh_han_count_below_100";
  }
  if (record.target_lang === "en" && countEnglishWords(targetText) < 30) {
    return "target_en_word_count_below_30";
  }
  return null;
}

function outputRecord(record, targetText, batchId, inputCodepoints) {
  return {
    id: record.id,
    source_lang: record.source_lang,
    target_lang: record.target_lang,
    source_text: record.source_text,
    target_text: targetText,
    domain: record.domain,
    translation_method: "google_mt",
    provider: PROVIDER,
    provider_batch_id: batchId,
    input_codepoints: inputCodepoints,
    translated_at: new Date().toISOString(),
    review_status: "pending",
    review_notes: "",
  };
}

function loadOutputs(domains) {
  const state = new Map();
  for (const domain of domains) {
    const accepted = readArray(acceptedFile(domain));
    const rejected = readArray(rejectedFile(domain));
    state.set(domain, { accepted, rejected });
  }
  return state;
}

function allKnownOutputIds(outputState) {
  const ids = new Set();
  for (const { accepted, rejected } of outputState.values()) {
    for (const item of [...accepted, ...rejected]) ids.add(item.id);
  }
  return ids;
}

function attemptedRecordIds(ledger) {
  return new Set(ledger.requests.flatMap((request) => request.record_ids ?? []));
}

function loadCandidateGroups(domains, directionNames, excludedIds) {
  const groups = [];
  for (const domain of domains) {
    const cleaned = readJson(zhEnDataFile("cleaned", domain));
    for (const directionName of directionNames) {
      const direction = DIRECTIONS[directionName];
      const records = cleaned.records
        .filter(
          (record) =>
            record.source_lang === direction.source_lang &&
            record.target_lang === direction.target_lang &&
            !normalizeText(record.target_text) &&
            !excludedIds.has(record.id),
        )
        .sort((left, right) => left.id.localeCompare(right.id));
      groups.push({
        key: `${domain}/${directionName}`,
        domain,
        direction_name: directionName,
        source_lang: direction.source_lang,
        target_lang: direction.target_lang,
        records,
        index: 0,
      });
    }
  }
  return groups;
}

function cloneGroups(groups) {
  return groups.map((group) => ({ ...group, records: [...group.records], index: 0 }));
}

function nextBatchFromGroup(
  group,
  maximumCodepoints,
  maximumRecords,
  remainingRecordLimit,
) {
  if (group.index >= group.records.length || remainingRecordLimit < 1) return null;
  const records = [];
  let codepoints = 0;
  let cursor = group.index;
  const recordLimit = Math.min(maximumRecords, remainingRecordLimit);
  while (cursor < group.records.length && records.length < recordLimit) {
    const record = group.records[cursor];
    const size = countCodepoints(record.source_text);
    // Near the monthly safety cap, the remaining allowance can be smaller than
    // the next record. Stop cleanly instead of spending past the cap.
    if (size > maximumCodepoints && records.length === 0) return null;
    if (codepoints + size > maximumCodepoints) break;
    records.push(record);
    codepoints += size;
    cursor += 1;
  }
  if (records.length === 0) return null;
  group.index = cursor;
  return {
    domain: group.domain,
    direction_name: group.direction_name,
    source_lang: group.source_lang,
    target_lang: group.target_lang,
    records,
    input_codepoints: codepoints,
  };
}

function chooseNextBatch(
  groups,
  startIndex,
  maximumCodepoints,
  maximumRecords,
  remainingRecordLimit,
) {
  for (let offset = 0; offset < groups.length; offset += 1) {
    const groupIndex = (startIndex + offset) % groups.length;
    const batch = nextBatchFromGroup(
      groups[groupIndex],
      maximumCodepoints,
      maximumRecords,
      remainingRecordLimit,
    );
    if (batch) return { batch, nextGroupIndex: (groupIndex + 1) % groups.length };
  }
  return null;
}

export function planTranslation({
  groups,
  availableCodepoints,
  batchCodepoints,
  batchRecords,
  maximumRecords,
}) {
  const workingGroups = cloneGroups(groups);
  const byGroup = new Map(workingGroups.map((group) => [group.key, { records: 0, codepoints: 0 }]));
  let plannedCodepoints = 0;
  let plannedRecords = 0;
  let plannedRequests = 0;
  let groupIndex = 0;
  while (plannedRecords < maximumRecords) {
    const remainingBudget = availableCodepoints - plannedCodepoints;
    if (remainingBudget < 1) break;
    const chosen = chooseNextBatch(
      workingGroups,
      groupIndex,
      Math.min(batchCodepoints, remainingBudget),
      batchRecords,
      maximumRecords - plannedRecords,
    );
    if (!chosen) break;
    groupIndex = chosen.nextGroupIndex;
    const { batch } = chosen;
    const stats = byGroup.get(`${batch.domain}/${batch.direction_name}`);
    stats.records += batch.records.length;
    stats.codepoints += batch.input_codepoints;
    plannedCodepoints += batch.input_codepoints;
    plannedRecords += batch.records.length;
    plannedRequests += 1;
  }
  return { plannedCodepoints, plannedRecords, plannedRequests, byGroup };
}

function acquireLock() {
  fs.mkdirSync(TRANSLATIONS_DIR, { recursive: true });
  try {
    fs.writeFileSync(
      LOCK_FILE,
      `${JSON.stringify({ pid: process.pid, started_at: new Date().toISOString() }, null, 2)}\n`,
      { encoding: "utf8", flag: "wx" },
    );
  } catch (error) {
    if (error.code === "EEXIST") {
      throw new Error(
        `检测到翻译锁文件 ${LOCK_FILE}；确认没有其他翻译进程后再手动删除该锁文件`,
      );
    }
    throw error;
  }
}

function releaseLock() {
  fs.rmSync(LOCK_FILE, { force: true });
}

function apiErrorMessage(status, body) {
  try {
    const parsed = JSON.parse(body);
    return `HTTP ${status}: ${parsed.error?.message ?? body.slice(0, 500)}`;
  } catch {
    return `HTTP ${status}: ${body.slice(0, 500)}`;
  }
}

export async function callGoogleBasic({
  apiKey,
  sourceLanguage,
  targetLanguage,
  texts,
  timeoutMilliseconds = DEFAULT_TIMEOUT_MS,
  fetchImplementation = fetch,
  endpoint = GOOGLE_ENDPOINT,
}) {
  const response = await fetchImplementation(`${endpoint}?key=${encodeURIComponent(apiKey)}`, {
    method: "POST",
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify({
      q: texts,
      source: sourceLanguage,
      target: targetLanguage,
      format: "text",
    }),
    signal: AbortSignal.timeout(timeoutMilliseconds),
  });
  const body = await response.text();
  if (!response.ok) {
    const error = new Error(apiErrorMessage(response.status, body));
    error.status = response.status;
    throw error;
  }
  const parsed = JSON.parse(body);
  const translations = parsed.data?.translations;
  if (!Array.isArray(translations) || translations.length !== texts.length) {
    throw new Error("Google API 返回的译文数量与请求数量不一致");
  }
  return translations.map((item) => normalizeText(decodeHtmlEntities(item.translatedText)));
}

function isRetryable(error) {
  return (
    error?.name === "TimeoutError" ||
    error?.name === "AbortError" ||
    error?.status === 429 ||
    (Number(error?.status) >= 500 && Number(error?.status) <= 599)
  );
}

function appendOutputs(outputState, batch, translatedTexts, batchId) {
  const domainOutput = outputState.get(batch.domain);
  const accepted = [];
  const rejected = [];
  for (let index = 0; index < batch.records.length; index += 1) {
    const record = batch.records[index];
    const targetText = translatedTexts[index];
    const output = outputRecord(
      record,
      targetText,
      batchId,
      countCodepoints(record.source_text),
    );
    const failure = qualityFailure(record, targetText);
    if (failure) rejected.push({ ...output, rejection_reason: failure });
    else accepted.push(output);
  }
  domainOutput.accepted.push(...accepted);
  domainOutput.rejected.push(...rejected);
  writeJsonAtomic(acceptedFile(batch.domain), domainOutput.accepted);
  writeJsonAtomic(rejectedFile(batch.domain), domainOutput.rejected);
  return { accepted: accepted.length, rejected: rejected.length };
}

async function executeBatch({
  batch,
  apiKey,
  ledger,
  ledgerFile,
  outputState,
  maximumRetries,
  timeoutMilliseconds,
}) {
  const batchId = `google_batch_${sha256(
    `${batch.domain}\n${batch.direction_name}\n${batch.records.map((record) => record.id).join("\n")}`,
  ).slice(0, 20)}`;
  let lastError;
  for (let attemptNumber = 1; attemptNumber <= maximumRetries + 1; attemptNumber += 1) {
    if (
      ledger.attempted_codepoints + batch.input_codepoints >
      ledger.safety_budget_codepoints
    ) {
      throw new Error("本月本地安全预算已不足以重试当前批次，已停止且未发送重试请求");
    }
    const request = {
      request_id: `${batchId}_attempt_${attemptNumber}_${Date.now()}`,
      batch_id: batchId,
      attempt_number: attemptNumber,
      domain: batch.domain,
      direction: batch.direction_name,
      record_ids: batch.records.map((record) => record.id),
      input_codepoints: batch.input_codepoints,
      status: "attempting",
      attempted_at: new Date().toISOString(),
    };
    ledger.requests.push(request);
    ledger.attempted_codepoints += batch.input_codepoints;
    ledger.requests_attempted += 1;
    persistLedger(ledgerFile, ledger);

    try {
      const translatedTexts = await callGoogleBasic({
        apiKey,
        sourceLanguage: batch.source_lang,
        targetLanguage: batch.target_lang,
        texts: batch.records.map((record) => record.source_text),
        timeoutMilliseconds,
      });
      const counts = appendOutputs(outputState, batch, translatedTexts, batchId);
      request.status = "completed";
      request.completed_at = new Date().toISOString();
      request.accepted_records = counts.accepted;
      request.rejected_records = counts.rejected;
      ledger.successful_codepoints += batch.input_codepoints;
      ledger.records_accepted += counts.accepted;
      ledger.records_rejected += counts.rejected;
      persistLedger(ledgerFile, ledger);
      return counts;
    } catch (error) {
      lastError = error;
      request.status = "failed";
      request.failed_at = new Date().toISOString();
      request.error = String(error.message).slice(0, 1000);
      persistLedger(ledgerFile, ledger);
      if (!isRetryable(error) || attemptNumber > maximumRetries) break;
      await sleep(Math.min(8_000, 1_000 * 2 ** (attemptNumber - 1)));
    }
  }
  throw new Error(`批次 ${batchId} 翻译失败：${lastError?.message ?? "未知错误"}`);
}

function printStatus(ledger, outputState) {
  console.table([
    {
      billing_month: ledger.billing_month,
      free_tier: ledger.monthly_free_tier_codepoints,
      local_safety_budget: ledger.safety_budget_codepoints,
      attempted: ledger.attempted_codepoints,
      successful: ledger.successful_codepoints,
      remaining_local_budget: Math.max(
        0,
        ledger.safety_budget_codepoints - ledger.attempted_codepoints,
      ),
      requests: ledger.requests_attempted,
      accepted: ledger.records_accepted,
      rejected: ledger.records_rejected,
    },
  ]);
  console.table(
    [...outputState].map(([domain, value]) => ({
      domain,
      accepted_output_rows: value.accepted.length,
      rejected_output_rows: value.rejected.length,
    })),
  );
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  const domains = parseDomains(args.domain);
  const directionNames = parseDirections(args.direction);
  const month = billingMonth(args["billing-month"]);
  const safetyBudgetCodepoints = positiveInteger(
    args["safety-budget-codepoints"],
    "--safety-budget-codepoints",
    DEFAULT_SAFETY_BUDGET_CODEPOINTS,
  );
  const batchCodepoints = positiveInteger(
    args["batch-codepoints"],
    "--batch-codepoints",
    DEFAULT_BATCH_CODEPOINTS,
  );
  if (batchCodepoints > 5_000) {
    throw new Error("为遵守 Google 推荐值，--batch-codepoints 不能超过 5000");
  }
  const batchRecords = positiveInteger(
    args["batch-records"],
    "--batch-records",
    DEFAULT_BATCH_RECORDS,
  );
  const maximumRecords = positiveInteger(
    args["max-records"],
    "--max-records",
    Number.MAX_SAFE_INTEGER,
  );
  const maximumRetries = positiveInteger(
    args["max-attempts"],
    "--max-attempts",
    DEFAULT_RETRIES + 1,
  ) - 1;
  const timeoutMilliseconds = positiveInteger(
    args["timeout-ms"],
    "--timeout-ms",
    DEFAULT_TIMEOUT_MS,
  );

  const { filePath: ledgerFile, ledger } = loadLedger(month, safetyBudgetCodepoints);
  const allOutputState = loadOutputs(ZH_EN_DOMAINS);
  if (args.status) {
    printStatus(ledger, allOutputState);
    return;
  }

  const excludedIds = allKnownOutputIds(allOutputState);
  for (const id of attemptedRecordIds(ledger)) excludedIds.add(id);
  const groups = loadCandidateGroups(domains, directionNames, excludedIds);
  const availableCodepoints = Math.max(
    0,
    ledger.safety_budget_codepoints - ledger.attempted_codepoints,
  );
  const plan = planTranslation({
    groups,
    availableCodepoints,
    batchCodepoints,
    batchRecords,
    maximumRecords,
  });
  console.table(
    groups.map((group) => {
      const planned = plan.byGroup.get(group.key);
      return {
        group: group.key,
        available_records: group.records.length,
        planned_records: planned.records,
        planned_codepoints: planned.codepoints,
      };
    }),
  );
  console.log(
    `本次计划：${plan.plannedRecords} 条，${plan.plannedCodepoints.toLocaleString()} 个代码点，约 ${plan.plannedRequests} 个请求。`,
  );
  console.log(
    `本月账本已尝试 ${ledger.attempted_codepoints.toLocaleString()}，本地安全上限 ${ledger.safety_budget_codepoints.toLocaleString()}。`,
  );
  if (!args.execute) {
    console.log("当前为预演模式，没有调用 API。确认云端配额后追加 --execute。 ");
    return;
  }
  if (!args["confirm-free-tier-and-quota"]) {
    throw new Error(
      "执行前必须确认当前结算账号仍有免费额度、项目专用于本任务且云端字符配额不超过 480000；确认后追加 --confirm-free-tier-and-quota",
    );
  }
  const apiKey = process.env.GOOGLE_TRANSLATE_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("缺少环境变量 GOOGLE_TRANSLATE_API_KEY；不要把密钥写入命令参数或项目文件");
  }
  if (plan.plannedRecords === 0) {
    console.log("没有可执行的翻译批次。");
    return;
  }

  acquireLock();
  const executionGroups = cloneGroups(groups);
  let groupIndex = 0;
  let processedRecords = 0;
  try {
    persistLedger(ledgerFile, ledger);
    while (processedRecords < maximumRecords) {
      const remainingBudget =
        ledger.safety_budget_codepoints - ledger.attempted_codepoints;
      if (remainingBudget < 1) break;
      const chosen = chooseNextBatch(
        executionGroups,
        groupIndex,
        Math.min(batchCodepoints, remainingBudget),
        batchRecords,
        maximumRecords - processedRecords,
      );
      if (!chosen) break;
      groupIndex = chosen.nextGroupIndex;
      const { batch } = chosen;
      const counts = await executeBatch({
        batch,
        apiKey,
        ledger,
        ledgerFile,
        outputState: allOutputState,
        maximumRetries,
        timeoutMilliseconds,
      });
      processedRecords += batch.records.length;
      console.log(
        `${batch.domain}/${batch.direction_name}: +${counts.accepted} 可导入，+${counts.rejected} 待处理；累计尝试 ${ledger.attempted_codepoints.toLocaleString()}/${ledger.safety_budget_codepoints.toLocaleString()} 代码点`,
      );
    }
  } finally {
    releaseLock();
  }
  printStatus(ledger, allOutputState);
  console.log("翻译结果已保存，但尚未自动导入 cleaned JSON。请先检查 rejected 文件和抽样质量。");
}

const isDirectExecution =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectExecution) {
  main().catch((error) => {
    console.error(`Google 批量翻译失败：${error.message}`);
    process.exitCode = 1;
  });
}
