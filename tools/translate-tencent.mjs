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

const PROVIDER = "tencent_tokenhub";
const DEFAULT_MODEL = "hy-mt2-pro";
const ALLOWED_MODELS = new Set(["hy-mt2-pro", "hy-mt2-plus", "hy-mt2-lite"]);
const TOKENHUB_ENDPOINT = "https://tokenhub.tencentmaas.com/v1/chat/completions";
const FREE_PACKAGE_TOKENS = 1_000_000;
const DEFAULT_SAFETY_BUDGET_TOKENS = 900_000;
const DEFAULT_BATCH_SOURCE_CODEPOINTS = 1_800;
const DEFAULT_BATCH_RECORDS = 5;
const DEFAULT_MAX_OUTPUT_TOKENS = 3_000;
const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_TIMEOUT_MS = 90_000;
const DEFAULT_DELAY_MS = 1_100;
const TRANSLATIONS_DIR = zhEnCrawledDirectory("translations");
const LEDGER_FILE = path.join(TRANSLATIONS_DIR, "tencent_tokenhub_usage.json");
const LOCK_FILE = path.join(TRANSLATIONS_DIR, ".tencent_translate.lock");

const DIRECTIONS = {
  "zh-en": {
    source_lang: "zh-CN",
    target_lang: "en",
    target_name: "英语",
  },
  "en-zh": {
    source_lang: "en",
    target_lang: "zh-CN",
    target_name: "简体中文",
  },
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

function parseModel(value) {
  const model = String(value ?? DEFAULT_MODEL);
  if (!ALLOWED_MODELS.has(model)) {
    throw new Error(`--model 必须是 ${[...ALLOWED_MODELS].join("、")} 之一`);
  }
  return model;
}

function acceptedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.tencent_mt.json`);
}

function rejectedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.tencent_mt_rejected.json`);
}

function googleAcceptedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.google_mt.json`);
}

function readArray(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const value = readJson(filePath);
  if (!Array.isArray(value)) throw new Error(`${filePath} 顶层必须是数组`);
  return value;
}

function createLedger(safetyBudgetTokens) {
  return {
    schema_version: 1,
    provider: PROVIDER,
    free_package_tokens: FREE_PACKAGE_TOKENS,
    safety_budget_tokens: safetyBudgetTokens,
    accounted_tokens: 0,
    reported_tokens: 0,
    requests_attempted: 0,
    records_accepted: 0,
    records_rejected: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    requests: [],
  };
}

function loadLedger(requestedSafetyBudget) {
  const ledger = fs.existsSync(LEDGER_FILE)
    ? readJson(LEDGER_FILE)
    : createLedger(requestedSafetyBudget);
  if (ledger.provider !== PROVIDER) {
    throw new Error(`${LEDGER_FILE} 的服务商不匹配`);
  }
  if (requestedSafetyBudget > DEFAULT_SAFETY_BUDGET_TOKENS) {
    throw new Error(
      `安全预算不能超过 ${DEFAULT_SAFETY_BUDGET_TOKENS.toLocaleString()} Tokens`,
    );
  }
  const existingBudget = Number(
    ledger.safety_budget_tokens ?? DEFAULT_SAFETY_BUDGET_TOKENS,
  );
  if (requestedSafetyBudget > existingBudget) {
    throw new Error(
      `免费包账本安全预算已经锁定为 ${existingBudget.toLocaleString()}，不能在脚本中调高`,
    );
  }
  ledger.safety_budget_tokens = Math.min(existingBudget, requestedSafetyBudget);
  ledger.free_package_tokens = FREE_PACKAGE_TOKENS;
  ledger.accounted_tokens ??= 0;
  ledger.reported_tokens ??= 0;
  ledger.requests ??= [];
  for (const request of ledger.requests) {
    if (
      request.status === "failed" &&
      /JSON|Unterminated string|Expected property name|Unexpected token/.test(
        String(request.error ?? ""),
      )
    ) {
      request.status = "invalid_output_retryable";
      request.migration_note =
        "旧版 JSON 批量输出解析失败；保留额度占用并允许使用分隔符模式重试";
    } else if (
      request.status === "failed" &&
      /fetch failed|timeout|timed out|aborted|ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|socket/i.test(
        String(request.error ?? ""),
      )
    ) {
      request.status = "network_error_retryable";
      request.migration_note = "网络异常；保留额度占用并允许下次启动重试";
    }
  }
  return ledger;
}

function persistLedger(ledger) {
  ledger.updated_at = new Date().toISOString();
  writeJsonAtomic(LEDGER_FILE, ledger);
}

function loadOutputs() {
  const state = new Map();
  for (const domain of ZH_EN_DOMAINS) {
    state.set(domain, {
      accepted: readArray(acceptedFile(domain)),
      rejected: readArray(rejectedFile(domain)),
    });
  }
  return state;
}

function allKnownOutputIds(outputState) {
  const ids = new Set();
  for (const domain of ZH_EN_DOMAINS) {
    for (const item of readArray(googleAcceptedFile(domain))) ids.add(item.id);
  }
  for (const { accepted, rejected } of outputState.values()) {
    for (const item of [...accepted, ...rejected]) ids.add(item.id);
  }
  return ids;
}

function attemptedRecordIds(ledger) {
  return new Set(
    ledger.requests
      .filter(
        (request) =>
          request.status !== "rejected_before_inference" &&
          request.status !== "invalid_output_retryable" &&
          request.status !== "network_error_retryable",
      )
      .flatMap((request) => request.record_ids ?? []),
  );
}

function retryIndividuallyRecordIds(ledger) {
  return new Set(
    ledger.requests
      .filter((request) => request.status === "invalid_output_retryable")
      .flatMap((request) => request.record_ids ?? []),
  );
}

function loadCandidateGroups(
  domains,
  directionNames,
  excludedIds,
  retryIndividuallyIds,
) {
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
        target_name: direction.target_name,
        retry_individually_ids: retryIndividuallyIds,
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

export function buildTranslationPrompt(targetName, records) {
  for (const record of records) {
    if (record.source_text.includes("<SEP>")) {
      throw new Error(`${record.id}: 源文本包含保留分隔符 <SEP>`);
    }
  }
  if (records.length === 1) {
    return [
      `请将以下文本准确翻译为${targetName}。`,
      "不要省略、概括、解释或增加原文没有的信息，只输出译文。",
      records[0].source_text,
    ].join("\n");
  }
  return [
    `请将以下文本准确翻译为${targetName}。`,
    "不要省略、概括、解释或增加原文没有的信息，只输出译文。",
    "你必须在译文中保留等量的 <SEP> 分隔符，绝对不可遗漏、转义或翻译该符号，并保持各段顺序。",
    records.map((record) => record.source_text).join("\n<SEP>\n"),
  ].join("\n");
}

export function reservedTokensForRequest(prompt, maximumOutputTokens) {
  // UTF-8 字节数是输入 token 数的保守上界，再加上服务端最大输出和缓冲。
  return Buffer.byteLength(prompt, "utf8") + maximumOutputTokens + 512;
}

function nextBatchFromGroup(
  group,
  maximumSourceCodepoints,
  maximumBatchRecords,
  remainingRecordLimit,
  remainingTokenBudget,
  maximumOutputTokens,
) {
  if (group.index >= group.records.length || remainingRecordLimit < 1) return null;
  const records = [];
  let sourceCodepoints = 0;
  let cursor = group.index;
  const forceSingle = group.retry_individually_ids?.has(group.records[group.index].id);
  const recordLimit = Math.min(
    forceSingle ? 1 : maximumBatchRecords,
    remainingRecordLimit,
  );
  while (cursor < group.records.length && records.length < recordLimit) {
    const record = group.records[cursor];
    const size = countCodepoints(record.source_text);
    if (size > maximumSourceCodepoints && records.length === 0) {
      throw new Error(`${record.id}: 单条源文本超过 --batch-source-codepoints`);
    }
    if (sourceCodepoints + size > maximumSourceCodepoints) break;
    records.push(record);
    sourceCodepoints += size;
    cursor += 1;
  }
  while (records.length > 0) {
    const prompt = buildTranslationPrompt(group.target_name, records);
    const reservedTokens = reservedTokensForRequest(prompt, maximumOutputTokens);
    if (reservedTokens <= remainingTokenBudget) {
      group.index += records.length;
      return {
        domain: group.domain,
        direction_name: group.direction_name,
        source_lang: group.source_lang,
        target_lang: group.target_lang,
        target_name: group.target_name,
        records,
        source_codepoints: records.reduce(
          (sum, record) => sum + countCodepoints(record.source_text),
          0,
        ),
        prompt,
        reserved_tokens: reservedTokens,
      };
    }
    records.pop();
  }
  return null;
}

function chooseNextBatch(
  groups,
  startIndex,
  maximumSourceCodepoints,
  maximumBatchRecords,
  remainingRecordLimit,
  remainingTokenBudget,
  maximumOutputTokens,
) {
  for (let offset = 0; offset < groups.length; offset += 1) {
    const groupIndex = (startIndex + offset) % groups.length;
    const batch = nextBatchFromGroup(
      groups[groupIndex],
      maximumSourceCodepoints,
      maximumBatchRecords,
      remainingRecordLimit,
      remainingTokenBudget,
      maximumOutputTokens,
    );
    if (batch) return { batch, nextGroupIndex: (groupIndex + 1) % groups.length };
  }
  return null;
}

export function planTranslation({
  groups,
  availableTokens,
  maximumSourceCodepoints,
  maximumBatchRecords,
  maximumRecords,
  maximumOutputTokens,
}) {
  const workingGroups = cloneGroups(groups);
  const byGroup = new Map(
    workingGroups.map((group) => [
      group.key,
      { records: 0, source_codepoints: 0, reserved_tokens: 0 },
    ]),
  );
  let plannedRecords = 0;
  let plannedSourceCodepoints = 0;
  let plannedReservedTokens = 0;
  let plannedRequests = 0;
  let groupIndex = 0;
  while (plannedRecords < maximumRecords) {
    const remainingTokens = availableTokens - plannedReservedTokens;
    if (remainingTokens < 1) break;
    const chosen = chooseNextBatch(
      workingGroups,
      groupIndex,
      maximumSourceCodepoints,
      maximumBatchRecords,
      maximumRecords - plannedRecords,
      remainingTokens,
      maximumOutputTokens,
    );
    if (!chosen) break;
    groupIndex = chosen.nextGroupIndex;
    const { batch } = chosen;
    const stats = byGroup.get(`${batch.domain}/${batch.direction_name}`);
    stats.records += batch.records.length;
    stats.source_codepoints += batch.source_codepoints;
    stats.reserved_tokens += batch.reserved_tokens;
    plannedRecords += batch.records.length;
    plannedSourceCodepoints += batch.source_codepoints;
    plannedReservedTokens += batch.reserved_tokens;
    plannedRequests += 1;
  }
  return {
    plannedRecords,
    plannedSourceCodepoints,
    plannedReservedTokens,
    plannedRequests,
    byGroup,
  };
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
        `检测到翻译锁文件 ${LOCK_FILE}；确认没有其他腾讯翻译进程后再手动删除`,
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

export async function callTencentTokenHub({
  apiKey,
  model,
  prompt,
  maximumOutputTokens,
  timeoutMilliseconds = DEFAULT_TIMEOUT_MS,
  fetchImplementation = fetch,
  endpoint = TOKENHUB_ENDPOINT,
}) {
  const response = await fetchImplementation(endpoint, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json; charset=utf-8",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      max_tokens: maximumOutputTokens,
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
  const content = parsed.choices?.[0]?.message?.content;
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("TokenHub 响应缺少 choices[0].message.content");
  }
  return {
    content,
    usage: {
      prompt_tokens: Number(parsed.usage?.prompt_tokens ?? 0),
      completion_tokens: Number(parsed.usage?.completion_tokens ?? 0),
      total_tokens: Number(parsed.usage?.total_tokens ?? 0),
    },
    response_id: parsed.id ?? null,
  };
}

function stripJsonFence(value) {
  const text = String(value ?? "").trim();
  const fence = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return fence ? fence[1].trim() : text;
}

export function parseTranslatedRecords(content, sourceRecords) {
  const text = stripJsonFence(content);
  if (sourceRecords.length === 1) {
    const translatedText = normalizeText(text);
    if (!translatedText) {
      const error = new Error("模型返回了空译文");
      error.invalidTranslationOutput = true;
      throw error;
    }
    return [translatedText];
  }
  const translatedTexts = text.split("<SEP>").map((item) => normalizeText(item));
  if (
    translatedTexts.length !== sourceRecords.length ||
    translatedTexts.some((item) => !item)
  ) {
    const error = new Error(
      `模型返回的分隔段数不正确：期望 ${sourceRecords.length}，实际 ${translatedTexts.length}`,
    );
    error.invalidTranslationOutput = true;
    throw error;
  }
  return translatedTexts;
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

function outputRecord(record, targetText, batchId, model, usage, responseId) {
  return {
    id: record.id,
    source_lang: record.source_lang,
    target_lang: record.target_lang,
    source_text: record.source_text,
    target_text: targetText,
    domain: record.domain,
    translation_method: "llm_mt",
    provider: PROVIDER,
    provider_model: model,
    provider_batch_id: batchId,
    provider_response_id: responseId,
    provider_batch_usage: usage,
    input_codepoints: countCodepoints(record.source_text),
    translated_at: new Date().toISOString(),
    review_status: "pending",
    review_notes: "",
  };
}

function appendOutputs(outputState, batch, translatedTexts, batchId, model, result) {
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
      model,
      result.usage,
      result.response_id,
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

function reconcileProviderUsage(ledger, request, batch, result) {
  const reportedTokens =
    Number.isInteger(result.usage.total_tokens) && result.usage.total_tokens > 0
      ? result.usage.total_tokens
      : batch.reserved_tokens;
  ledger.accounted_tokens += reportedTokens - batch.reserved_tokens;
  ledger.reported_tokens += reportedTokens;
  request.accounted_tokens = reportedTokens;
  request.provider_usage = result.usage;
  request.provider_response_id = result.response_id;
  return reportedTokens;
}

function rebuildBatch(batch, records, maximumOutputTokens) {
  const prompt = buildTranslationPrompt(batch.target_name, records);
  return {
    ...batch,
    records,
    source_codepoints: records.reduce(
      (sum, record) => sum + countCodepoints(record.source_text),
      0,
    ),
    prompt,
    reserved_tokens: reservedTokensForRequest(prompt, maximumOutputTokens),
  };
}

function isRetryable(error) {
  const message = `${error?.message ?? ""} ${error?.cause?.code ?? ""}`;
  return (
    error instanceof TypeError ||
    error?.name === "TimeoutError" ||
    error?.name === "AbortError" ||
    error?.status === 429 ||
    (Number(error?.status) >= 500 && Number(error?.status) <= 599) ||
    /fetch failed|timeout|timed out|aborted|ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|socket/i.test(
      message,
    )
  );
}

function wasRejectedBeforeInference(error) {
  return [400, 401, 403, 404, 409, 422, 429].includes(Number(error?.status));
}

async function executeBatch({
  batch,
  apiKey,
  model,
  ledger,
  outputState,
  maximumAttempts,
  maximumOutputTokens,
  timeoutMilliseconds,
}) {
  const batchId = `tencent_batch_${sha256(
    `${batch.domain}\n${batch.direction_name}\n${batch.records.map((record) => record.id).join("\n")}`,
  ).slice(0, 20)}`;
  let lastError;
  for (let attemptNumber = 1; attemptNumber <= maximumAttempts; attemptNumber += 1) {
    if (
      ledger.accounted_tokens + batch.reserved_tokens >
      ledger.safety_budget_tokens
    ) {
      throw new Error("免费包安全预算不足以发送或重试当前批次，已停止");
    }
    const request = {
      request_id: `${batchId}_attempt_${attemptNumber}_${Date.now()}`,
      batch_id: batchId,
      attempt_number: attemptNumber,
      model,
      domain: batch.domain,
      direction: batch.direction_name,
      record_ids: batch.records.map((record) => record.id),
      source_codepoints: batch.source_codepoints,
      reserved_tokens: batch.reserved_tokens,
      accounted_tokens: batch.reserved_tokens,
      status: "attempting",
      attempted_at: new Date().toISOString(),
    };
    ledger.requests.push(request);
    ledger.accounted_tokens += batch.reserved_tokens;
    ledger.requests_attempted += 1;
    persistLedger(ledger);

    let providerResult;
    try {
      providerResult = await callTencentTokenHub({
        apiKey,
        model,
        prompt: batch.prompt,
        maximumOutputTokens,
        timeoutMilliseconds,
      });
      const translatedTexts = parseTranslatedRecords(providerResult.content, batch.records);
      const counts = appendOutputs(
        outputState,
        batch,
        translatedTexts,
        batchId,
        model,
        providerResult,
      );
      reconcileProviderUsage(ledger, request, batch, providerResult);
      ledger.records_accepted += counts.accepted;
      ledger.records_rejected += counts.rejected;
      request.status = "completed";
      request.completed_at = new Date().toISOString();
      request.accepted_records = counts.accepted;
      request.rejected_records = counts.rejected;
      persistLedger(ledger);
      return counts;
    } catch (error) {
      lastError = error;
      const retryable = isRetryable(error);
      if (wasRejectedBeforeInference(error)) {
        ledger.accounted_tokens -= batch.reserved_tokens;
        request.accounted_tokens = 0;
        request.status = "rejected_before_inference";
      } else if (error.invalidTranslationOutput) {
        if (providerResult) {
          reconcileProviderUsage(ledger, request, batch, providerResult);
        }
        request.status = "invalid_output_retryable";
      } else if (retryable) {
        request.status = "network_error_retryable";
      } else {
        request.status = "failed";
      }
      request.failed_at = new Date().toISOString();
      request.error = String(error.message).slice(0, 1000);
      persistLedger(ledger);
      if (error.invalidTranslationOutput && batch.records.length > 1) {
        return {
          accepted: 0,
          rejected: 0,
          retry_individually: true,
        };
      }
      if (!retryable || attemptNumber >= maximumAttempts) break;
      await sleep(Math.min(8_000, 1_000 * 2 ** (attemptNumber - 1)));
    }
  }
  throw new Error(`批次 ${batchId} 翻译失败：${lastError?.message ?? "未知错误"}`);
}

function printStatus(ledger, outputState) {
  console.table([
    {
      free_package_tokens: ledger.free_package_tokens,
      local_safety_budget: ledger.safety_budget_tokens,
      accounted: ledger.accounted_tokens,
      provider_reported: ledger.reported_tokens,
      remaining_local_budget: Math.max(
        0,
        ledger.safety_budget_tokens - ledger.accounted_tokens,
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
  const model = parseModel(args.model);
  const safetyBudgetTokens = positiveInteger(
    args["safety-budget-tokens"],
    "--safety-budget-tokens",
    DEFAULT_SAFETY_BUDGET_TOKENS,
  );
  const maximumSourceCodepoints = positiveInteger(
    args["batch-source-codepoints"],
    "--batch-source-codepoints",
    DEFAULT_BATCH_SOURCE_CODEPOINTS,
  );
  if (maximumSourceCodepoints > 3_000) {
    throw new Error("--batch-source-codepoints 不能超过 3000");
  }
  const maximumBatchRecords = positiveInteger(
    args["batch-records"],
    "--batch-records",
    DEFAULT_BATCH_RECORDS,
  );
  if (maximumBatchRecords > 20) throw new Error("--batch-records 不能超过 20");
  const maximumOutputTokens = positiveInteger(
    args["max-output-tokens"],
    "--max-output-tokens",
    DEFAULT_MAX_OUTPUT_TOKENS,
  );
  if (maximumOutputTokens > 4_000) {
    throw new Error("Hy-MT2 最大输出为 4000 Tokens");
  }
  const maximumRecords = positiveInteger(
    args["max-records"],
    "--max-records",
    Number.MAX_SAFE_INTEGER,
  );
  const maximumAttempts = positiveInteger(
    args["max-attempts"],
    "--max-attempts",
    DEFAULT_MAX_ATTEMPTS,
  );
  const timeoutMilliseconds = positiveInteger(
    args["timeout-ms"],
    "--timeout-ms",
    DEFAULT_TIMEOUT_MS,
  );
  const delayMilliseconds = positiveInteger(
    args["delay-ms"],
    "--delay-ms",
    DEFAULT_DELAY_MS,
  );

  const ledger = loadLedger(safetyBudgetTokens);
  const outputState = loadOutputs();
  if (args.status) {
    printStatus(ledger, outputState);
    return;
  }

  const excludedIds = allKnownOutputIds(outputState);
  for (const id of attemptedRecordIds(ledger)) excludedIds.add(id);
  const groups = loadCandidateGroups(
    domains,
    directionNames,
    excludedIds,
    retryIndividuallyRecordIds(ledger),
  );
  const availableTokens = Math.max(
    0,
    ledger.safety_budget_tokens - ledger.accounted_tokens,
  );
  const plan = planTranslation({
    groups,
    availableTokens,
    maximumSourceCodepoints,
    maximumBatchRecords,
    maximumRecords,
    maximumOutputTokens,
  });
  console.table(
    groups.map((group) => {
      const planned = plan.byGroup.get(group.key);
      return {
        group: group.key,
        available_records: group.records.length,
        planned_records: planned.records,
        source_codepoints: planned.source_codepoints,
        reserved_tokens: planned.reserved_tokens,
      };
    }),
  );
  console.log(
    `保守预演：${plan.plannedRecords} 条，${plan.plannedSourceCodepoints.toLocaleString()} 个源文代码点，${plan.plannedReservedTokens.toLocaleString()} 个预留 Tokens，约 ${plan.plannedRequests} 个请求。`,
  );
  console.log(
    `账本已计 ${ledger.accounted_tokens.toLocaleString()}，本地安全上限 ${ledger.safety_budget_tokens.toLocaleString()} Tokens。`,
  );
  console.log("真实执行会按响应 usage 释放多余预留，因此通常可翻译更多记录。");
  if (!args.execute) {
    console.log("当前为预演模式，没有调用腾讯 API。确认免费体验和后付费状态后追加 --execute。");
    return;
  }
  if (!args["confirm-free-only-no-postpaid"]) {
    throw new Error(
      "执行前必须确认 Hy-MT2 免费体验已领取、剩余额度足够且后付费关闭；确认后追加 --confirm-free-only-no-postpaid",
    );
  }
  if (delayMilliseconds < 1_000) {
    throw new Error("执行模式下 --delay-ms 不能小于 1000");
  }
  const apiKey = process.env.TOKENHUB_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("缺少环境变量 TOKENHUB_API_KEY；不要把密钥写入命令参数或项目文件");
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
    persistLedger(ledger);
    while (processedRecords < maximumRecords) {
      const remainingTokens = ledger.safety_budget_tokens - ledger.accounted_tokens;
      if (remainingTokens < 1) break;
      const chosen = chooseNextBatch(
        executionGroups,
        groupIndex,
        maximumSourceCodepoints,
        maximumBatchRecords,
        maximumRecords - processedRecords,
        remainingTokens,
        maximumOutputTokens,
      );
      if (!chosen) break;
      groupIndex = chosen.nextGroupIndex;
      const { batch } = chosen;
      let counts = await executeBatch({
        batch,
        apiKey,
        model,
        ledger,
        outputState,
        maximumAttempts,
        maximumOutputTokens,
        timeoutMilliseconds,
      });
      if (counts.retry_individually) {
        console.log(
          `${batch.domain}/${batch.direction_name}: 批量结果无法可靠对齐，自动拆分 ${batch.records.length} 条重试`,
        );
        const combined = { accepted: 0, rejected: 0 };
        for (const record of batch.records) {
          await sleep(delayMilliseconds);
          const singleBatch = rebuildBatch(batch, [record], maximumOutputTokens);
          const singleCounts = await executeBatch({
            batch: singleBatch,
            apiKey,
            model,
            ledger,
            outputState,
            maximumAttempts,
            maximumOutputTokens,
            timeoutMilliseconds,
          });
          combined.accepted += singleCounts.accepted;
          combined.rejected += singleCounts.rejected;
        }
        counts = combined;
      }
      processedRecords += batch.records.length;
      console.log(
        `${batch.domain}/${batch.direction_name}: +${counts.accepted} 可导入，+${counts.rejected} 待处理；账本 ${ledger.accounted_tokens.toLocaleString()}/${ledger.safety_budget_tokens.toLocaleString()} Tokens`,
      );
      if (processedRecords < maximumRecords) await sleep(delayMilliseconds);
    }
  } finally {
    releaseLock();
  }
  printStatus(ledger, outputState);
  console.log("腾讯译文已保存，但尚未自动导入 cleaned JSON。请先检查 rejected 文件并抽样审核。");
}

const isDirectExecution =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectExecution) {
  main().catch((error) => {
    console.error(`腾讯批量翻译失败：${error.message}`);
    process.exitCode = 1;
  });
}
