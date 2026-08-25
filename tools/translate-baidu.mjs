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

const PROVIDER = "baidu_large_model_text_translation";
const PROVIDER_MODEL = "ai_text_translate";
const BAIDU_ENDPOINT = "https://fanyi-api.baidu.com/ait/api/aiTextTranslate";
const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_TIMEOUT_MS = 60_000;
const TRANSLATIONS_DIR = zhEnCrawledDirectory("translations");
const LOCK_FILE = path.join(TRANSLATIONS_DIR, ".baidu_translate.lock");

// The user has the authenticated advanced service. Keep 10% of its advertised
// one-million-character free allocation unused as a local safety buffer. This
// ledger intentionally does not reset each month: the public product page says
// "free characters" rather than promising a recurring monthly allocation.
const PLANS = {
  advanced: {
    free_package_codepoints: 1_000_000,
    default_safety_budget_codepoints: 900_000,
    maximum_query_codepoints: 6_000,
    default_request_codepoints: 1_800,
    default_delay_ms: 1_100,
    minimum_delay_ms: 120,
  },
};

const DIRECTIONS = {
  "zh-en": {
    source_lang: "zh-CN",
    target_lang: "en",
    source_api_lang: "zh",
    target_api_lang: "en",
  },
  "en-zh": {
    source_lang: "en",
    target_lang: "zh-CN",
    source_api_lang: "en",
    target_api_lang: "zh",
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

function parsePlan(value) {
  const planName = String(value ?? "advanced").toLowerCase();
  if (!PLANS[planName]) {
    throw new Error(`--plan 必须是 ${Object.keys(PLANS).join("、")} 之一`);
  }
  return { name: planName, ...PLANS[planName] };
}

const USAGE_FILE = path.join(TRANSLATIONS_DIR, "baidu_llm_usage.json");

function acceptedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.baidu_mt.json`);
}

function rejectedFile(domain) {
  return path.join(TRANSLATIONS_DIR, `${domain}.baidu_mt_rejected.json`);
}

function otherAcceptedFiles(domain) {
  return [
    path.join(TRANSLATIONS_DIR, `${domain}.google_mt.json`),
    path.join(TRANSLATIONS_DIR, `${domain}.tencent_mt.json`),
  ];
}

function readArray(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const value = readJson(filePath);
  if (!Array.isArray(value)) throw new Error(`${filePath} 顶层必须是数组`);
  return value;
}

function createLedger(plan, safetyBudgetCodepoints) {
  return {
    schema_version: 1,
    provider: PROVIDER,
    baidu_plan: plan.name,
    free_package_codepoints: plan.free_package_codepoints,
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

function loadLedger(plan, requestedSafetyBudget) {
  const ledger = fs.existsSync(USAGE_FILE)
    ? readJson(USAGE_FILE)
    : createLedger(plan, requestedSafetyBudget);
  if (ledger.provider !== PROVIDER) {
    throw new Error(`${USAGE_FILE} 的服务商不匹配`);
  }
  if (ledger.baidu_plan !== plan.name) {
    throw new Error(
      `${USAGE_FILE} 已按 ${ledger.baidu_plan} 版本建立，不能改用 ${plan.name}；请以百度控制台实际版本为准`,
    );
  }
  if (requestedSafetyBudget > plan.default_safety_budget_codepoints) {
    throw new Error(
      `${plan.name} 版本的本地安全预算不能超过 ${plan.default_safety_budget_codepoints.toLocaleString()} 字符`,
    );
  }
  const existingBudget = Number(ledger.safety_budget_codepoints);
  if (requestedSafetyBudget > existingBudget) {
    throw new Error(
      `百度免费包账本安全预算已经锁定为 ${existingBudget.toLocaleString()}，不能在脚本中调高`,
    );
  }
  ledger.safety_budget_codepoints = Math.min(existingBudget, requestedSafetyBudget);
  ledger.free_package_codepoints = plan.free_package_codepoints;
  ledger.attempted_codepoints ??= 0;
  ledger.successful_codepoints ??= 0;
  ledger.requests_attempted ??= 0;
  ledger.records_accepted ??= 0;
  ledger.records_rejected ??= 0;
  ledger.requests ??= [];
  return { filePath: USAGE_FILE, ledger };
}

function persistLedger(filePath, ledger) {
  ledger.updated_at = new Date().toISOString();
  writeJsonAtomic(filePath, ledger);
}

function loadOutputs() {
  const outputState = new Map();
  for (const domain of ZH_EN_DOMAINS) {
    outputState.set(domain, {
      accepted: readArray(acceptedFile(domain)),
      rejected: readArray(rejectedFile(domain)),
    });
  }
  return outputState;
}

function allKnownOutputIds(outputState) {
  const ids = new Set();
  for (const domain of ZH_EN_DOMAINS) {
    for (const filePath of otherAcceptedFiles(domain)) {
      for (const item of readArray(filePath)) ids.add(item.id);
    }
    const output = outputState.get(domain);
    for (const item of [...output.accepted, ...output.rejected]) ids.add(item.id);
  }
  return ids;
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
        ...direction,
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

function chooseNextRecord(groups, startIndex, remainingCodepoints) {
  for (let offset = 0; offset < groups.length; offset += 1) {
    const groupIndex = (startIndex + offset) % groups.length;
    const group = groups[groupIndex];
    const record = group.records[group.index];
    if (!record) continue;
    const inputCodepoints = countCodepoints(record.source_text);
    if (inputCodepoints > remainingCodepoints) continue;
    group.index += 1;
    return {
      group,
      record,
      inputCodepoints,
      nextGroupIndex: (groupIndex + 1) % groups.length,
    };
  }
  return null;
}

function findChunkBoundary(characters, maximumCodepoints) {
  const minimumPreferred = Math.floor(maximumCodepoints * 0.55);
  const punctuation = new Set(["。", "！", "？", "；", ".", "!", "?", ";"]);
  for (let index = maximumCodepoints - 1; index >= minimumPreferred; index -= 1) {
    if (punctuation.has(characters[index])) return index + 1;
  }
  for (let index = maximumCodepoints - 1; index >= minimumPreferred; index -= 1) {
    if (/\s/u.test(characters[index])) return index + 1;
  }
  return maximumCodepoints;
}

export function splitForBaidu(value, maximumCodepoints) {
  const text = normalizeText(value);
  const chunks = [];
  let remaining = [...text];
  while (remaining.length > maximumCodepoints) {
    const cutAt = findChunkBoundary(remaining, maximumCodepoints);
    const chunk = normalizeText(remaining.slice(0, cutAt).join(""));
    if (chunk) chunks.push(chunk);
    remaining = remaining.slice(cutAt);
    while (remaining.length > 0 && /\s/u.test(remaining[0])) remaining.shift();
  }
  const finalChunk = normalizeText(remaining.join(""));
  if (finalChunk) chunks.push(finalChunk);
  return chunks;
}

function planTranslation({
  groups,
  availableCodepoints,
  maximumRecords,
  requestCodepoints,
}) {
  const workingGroups = cloneGroups(groups);
  const byGroup = new Map(
    workingGroups.map((group) => [group.key, { records: 0, codepoints: 0, requests: 0 }]),
  );
  let plannedCodepoints = 0;
  let plannedRecords = 0;
  let plannedRequests = 0;
  let groupIndex = 0;
  while (plannedRecords < maximumRecords) {
    const chosen = chooseNextRecord(
      workingGroups,
      groupIndex,
      availableCodepoints - plannedCodepoints,
    );
    if (!chosen) break;
    groupIndex = chosen.nextGroupIndex;
    const requestCount = splitForBaidu(
      chosen.record.source_text,
      requestCodepoints,
    ).length;
    const stats = byGroup.get(chosen.group.key);
    stats.records += 1;
    stats.codepoints += chosen.inputCodepoints;
    stats.requests += requestCount;
    plannedCodepoints += chosen.inputCodepoints;
    plannedRecords += 1;
    plannedRequests += requestCount;
  }
  return { byGroup, plannedCodepoints, plannedRecords, plannedRequests };
}

function apiErrorMessage(status, body) {
  try {
    const parsed = JSON.parse(body);
    return `HTTP ${status}: ${parsed.error_msg ?? parsed.message ?? body.slice(0, 500)}`;
  } catch {
    return `HTTP ${status}: ${body.slice(0, 500)}`;
  }
}

export async function callBaiduTranslate({
  apiKey,
  appId,
  sourceLanguage,
  targetLanguage,
  text,
  timeoutMilliseconds = DEFAULT_TIMEOUT_MS,
  fetchImplementation = fetch,
  endpoint = BAIDU_ENDPOINT,
}) {
  const response = await fetchImplementation(endpoint, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json; charset=utf-8",
    },
    body: JSON.stringify({
      appid: appId,
      q: text,
      from: sourceLanguage,
      to: targetLanguage,
    }),
    signal: AbortSignal.timeout(timeoutMilliseconds),
  });
  const responseBody = await response.text();
  if (!response.ok) {
    const error = new Error(apiErrorMessage(response.status, responseBody));
    error.status = response.status;
    throw error;
  }
  let parsed;
  try {
    parsed = JSON.parse(responseBody);
  } catch {
    throw new Error(`百度 API 返回了无法解析的 JSON：${responseBody.slice(0, 300)}`);
  }
  if (parsed.error_code) {
    const error = new Error(`百度错误 ${parsed.error_code}: ${parsed.error_msg ?? "未知错误"}`);
    error.baiduCode = String(parsed.error_code);
    throw error;
  }
  if (!Array.isArray(parsed.trans_result) || parsed.trans_result.length === 0) {
    throw new Error("百度 API 响应中没有 trans_result");
  }
  const separator = targetLanguage === "zh" ? "\n" : " ";
  return normalizeText(parsed.trans_result.map((item) => item.dst ?? "").join(separator));
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

function outputRecord(record, targetText, recordBatchId, inputCodepoints, requestIds) {
  return {
    id: record.id,
    source_lang: record.source_lang,
    target_lang: record.target_lang,
    source_text: record.source_text,
    target_text: targetText,
    domain: record.domain,
    // The submission schema has no provider-neutral NMT value. Keep llm_mt for
    // schema compatibility and record the exact engine in provider fields.
    translation_method: "llm_mt",
    provider: PROVIDER,
    provider_model: PROVIDER_MODEL,
    provider_batch_id: recordBatchId,
    provider_request_ids: requestIds,
    input_codepoints: inputCodepoints,
    translated_at: new Date().toISOString(),
    review_status: "pending",
    review_notes: "",
  };
}

function appendOutput(outputState, group, record, targetText, batchId, inputCodepoints, requestIds) {
  const domainOutput = outputState.get(group.domain);
  const output = outputRecord(record, targetText, batchId, inputCodepoints, requestIds);
  const failure = qualityFailure(record, targetText);
  if (failure) {
    domainOutput.rejected.push({ ...output, rejection_reason: failure });
  } else {
    domainOutput.accepted.push(output);
  }
  writeJsonAtomic(acceptedFile(group.domain), domainOutput.accepted);
  writeJsonAtomic(rejectedFile(group.domain), domainOutput.rejected);
  return failure ? { accepted: 0, rejected: 1 } : { accepted: 1, rejected: 0 };
}

function appendProviderRejection(outputState, chosen, error) {
  const { group, record, inputCodepoints } = chosen;
  const domainOutput = outputState.get(group.domain);
  const batchId =
    error.providerBatchId ??
    `baidu_record_${sha256(`${group.domain}\n${group.direction_name}\n${record.id}`).slice(0, 20)}`;
  domainOutput.rejected.push({
    id: record.id,
    source_lang: record.source_lang,
    target_lang: record.target_lang,
    source_text: record.source_text,
    target_text: "",
    domain: record.domain,
    translation_method: "llm_mt",
    provider: PROVIDER,
    provider_model: PROVIDER_MODEL,
    provider_batch_id: batchId,
    provider_request_ids: error.providerRequestIds ?? [],
    provider_error_code: String(error.baiduCode ?? "unknown"),
    provider_error_message: String(error.message).slice(0, 1000),
    input_codepoints: inputCodepoints,
    translated_at: null,
    review_status: "rejected",
    review_notes: "",
    rejection_reason: `provider_content_policy_${error.baiduCode ?? "unknown"}`,
    retranslation_required: true,
  });
  writeJsonAtomic(rejectedFile(group.domain), domainOutput.rejected);
  return { accepted: 0, rejected: 1 };
}

function isRecordLevelProviderRejection(error) {
  return String(error?.baiduCode ?? "") === "20003";
}

function isRetryable(error) {
  return (
    error?.name === "TimeoutError" ||
    error?.name === "AbortError" ||
    error?.status === 429 ||
    (Number(error?.status) >= 500 && Number(error?.status) <= 599) ||
    ["52001", "52002", "54003"].includes(String(error?.baiduCode ?? "")) ||
    /fetch failed|ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|socket/i.test(
      String(error?.message ?? ""),
    )
  );
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
        `检测到翻译锁文件 ${LOCK_FILE}；确认没有其他百度翻译进程后再手动删除该锁文件`,
      );
    }
    throw error;
  }
}

function releaseLock() {
  fs.rmSync(LOCK_FILE, { force: true });
}

async function executeChunk({
  apiKey,
  appId,
  group,
  record,
  chunk,
  chunkIndex,
  totalChunks,
  batchId,
  ledger,
  ledgerFile,
  maximumAttempts,
  timeoutMilliseconds,
  waitForRateLimit,
}) {
  const inputCodepoints = countCodepoints(chunk);
  let lastError;
  let lastRequestId = null;
  for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
    if (ledger.attempted_codepoints + inputCodepoints > ledger.safety_budget_codepoints) {
      throw new Error("本月本地安全预算不足以发送或重试当前文本块，已安全停止");
    }
    await waitForRateLimit();
    const request = {
      request_id: `baidu_request_${sha256(
        `${batchId}\n${chunkIndex}\n${attempt}\n${Date.now()}`,
      ).slice(0, 20)}`,
      batch_id: batchId,
      attempt_number: attempt,
      domain: group.domain,
      direction: group.direction_name,
      record_id: record.id,
      chunk_index: chunkIndex,
      total_chunks: totalChunks,
      input_codepoints: inputCodepoints,
      status: "attempting",
      attempted_at: new Date().toISOString(),
    };
    ledger.requests.push(request);
    lastRequestId = request.request_id;
    ledger.attempted_codepoints += inputCodepoints;
    ledger.requests_attempted += 1;
    persistLedger(ledgerFile, ledger);
    try {
      const targetText = await callBaiduTranslate({
        apiKey,
        appId,
        sourceLanguage: group.source_api_lang,
        targetLanguage: group.target_api_lang,
        text: chunk,
        timeoutMilliseconds,
      });
      request.status = "completed";
      request.completed_at = new Date().toISOString();
      ledger.successful_codepoints += inputCodepoints;
      persistLedger(ledgerFile, ledger);
      return { targetText, requestId: request.request_id };
    } catch (error) {
      lastError = error;
      request.status = isRetryable(error) ? "failed_retryable" : "failed_terminal";
      request.failed_at = new Date().toISOString();
      request.error = String(error.message).slice(0, 1000);
      persistLedger(ledgerFile, ledger);
      if (!isRetryable(error) || attempt >= maximumAttempts) break;
      await sleep(Math.min(8_000, 1_000 * 2 ** (attempt - 1)));
    }
  }
  const failure = new Error(
    `记录 ${record.id} 的第 ${chunkIndex + 1} 个文本块翻译失败：${lastError?.message ?? "未知错误"}`,
  );
  failure.baiduCode = lastError?.baiduCode;
  failure.providerRequestId = lastRequestId;
  throw failure;
}

async function executeRecord({
  chosen,
  apiKey,
  appId,
  requestCodepoints,
  ledger,
  ledgerFile,
  outputState,
  maximumAttempts,
  timeoutMilliseconds,
  waitForRateLimit,
}) {
  const { group, record, inputCodepoints } = chosen;
  const chunks = splitForBaidu(record.source_text, requestCodepoints);
  const batchId = `baidu_record_${sha256(
    `${group.domain}\n${group.direction_name}\n${record.id}`,
  ).slice(0, 20)}`;
  const targetChunks = [];
  const requestIds = [];
  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    let result;
    try {
      result = await executeChunk({
        apiKey,
        appId,
        group,
        record,
        chunk: chunks[chunkIndex],
        chunkIndex,
        totalChunks: chunks.length,
        batchId,
        ledger,
        ledgerFile,
        maximumAttempts,
        timeoutMilliseconds,
        waitForRateLimit,
      });
    } catch (error) {
      error.providerBatchId = batchId;
      error.providerRequestIds = [
        ...requestIds,
        ...(error.providerRequestId ? [error.providerRequestId] : []),
      ];
      throw error;
    }
    targetChunks.push(result.targetText);
    requestIds.push(result.requestId);
  }
  const targetText = normalizeText(targetChunks.join("\n"));
  const counts = appendOutput(
    outputState,
    group,
    record,
    targetText,
    batchId,
    inputCodepoints,
    requestIds,
  );
  ledger.records_accepted += counts.accepted;
  ledger.records_rejected += counts.rejected;
  persistLedger(ledgerFile, ledger);
  return counts;
}

function printStatus(ledger, outputState) {
  console.table([
    {
      baidu_plan: ledger.baidu_plan,
      free_package: ledger.free_package_codepoints,
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
  const plan = parsePlan(args.plan);
  const safetyBudgetCodepoints = positiveInteger(
    args["safety-budget-codepoints"],
    "--safety-budget-codepoints",
    plan.default_safety_budget_codepoints,
  );
  const requestCodepoints = positiveInteger(
    args["request-codepoints"],
    "--request-codepoints",
    plan.default_request_codepoints,
  );
  if (requestCodepoints > plan.maximum_query_codepoints) {
    throw new Error(
      `${plan.name} 版本单次请求不能超过 ${plan.maximum_query_codepoints.toLocaleString()} 字符`,
    );
  }
  const delayMilliseconds = positiveInteger(
    args["delay-ms"],
    "--delay-ms",
    plan.default_delay_ms,
  );
  if (delayMilliseconds < plan.minimum_delay_ms) {
    throw new Error(
      `${plan.name} 版本为避免超过 QPS，--delay-ms 不能低于 ${plan.minimum_delay_ms}`,
    );
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

  const { filePath: ledgerFile, ledger } = loadLedger(plan, safetyBudgetCodepoints);
  const outputState = loadOutputs();
  if (args.status) {
    printStatus(ledger, outputState);
    return;
  }

  const excludedIds = allKnownOutputIds(outputState);
  const groups = loadCandidateGroups(domains, directionNames, excludedIds);
  const availableCodepoints = Math.max(
    0,
    ledger.safety_budget_codepoints - ledger.attempted_codepoints,
  );
  const planned = planTranslation({
    groups,
    availableCodepoints,
    maximumRecords,
    requestCodepoints,
  });
  console.table(
    groups.map((group) => {
      const stats = planned.byGroup.get(group.key);
      return {
        group: group.key,
        available_records: group.records.length,
        planned_records: stats.records,
        planned_codepoints: stats.codepoints,
        planned_requests: stats.requests,
      };
    }),
  );
  console.log(
    `百度 ${plan.name} 版预演：${planned.plannedRecords} 条，${planned.plannedCodepoints.toLocaleString()} 个源文代码点，约 ${planned.plannedRequests} 个请求。`,
  );
  console.log(
    `免费包账本已计 ${ledger.attempted_codepoints.toLocaleString()}，本地安全上限 ${ledger.safety_budget_codepoints.toLocaleString()} 字符。`,
  );
  if (!args.execute) {
    console.log("当前为预演模式，没有调用百度 API。确认控制台版本和剩余额度后追加 --execute。 ");
    return;
  }
  if (!args["confirm-baidu-plan-and-free-quota"]) {
    throw new Error(
      "执行前必须确认 --plan 与百度控制台一致、免费额度尚未被其他项目占用且未开启超额付费；确认后追加 --confirm-baidu-plan-and-free-quota",
    );
  }
  const apiKey = process.env.BAIDU_TRANSLATE_API_KEY?.trim();
  const appId = process.env.BAIDU_TRANSLATE_APP_ID?.trim();
  if (!apiKey || !appId) {
    throw new Error(
      "缺少 BAIDU_TRANSLATE_API_KEY 或 BAIDU_TRANSLATE_APP_ID；API Key 在“API Key 管理”，APP ID 在“开发者中心 -> 开发者信息”",
    );
  }
  if (planned.plannedRecords === 0) {
    console.log("没有可执行的百度翻译记录。 ");
    return;
  }

  acquireLock();
  const executionGroups = cloneGroups(groups);
  let groupIndex = 0;
  let processedRecords = 0;
  let nextRequestAt = 0;
  const waitForRateLimit = async () => {
    const delay = nextRequestAt - Date.now();
    if (delay > 0) await sleep(delay);
    nextRequestAt = Date.now() + delayMilliseconds;
  };
  try {
    persistLedger(ledgerFile, ledger);
    while (processedRecords < maximumRecords) {
      const remainingCodepoints =
        ledger.safety_budget_codepoints - ledger.attempted_codepoints;
      const chosen = chooseNextRecord(executionGroups, groupIndex, remainingCodepoints);
      if (!chosen) break;
      groupIndex = chosen.nextGroupIndex;
      let counts;
      try {
        counts = await executeRecord({
          chosen,
          apiKey,
          appId,
          requestCodepoints,
          ledger,
          ledgerFile,
          outputState,
          maximumAttempts,
          timeoutMilliseconds,
          waitForRateLimit,
        });
      } catch (error) {
        if (!isRecordLevelProviderRejection(error)) throw error;
        counts = appendProviderRejection(outputState, chosen, error);
        ledger.records_rejected += 1;
        persistLedger(ledgerFile, ledger);
        console.warn(
          `${chosen.group.key}: ${chosen.record.id} 被百度内容策略拒绝（${error.baiduCode}），已标记为需要其他服务重新翻译并继续。`,
        );
      }
      processedRecords += 1;
      console.log(
        `${chosen.group.key}: +${counts.accepted} 可导入，+${counts.rejected} 待处理；账本 ${ledger.attempted_codepoints.toLocaleString()}/${ledger.safety_budget_codepoints.toLocaleString()} 字符`,
      );
    }
  } finally {
    releaseLock();
  }
  printStatus(ledger, outputState);
  console.log("百度译文已保存，但尚未自动导入 cleaned JSON。 ");
}

const isDirectExecution =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectExecution) {
  main().catch((error) => {
    console.error(`百度批量翻译失败：${error.message}`);
    process.exitCode = 1;
  });
}
