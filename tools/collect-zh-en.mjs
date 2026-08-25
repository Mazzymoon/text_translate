#!/usr/bin/env node

import path from "node:path";
import { load } from "cheerio";
import {
  DATASET_ROOT,
  zhEnCrawledDirectory,
  ZH_EN_DOMAINS,
  countEnglishWords,
  countHanCharacters,
  emptyZhEnDocument,
  isIsoDateInRange,
  normalizeText,
  parseArguments,
  readJson,
  requireZhEnDomain,
  sha256,
  sleep,
  writeJsonAtomic,
  zhEnDataFile,
} from "./lib/corpus.mjs";

const SOURCE_FILE = path.join(zhEnCrawledDirectory("source"), "zh_en_sources.json");
const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
  "Chrome/131.0.0.0 Safari/537.36 BilingualCorpusStudentProject/2.0";
let requestDelayMilliseconds = 800;
const lastRequestStartedByHost = new Map();

async function waitForHost(url) {
  const host = new URL(url).host;
  const lastStartedAt = lastRequestStartedByHost.get(host) ?? 0;
  const waitMilliseconds = Math.max(
    0,
    requestDelayMilliseconds - (Date.now() - lastStartedAt),
  );
  if (waitMilliseconds > 0) await sleep(waitMilliseconds);
  lastRequestStartedByHost.set(host, Date.now());
}

function normalizeDate(value) {
  const text = String(value ?? "").trim();
  const numeric = text.match(/(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})/);
  if (numeric) {
    return `${numeric[1]}-${numeric[2].padStart(2, "0")}-${numeric[3].padStart(2, "0")}`;
  }
  const parsed = new Date(text);
  return Number.isNaN(parsed.valueOf()) ? null : parsed.toISOString().slice(0, 10);
}

async function fetchText(url, retries = 2, init = {}) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      await waitForHost(url);
      const response = await fetch(url, {
        ...init,
        headers: {
          "user-agent": USER_AGENT,
          accept: "text/html,application/xhtml+xml,application/json",
          "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
          ...init.headers,
        },
        redirect: "follow",
        signal: init.signal ?? AbortSignal.timeout(30_000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const maximumResponseBytes = 10 * 1024 * 1024;
      const declaredLength = Number(response.headers.get("content-length") ?? 0);
      if (declaredLength > maximumResponseBytes) {
        throw new Error(`响应正文超过 10 MB（${declaredLength} bytes）`);
      }
      const text = await response.text();
      if (Buffer.byteLength(text, "utf8") > maximumResponseBytes) {
        throw new Error("响应正文超过 10 MB");
      }
      return { text, finalUrl: response.url, status: response.status };
    } catch (error) {
      lastError = error;
      if (attempt < retries) await sleep(1000 * (attempt + 1));
    }
  }
  throw new Error(`${url} 抓取失败：${lastError.message}`);
}

function canonicalizeUrl(value) {
  const url = new URL(value);
  url.hash = "";
  for (const key of [...url.searchParams.keys()]) {
    if (/^(utm_|spm|from|eqid)/i.test(key)) url.searchParams.delete(key);
  }
  return url.toString();
}

function publicationDateHintFromUrl(value) {
  const pathname = new URL(value).pathname;
  const dashed = pathname.match(/\/(20\d{2})-(\d{2})\/(\d{2})\//);
  if (dashed) return { exact: `${dashed[1]}-${dashed[2]}-${dashed[3]}` };
  const yearMonthDay = pathname.match(/\/(20\d{2})\/(\d{2})(\d{2})\//);
  if (yearMonthDay) {
    return { exact: `${yearMonthDay[1]}-${yearMonthDay[2]}-${yearMonthDay[3]}` };
  }
  const compactYearMonthDay = pathname.match(/\/(20\d{2})(\d{2})(\d{2})\//);
  if (compactYearMonthDay) {
    return {
      exact: `${compactYearMonthDay[1]}-${compactYearMonthDay[2]}-${compactYearMonthDay[3]}`,
    };
  }
  const separated = pathname.match(/\/(20\d{2})\/(\d{1,2})\/(\d{1,2})\//);
  if (separated) {
    return {
      exact: `${separated[1]}-${separated[2].padStart(2, "0")}-${separated[3].padStart(2, "0")}`,
    };
  }
  const timestamp = pathname.match(/\/(20\d{6})\d{4,}\//);
  if (timestamp) {
    return {
      exact: `${timestamp[1].slice(0, 4)}-${timestamp[1].slice(4, 6)}-${timestamp[1].slice(6, 8)}`,
    };
  }
  const yearMonth = pathname.match(/\/(20\d{4})\//);
  if (yearMonth) {
    return { month: `${yearMonth[1].slice(0, 4)}-${yearMonth[1].slice(4, 6)}` };
  }
  return null;
}

function urlDateIsOutsideWindow(value, startDate, endDate) {
  const hint = publicationDateHintFromUrl(value);
  if (!hint) return false;
  if (hint.exact) return hint.exact < startDate || hint.exact > endDate;
  return hint.month < startDate.slice(0, 7) || hint.month > endDate.slice(0, 7);
}

function extractMeta($, names) {
  for (const name of names) {
    const value = $(`meta[name="${name}"], meta[property="${name}"]`).first().attr("content");
    if (value?.trim()) return normalizeText(value);
  }
  return null;
}

function sourceUrlIsExcluded(source, value) {
  const url = new URL(value);
  if (
    url.hostname === "www.gov.uk" &&
    url.pathname.startsWith("/government/statistics/announcements/")
  ) {
    return true;
  }
  return source.article_url_exclude_pattern
    ? new RegExp(source.article_url_exclude_pattern, "i").test(value)
    : false;
}

function discoverGovUkHtmlAttachments(html, pageUrl, source) {
  const page = new URL(pageUrl);
  if (page.hostname !== "www.gov.uk") return [];
  const rootMatch = page.pathname.match(
    /^\/government\/(?:publications|statistics)\/[^/]+/,
  );
  if (!rootMatch) return [];

  const rootPath = rootMatch[0];
  const pattern = new RegExp(source.article_url_pattern);
  const $ = load(html);
  return [
    ...new Set(
      $("a[href]")
        .map((_, element) => {
          const href = $(element).attr("href")?.trim();
          if (!href || (!href.startsWith("/") && !/^https?:\/\//i.test(href))) {
            return null;
          }
          try {
            return canonicalizeUrl(new URL(href, page).toString());
          } catch {
            return null;
          }
        })
        .get()
        .filter((url) => {
          if (!url) return false;
          const candidate = new URL(url);
          return (
            candidate.hostname === "www.gov.uk" &&
            candidate.pathname.startsWith(`${rootPath}/`) &&
            candidate.pathname !== page.pathname &&
            !/\.(?:pdf|docx?|xlsx?|csv|ods|zip)$/i.test(candidate.pathname) &&
            pattern.test(url) &&
            !sourceUrlIsExcluded(source, url)
          );
        }),
    ),
  ].slice(0, 10);
}

function extractArticle(html, source) {
  const $ = load(html);
  const title =
    extractMeta($, ["ArticleTitle", "og:title", "twitter:title"]) ||
    normalizeText($("h1").first().text()) ||
    normalizeText($("title").text());
  const publishedAt = normalizeDate(
    extractMeta($, [
      "publishdate",
      "PubDate",
      "article:published_time",
      "date",
      "dcterms.date",
      "govuk:first-published-at",
      "govuk:public-updated-at",
    ]) ??
      $("time").first().attr("datetime") ??
      $("time").first().text() ??
      $("body").text().match(/20\d{2}[-年]\d{1,2}[-月]\d{1,2}/)?.[0],
  );
  const author = extractMeta($, ["author", "byl", "dcterms.creator"]);
  const sourceName = extractMeta($, ["source", "ContentSource"]) ?? source.site_name;

  let body = null;
  const minimumBodyCharacters = source.source_lang === "zh-CN" ? 120 : 300;
  for (const selector of source.content_selectors ?? []) {
    const candidate = $(selector).first();
    if (
      candidate.length > 0 &&
      normalizeText(candidate.text()).length >= minimumBodyCharacters
    ) {
      body = candidate.clone();
      break;
    }
  }
  if (!body) throw new Error("没有找到足够长的正文区域");

  body
    .find(
      "script,style,noscript,svg,nav,form,button,figure,video,audio," +
        ".navigation,.breadcrumb,.share,.related-content,.gem-c-related-navigation",
    )
    .remove();
  const paragraphTexts = body
    .find("p,li")
    .map((_, element) => normalizeText($(element).text()))
    .get()
    .filter((text) => text.length >= 20)
    .filter((text) => !/^(Contact|Tags|Page Last Reviewed|责任编辑|下载|收藏)[:：]?/i.test(text));
  const rawText = normalizeText(
    paragraphTexts.length > 0 ? paragraphTexts.join("\n") : body.text(),
  );
  if (source.source_lang === "zh-CN" && countHanCharacters(rawText) < 100) {
    throw new Error("提取后的中文正文不足 100 个汉字");
  }
  if (source.source_lang === "en" && countEnglishWords(rawText) < 60) {
    throw new Error("提取后的英文正文不足 60 个单词");
  }

  return {
    title,
    published_at: publishedAt,
    author,
    source_name: sourceName,
    raw_content_html: body.html() ?? "",
    raw_text: rawText,
  };
}

function extractListingUrls(html, baseUrl, source) {
  const $ = load(html);
  const pattern = new RegExp(source.article_url_pattern);
  const titlePattern = source.title_include_pattern
    ? new RegExp(source.title_include_pattern, "i")
    : null;
  const titleExcludePattern = source.title_exclude_pattern
    ? new RegExp(source.title_exclude_pattern, "i")
    : null;

  return $("a[href]")
    .map((_, element) => {
      const title = normalizeText($(element).attr("title") ?? $(element).text());
      if (titlePattern && !titlePattern.test(title)) return null;
      if (titleExcludePattern && titleExcludePattern.test(title)) return null;
      try {
        return canonicalizeUrl(new URL($(element).attr("href"), baseUrl).toString());
      } catch {
        return null;
      }
    })
    .get()
    .filter((url) => url && pattern.test(url) && !sourceUrlIsExcluded(source, url));
}

function expandStaticListingPages(source, maximumListingPages) {
  const pages = [source.url];
  const pagination = source.pagination;
  if (!pagination?.url_template) return pages;

  const startPage = Number(pagination.start_page ?? 1);
  const endPage = Number(pagination.end_page ?? startPage);
  if (!Number.isInteger(startPage) || !Number.isInteger(endPage) || endPage < startPage) {
    throw new Error(`${source.id} 的静态分页配置无效`);
  }
  for (
    let page = startPage;
    page <= endPage && pages.length < maximumListingPages;
    page += 1
  ) {
    pages.push(pagination.url_template.replaceAll("{page}", String(page)));
  }
  return [...new Set(pages.map(canonicalizeUrl))];
}

async function discoverStaticListingUrls(source, options) {
  const listingPages = expandStaticListingPages(source, options.maximumListingPages);
  const discovered = [];
  for (let index = 0; index < listingPages.length; index += 1) {
    const listingUrl = listingPages[index];
    try {
      const { text, finalUrl } = await fetchText(listingUrl);
      discovered.push(...extractListingUrls(text, finalUrl, source));
    } catch (error) {
      if (index === 0) throw error;
      console.warn(`跳过列表页 ${listingUrl}: ${error.message}`);
      break;
    }
  }
  return [...new Set(discovered)];
}

async function discoverJsonFragmentListingUrls(source, options) {
  const pagination = source.pagination ?? {};
  const startPage = Number(pagination.start_page ?? 1);
  const configuredEndPage = Number(pagination.end_page ?? startPage);
  const endPage = Math.min(
    configuredEndPage,
    startPage + options.maximumListingPages - 1,
  );
  const pageSize = Number(pagination.page_size ?? 20);
  if (
    !Number.isInteger(startPage) ||
    !Number.isInteger(endPage) ||
    !Number.isInteger(pageSize) ||
    endPage < startPage ||
    pageSize < 1
  ) {
    throw new Error(`${source.id} 的 JSON 片段分页配置无效`);
  }

  const discovered = [];
  for (let page = startPage; page <= endPage; page += 1) {
    const requestUrl = new URL(source.url);
    for (const [key, value] of Object.entries(source.query ?? {})) {
      requestUrl.searchParams.set(key, String(value));
    }
    if (!(pagination.first_page_without_parameter && page === startPage)) {
      requestUrl.searchParams.set(
        pagination.parameter ?? "paramJson",
        JSON.stringify({ pageNo: page, pageSize }),
      );
    }
    const { text } = await fetchText(requestUrl.toString());
    const payload = JSON.parse(text);
    const fragment = payload?.data?.html;
    if (typeof fragment !== "string") {
      throw new Error(`${source.id} 第 ${page} 页没有返回 data.html`);
    }
    discovered.push(
      ...extractListingUrls(fragment, source.listing_base_url ?? source.url, source),
    );
  }
  return [...new Set(discovered)];
}

async function discoverJsonListingUrls(source) {
  const { text, finalUrl } = await fetchText(source.url);
  const payload = JSON.parse(text);
  const pattern = new RegExp(source.article_url_pattern);
  const titlePattern = source.title_include_pattern
    ? new RegExp(source.title_include_pattern, "i")
    : null;
  const titleExcludePattern = source.title_exclude_pattern
    ? new RegExp(source.title_exclude_pattern, "i")
    : null;
  return [
    ...new Set(
      (payload.results ?? [])
        .filter((result) => !titlePattern || titlePattern.test(result.title ?? ""))
        .filter((result) => !titleExcludePattern || !titleExcludePattern.test(result.title ?? ""))
        .map((result) => {
          try {
            return canonicalizeUrl(new URL(result.link, finalUrl).toString());
          } catch {
            return null;
          }
        })
        .filter(
          (url) => url && pattern.test(url) && !sourceUrlIsExcluded(source, url),
        ),
    ),
  ];
}

async function discoverJsonPostListingUrls(source, options) {
  const pattern = new RegExp(source.article_url_pattern);
  const titlePattern = source.title_include_pattern
    ? new RegExp(source.title_include_pattern, "i")
    : null;
  const titleExcludePattern = source.title_exclude_pattern
    ? new RegExp(source.title_exclude_pattern, "i")
    : null;
  const pagination = source.pagination ?? {};
  const startPage = Number(pagination.start_page ?? 1);
  const configuredEndPage = Number(pagination.end_page ?? startPage);
  const endPage = Math.min(
    configuredEndPage,
    startPage + options.maximumListingPages - 1,
  );
  const discovered = [];

  if (source.listing_base_url) {
    const { text, finalUrl } = await fetchText(source.listing_base_url);
    discovered.push(...extractListingUrls(text, finalUrl, source));
  }

  for (let page = startPage; page <= endPage; page += 1) {
    const form = new URLSearchParams();
    for (const [key, value] of Object.entries(source.form ?? {})) {
      form.set(key, String(value).replaceAll("{page}", String(page)));
    }
    const { text } = await fetchText(source.url, 2, {
      method: "POST",
      headers: {
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "x-requested-with": "XMLHttpRequest",
        ...(source.listing_base_url ? { referer: source.listing_base_url } : {}),
      },
      body: form,
    });
    const payload = JSON.parse(text);
    const results = payload[source.results_field ?? "list"];
    if (!Array.isArray(results)) {
      throw new Error(`${source.id} 第 ${page} 页没有返回列表数组`);
    }
    for (const result of results) {
      const title = normalizeText(result[source.title_field ?? "topic"]);
      if (titlePattern && !titlePattern.test(title)) continue;
      if (titleExcludePattern && titleExcludePattern.test(title)) continue;
      try {
        const url = canonicalizeUrl(
          new URL(
            result[source.url_field ?? "infourl"],
            source.listing_base_url ?? source.url,
          ).toString(),
        );
        if (pattern.test(url) && !sourceUrlIsExcluded(source, url)) discovered.push(url);
      } catch {
        // Ignore malformed entries while retaining the rest of the page.
      }
    }
    if (results.length === 0) break;
  }
  return [...new Set(discovered)];
}

async function discoverUrls(source, options) {
  if (source.kind === "article") {
    return (source.urls ?? [source.url]).filter(Boolean).map(canonicalizeUrl);
  }
  if (source.kind === "listing") return discoverStaticListingUrls(source, options);
  if (source.kind === "json_fragment_listing") {
    return discoverJsonFragmentListingUrls(source, options);
  }
  if (source.kind === "json_listing") return discoverJsonListingUrls(source);
  if (source.kind === "json_post_listing") {
    return discoverJsonPostListingUrls(source, options);
  }
  throw new Error(
    `${source.id} 的 kind 必须是 listing、json_listing、json_post_listing、json_fragment_listing 或 article`,
  );
}

function interleaveSourceQueues(sourceQueues) {
  const queues = sourceQueues.map((entry) => ({ ...entry, index: 0 }));
  const result = [];
  let added = true;
  while (added) {
    added = false;
    for (const queue of queues) {
      if (queue.index < queue.urls.length) {
        result.push({ url: queue.urls[queue.index], source: queue.source });
        queue.index += 1;
        added = true;
      }
    }
  }
  return result;
}

function sortRawRecords(records) {
  return records.sort((left, right) =>
    [left.source_lang, left.source_id, left.published_at, left.canonical_url]
      .join("|")
      .localeCompare(
        [right.source_lang, right.source_id, right.published_at, right.canonical_url].join("|"),
      ),
  );
}

function createStats(source) {
  return {
    source_id: source.id,
    source_language: source.source_lang,
    discovered: 0,
    queued: 0,
    cached: 0,
    fetched: 0,
    saved: 0,
    duplicate: 0,
    date_rejected: 0,
    extraction_failed: 0,
  };
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  const domain = requireZhEnDomain(args.domain);
  const rawFile = zhEnDataFile("raw", domain);
  const maximumPerLanguage = args["max-articles-per-language"]
    ? Number(args["max-articles-per-language"])
    : Number.MAX_SAFE_INTEGER;
  const maximumListingPages = Number(args["max-listing-pages"] ?? 100);
  const delayMilliseconds = Number(args["delay-ms"] ?? 800);
  const checkpointEvery = Number(args["checkpoint-every"] ?? 100);
  const refresh = Boolean(args.refresh);
  const replaceDomain = Boolean(args["replace-domain"]);
  const discoverOnly = Boolean(args["discover-only"]);
  for (const [name, value, minimum] of [
    ["--max-articles-per-language", maximumPerLanguage, 1],
    ["--max-listing-pages", maximumListingPages, 1],
    ["--delay-ms", delayMilliseconds, 0],
    ["--checkpoint-every", checkpointEvery, 1],
  ]) {
    if (!Number.isInteger(value) || value < minimum) {
      throw new Error(`${name} 必须是${minimum === 0 ? "非负" : "正"}整数`);
    }
  }
  if (delayMilliseconds < 800) {
    console.warn("警告：正式采集建议保持 --delay-ms 不低于 800。当前值仅适合来源发现测试。");
  }
  requestDelayMilliseconds = delayMilliseconds;

  const config = readJson(SOURCE_FILE);
  const raw = replaceDomain
    ? emptyZhEnDocument(domain)
    : readJson(rawFile, emptyZhEnDocument(domain));
  if (raw.domain && raw.domain !== domain) {
    throw new Error(`${rawFile} 的 domain=${raw.domain} 与命令参数不一致`);
  }
  raw.schema_version = 2;
  raw.domain = domain;
  raw.records = raw.records.filter((record) => record.domain === domain);
  const otherDomainUrls = new Set(
    ZH_EN_DOMAINS.filter((otherDomain) => otherDomain !== domain).flatMap(
      (otherDomain) =>
        readJson(
          zhEnDataFile("raw", otherDomain),
          emptyZhEnDocument(otherDomain),
        ).records.map((record) => record.canonical_url),
    ),
  );
  const rawCountBeforeCrossDomainDedupe = raw.records.length;
  raw.records = raw.records.filter(
    (record) => !otherDomainUrls.has(record.canonical_url),
  );
  if (raw.records.length !== rawCountBeforeCrossDomainDedupe) {
    console.warn(
      `${domain}: removed ${rawCountBeforeCrossDomainDedupe - raw.records.length} raw pages already owned by another domain`,
    );
  }
  const configuredSourcesById = new Map(
    config.sources
      .filter((source) => source.domain === domain)
      .map((source) => [source.id, source]),
  );
  raw.records = raw.records.map((record) => {
    const source = configuredSourcesById.get(record.source_id);
    return {
      ...record,
      rights_evidence_url: record.rights_evidence_url ?? source?.rights_evidence_url ?? null,
      license_name: record.license_name ?? source?.license_name ?? null,
      license_url: record.license_url ?? source?.license_url ?? null,
      attribution: record.attribution ?? source?.attribution ?? null,
    };
  });

  const requestedSourceId = args["source-id"] ? String(args["source-id"]) : null;
  const requestedSourceLanguage = args["source-language"]
    ? String(args["source-language"])
    : null;
  if (requestedSourceLanguage && !["zh-CN", "en"].includes(requestedSourceLanguage)) {
    throw new Error("--source-language 必须是 zh-CN 或 en");
  }
  const sources = config.sources.filter(
    (source) =>
      source.enabled &&
      source.domain === domain &&
      source.rights_status !== "prohibited" &&
      (!requestedSourceId || source.id === requestedSourceId) &&
      (!requestedSourceLanguage || source.source_lang === requestedSourceLanguage),
  );
  if (sources.length === 0) throw new Error(`没有为 ${domain} 配置可用来源`);

  const statsBySource = new Map(sources.map((source) => [source.id, createStats(source)]));
  const sourceQueues = [];
  const globallyDiscovered = new Set();
  for (const source of sources) {
    try {
      const urls = await discoverUrls(source, { maximumListingPages, delayMilliseconds });
      const stats = statsBySource.get(source.id);
      stats.discovered = urls.length;
      const uniqueUrls = [];
      for (const url of urls) {
        if (globallyDiscovered.has(url) || otherDomainUrls.has(url)) {
          stats.duplicate += 1;
          continue;
        }
        globallyDiscovered.add(url);
        uniqueUrls.push(url);
      }
      stats.queued = uniqueUrls.length;
      sourceQueues.push({ source, urls: uniqueUrls });
      console.log(`${source.id}: 发现 ${urls.length} 个候选网页`);
    } catch (error) {
      console.warn(`跳过来源 ${source.id}: ${error.message}`);
    }
  }

  if (discoverOnly) {
    console.table([...statsBySource.values()]);
    console.log(
      `来源发现检查完成，共 ${globallyDiscovered.size} 个去重候选网页；未修改原始数据。`,
    );
    return;
  }

  const recordsByUrl = new Map(raw.records.map((record) => [record.canonical_url, record]));
  const recordsById = new Map(raw.records.map((record) => [record.id, record]));
  const savedCounts = new Map(
    ["zh-CN", "en"].map((language) => [
      language,
      raw.records.filter((record) => record.source_lang === language).length,
    ]),
  );
  let newlySavedSinceCheckpoint = 0;

  const persist = () => {
    raw.records = sortRawRecords([...recordsById.values()]);
    raw.generated_at = new Date().toISOString();
    raw.collection_summary = {
      collected_at: raw.generated_at,
      source_stats: [...statsBySource.values()],
      language_counts: Object.fromEntries(savedCounts),
    };
    writeJsonAtomic(rawFile, raw);
  };

  const workItems = interleaveSourceQueues(sourceQueues);
  const workItemUrls = new Set(workItems.map((item) => item.url));
  const enqueueAttachments = (html, pageUrl, source, currentIndex) => {
    const additions = discoverGovUkHtmlAttachments(html, pageUrl, source).filter(
      (url) => !workItemUrls.has(url) && !recordsByUrl.has(url),
    );
    if (additions.length === 0) return;
    const stats = statsBySource.get(source.id);
    stats.discovered += additions.length;
    stats.queued += additions.length;
    for (const url of additions) workItemUrls.add(url);
    workItems.splice(
      currentIndex + 1,
      0,
      ...additions.map((url) => ({ url, source })),
    );
  };

  for (let workIndex = 0; workIndex < workItems.length; workIndex += 1) {
    const item = workItems[workIndex];
    const language = item.source.source_lang;
    const stats = statsBySource.get(item.source.id);
    if ((savedCounts.get(language) ?? 0) >= maximumPerLanguage) continue;
    if (
      urlDateIsOutsideWindow(
        item.url,
        config.collection_window.start_date,
        config.collection_window.end_date,
      )
    ) {
      stats.date_rejected += 1;
      continue;
    }

    const existing = recordsByUrl.get(item.url);
    if (existing && !refresh) {
      stats.cached += 1;
      enqueueAttachments(
        existing.raw_content_html ?? "",
        existing.canonical_url,
        item.source,
        workIndex,
      );
      continue;
    }

    try {
      const response = await fetchText(item.url);
      stats.fetched += 1;
      const canonicalUrl = canonicalizeUrl(response.finalUrl);
      enqueueAttachments(response.text, canonicalUrl, item.source, workIndex);
      if (otherDomainUrls.has(canonicalUrl)) {
        stats.duplicate += 1;
        continue;
      }
      if (recordsByUrl.has(canonicalUrl) && !refresh) {
        stats.duplicate += 1;
        continue;
      }
      const article = extractArticle(response.text, item.source);
      if (
        !isIsoDateInRange(
          article.published_at,
          config.collection_window.start_date,
          config.collection_window.end_date,
        )
      ) {
        stats.date_rejected += 1;
        continue;
      }

      const id = `raw_${sha256(canonicalUrl).slice(0, 16)}`;
      const record = {
        id,
        language_pair: "zh_en",
        domain,
        source_lang: language,
        source_url: item.url,
        canonical_url: canonicalUrl,
        source_id: item.source.id,
        source_site: item.source.site_name,
        title: article.title,
        author: article.author,
        published_at: article.published_at,
        crawled_at: new Date().toISOString(),
        http_status: response.status,
        rights_status: item.source.rights_status,
        rights_note: item.source.rights_note,
        rights_evidence_url: item.source.rights_evidence_url ?? null,
        license_name: item.source.license_name ?? null,
        license_url: item.source.license_url ?? null,
        attribution: item.source.attribution ?? null,
        content_sha256: sha256(article.raw_text),
        raw_content_html: article.raw_content_html,
        raw_text: article.raw_text,
      };

      const prior = recordsById.get(id);
      if (prior) recordsByUrl.delete(prior.canonical_url);
      recordsById.set(id, record);
      recordsByUrl.set(canonicalUrl, record);
      if (!prior) savedCounts.set(language, (savedCounts.get(language) ?? 0) + 1);
      stats.saved += 1;
      newlySavedSinceCheckpoint += 1;
      if (newlySavedSinceCheckpoint >= checkpointEvery) {
        persist();
        newlySavedSinceCheckpoint = 0;
        console.log(
          `检查点：${domain} zh-CN=${savedCounts.get("zh-CN") ?? 0}, en=${savedCounts.get("en") ?? 0}`,
        );
      }
    } catch (error) {
      stats.extraction_failed += 1;
      console.warn(`跳过 ${item.url}: ${error.message}`);
    }
  }

  persist();
  console.table([...statsBySource.values()]);
  console.table(
    [...savedCounts].map(([sourceLanguage, count]) => ({
      domain,
      source_language: sourceLanguage,
      raw_pages: count,
    })),
  );
}

main().catch((error) => {
  console.error(`采集失败：${error.message}`);
  process.exitCode = 1;
});
