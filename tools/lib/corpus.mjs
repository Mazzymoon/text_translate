import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const toolsDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

export const PROJECT_ROOT = path.resolve(toolsDirectory, "..");
export const DATASET_ROOT = path.join(PROJECT_ROOT, "dataset");
export const CRAWLED_ZH_EN_ROOT = path.join(DATASET_ROOT, "crawled", "zh_en");
export const ZH_EN_DOMAINS = ["education", "technology", "finance"];

export function requireZhEnDomain(value) {
  const domain = String(value ?? "").trim();
  if (!ZH_EN_DOMAINS.includes(domain)) {
    throw new Error(`--domain 必须是 ${ZH_EN_DOMAINS.join("、")} 之一`);
  }
  return domain;
}

export function zhEnDataFile(stage, domain) {
  if (!["raw", "cleaned"].includes(stage)) {
    throw new Error(`未知的中英数据阶段：${stage}`);
  }
  return path.join(CRAWLED_ZH_EN_ROOT, stage, `${requireZhEnDomain(domain)}.json`);
}

export function zhEnCrawledDirectory(stage) {
  if (!["source", "translations", "review"].includes(stage)) {
    throw new Error(`未知的中英爬取数据目录：${stage}`);
  }
  return path.join(CRAWLED_ZH_EN_ROOT, stage);
}

export function emptyZhEnDocument(domain) {
  return {
    schema_version: 2,
    language_pair: "zh_en",
    domain: requireZhEnDomain(domain),
    generated_at: null,
    records: [],
  };
}

export function readJson(filePath, fallback = null) {
  if (!fs.existsSync(filePath)) {
    if (fallback !== null) return fallback;
    throw new Error(`文件不存在：${filePath}`);
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
}

export function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export function writeJsonAtomic(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  fs.renameSync(temporaryPath, filePath);
}

export function parseArguments(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith("--")) {
      throw new Error(`无法识别的参数：${argument}`);
    }
    const key = argument.slice(2);
    const next = argv[index + 1];
    if (next === undefined || next.startsWith("--")) {
      result[key] = true;
    } else {
      result[key] = next;
      index += 1;
    }
  }
  return result;
}

export function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

export function countHanCharacters(value) {
  return (String(value).match(/\p{Script=Han}/gu) ?? []).length;
}

export function countEnglishWords(value) {
  return (String(value).match(/[A-Za-z]+(?:['’-][A-Za-z]+)*/g) ?? []).length;
}

export function normalizeText(value) {
  return String(value)
    .replace(/\u00A0/g, " ")
    .replace(/[\t\f\v ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function isIsoDateInRange(value, startDate, endDate) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value ?? "")) return false;
  return value >= startDate && value <= endDate;
}

export function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function parseCsv(csvText, filePath = "CSV") {
  const text = String(csvText).replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];

    if (inQuotes) {
      if (character === '"') {
        if (text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += character;
      }
      continue;
    }

    if (character === '"' && field.length === 0) {
      inQuotes = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.endsWith("\r") ? field.slice(0, -1) : field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }

  if (inQuotes) {
    throw new Error(`${filePath}: CSV quote was not closed`);
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field.endsWith("\r") ? field.slice(0, -1) : field);
    rows.push(row);
  }

  return rows;
}

export function serializeCsv(rows, columns, options = {}) {
  const includeBom = Boolean(options.bom);
  const csv = [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvEscape(row[column])).join(",")),
  ].join("\n");
  return `${includeBom ? "\uFEFF" : ""}${csv}\n`;
}

export function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
