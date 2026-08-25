#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const ZH_EN_COLUMNS = [
  "source_lang",
  "target_lang",
  "source_text",
  "target_text",
  "zh_char_count",
  "domain",
  "translation_method",
];
const ZH_TH_COLUMNS = [
  "source_lang",
  "target_lang",
  "source_text",
  "target_text",
  "zh_char_count",
  "translation_method",
];

const ALLOWED_LANGUAGES = new Set(["zh-CN", "en", "th"]);
const ALLOWED_DOMAINS = new Set(["education", "technology", "finance"]);
const ALLOWED_METHODS = new Set(["human", "google_mt", "llm_mt"]);
const FILE_RULES = {
  "zh_en.csv": { otherLanguage: "en", domain: null, columns: ZH_EN_COLUMNS, requiresDomain: true },
  "zh_th.csv": { otherLanguage: "th", domain: null, columns: ZH_TH_COLUMNS, requiresDomain: false },
};

function parseArguments(argv) {
  const positional = [];
  let minZhChars = 100;
  let fixCounts = false;
  let requireComplete = false;

  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--min-zh-chars") {
      const value = Number(argv[index + 1]);
      if (!Number.isInteger(value) || value < 0) {
        throw new Error("--min-zh-chars 必须是非负整数");
      }
      minZhChars = value;
      index += 1;
    } else if (argv[index] === "--fix-counts") {
      fixCounts = true;
    } else if (argv[index] === "--require-complete") {
      requireComplete = true;
    } else {
      positional.push(argv[index]);
    }
  }

  return {
    inputPath: path.resolve(positional[0] ?? "dataset/final"),
    minZhChars,
    fixCounts,
    requireComplete,
  };
}

function parseCsv(csvText, filePath) {
  const text = csvText.replace(/^\uFEFF/, "");
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
    throw new Error(`${filePath}: CSV 引号没有闭合`);
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field.endsWith("\r") ? field.slice(0, -1) : field);
    rows.push(row);
  }

  return rows;
}

function listCsvFiles(inputPath) {
  const stat = fs.statSync(inputPath);
  if (stat.isFile()) {
    return [inputPath];
  }
  if (!stat.isDirectory()) {
    throw new Error(`输入路径不是文件或目录：${inputPath}`);
  }

  const files = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const childPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(childPath);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".csv")) {
        files.push(childPath);
      }
    }
  };
  visit(inputPath);
  return files.sort();
}

function countHanCharacters(text) {
  return (text.match(/\p{Script=Han}/gu) ?? []).length;
}

function serializeCsv(rows) {
  return `${rows
    .map((row) =>
      row
        .map((value) => {
          const text = String(value);
          return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
        })
        .join(","),
    )
    .join("\n")}\n`;
}

function validateHeader(header, filePath, requiredColumns) {
  if (
    header.length !== requiredColumns.length ||
    requiredColumns.some((column, index) => header[index] !== column)
  ) {
    return `${filePath}: 表头必须严格为 ${requiredColumns.join(",")}`;
  }
  return null;
}

function validateRecord(record, context) {
  const errors = [];
  const {
    source_lang: sourceLanguage,
    target_lang: targetLanguage,
    source_text: sourceText,
    target_text: targetText,
    zh_char_count: suppliedCount,
    domain,
    translation_method: translationMethod,
  } = record;
  const location = `${context.filePath}:${context.lineNumber}`;

  if (!ALLOWED_LANGUAGES.has(sourceLanguage)) {
    errors.push(`${location} source_lang 非法：${sourceLanguage}`);
  }
  if (!ALLOWED_LANGUAGES.has(targetLanguage)) {
    errors.push(`${location} target_lang 非法：${targetLanguage}`);
  }
  if (!sourceText.trim()) {
    errors.push(`${location} source_text 不能为空`);
  }
  if (!targetText.trim()) {
    errors.push(`${location} target_text 不能为空`);
  }
  if (context.fileRule?.requiresDomain && !ALLOWED_DOMAINS.has(domain)) {
    errors.push(`${location} domain 非法：${domain}`);
  }
  if (!ALLOWED_METHODS.has(translationMethod)) {
    errors.push(`${location} translation_method 非法：${translationMethod}`);
  }

  const sourceIsChinese = sourceLanguage === "zh-CN";
  const targetIsChinese = targetLanguage === "zh-CN";
  if (sourceIsChinese === targetIsChinese) {
    errors.push(`${location} 必须且只能有一侧语言为 zh-CN`);
  }

  const otherLanguage = sourceIsChinese ? targetLanguage : sourceLanguage;
  if (sourceIsChinese !== targetIsChinese && !["en", "th"].includes(otherLanguage)) {
    errors.push(`${location} 仅允许 zh-CN 与 en 或 th 配对`);
  }

  const chineseText = sourceIsChinese ? sourceText : targetIsChinese ? targetText : "";
  const actualCount = countHanCharacters(chineseText);
  const parsedCount = Number(suppliedCount);
  if (!/^\d+$/.test(suppliedCount) || !Number.isSafeInteger(parsedCount)) {
    errors.push(`${location} zh_char_count 必须是非负整数：${suppliedCount}`);
  } else if (parsedCount !== actualCount) {
    errors.push(`${location} zh_char_count=${parsedCount}，实际应为 ${actualCount}`);
  }
  if (actualCount < context.minZhChars) {
    errors.push(`${location} 中文正文只有 ${actualCount} 个汉字，少于 ${context.minZhChars}`);
  }

  if (context.fileRule) {
    if (otherLanguage !== context.fileRule.otherLanguage) {
      errors.push(`${location} 语言对与文件名不一致`);
    }
    if (context.fileRule.domain && domain !== context.fileRule.domain) {
      errors.push(`${location} domain 应为 ${context.fileRule.domain}`);
    }
  }

  return {
    errors,
    duplicateKey: [sourceLanguage, targetLanguage, sourceText, targetText].join("\u001F"),
    direction: `${sourceLanguage}->${targetLanguage}`,
    domain: context.fileRule?.requiresDomain ? domain : "-",
  };
}

function main() {
  const { inputPath, minZhChars, fixCounts, requireComplete } = parseArguments(
    process.argv.slice(2),
  );
  const files = listCsvFiles(inputPath);
  if (files.length === 0) {
    throw new Error(`没有找到 CSV 文件：${inputPath}`);
  }

  if (fs.statSync(inputPath).isDirectory()) {
    const expectedFiles = Object.keys(FILE_RULES).sort();
    const actualFiles = files.map((filePath) => path.basename(filePath)).sort();
    const missingFiles = expectedFiles.filter((fileName) => !actualFiles.includes(fileName));
    const unexpectedFiles = actualFiles.filter((fileName) => !expectedFiles.includes(fileName));

    if (missingFiles.length > 0 || unexpectedFiles.length > 0) {
      const details = [
        missingFiles.length > 0 ? `缺少：${missingFiles.join(", ")}` : null,
        unexpectedFiles.length > 0 ? `多余：${unexpectedFiles.join(", ")}` : null,
      ].filter(Boolean);
      throw new Error(`数据目录必须且只能包含 zh_en.csv 和 zh_th.csv（${details.join("；")}）`);
    }
  }

  const errors = [];
  const seenRecords = new Map();
  const summary = [];
  const completeCounts = new Map();

  for (const filePath of files) {
    const rows = parseCsv(fs.readFileSync(filePath, "utf8"), filePath);
    if (rows.length === 0) {
      errors.push(`${filePath}: 文件为空，缺少表头`);
      continue;
    }

    const fileRule = FILE_RULES[path.basename(filePath)] ?? null;
    const requiredColumns = fileRule?.columns ?? ZH_EN_COLUMNS;
    const headerError = validateHeader(rows[0], filePath, requiredColumns);
    if (headerError) {
      errors.push(headerError);
      continue;
    }

    const counts = new Map();
    let dataRows = 0;
    let correctedCounts = 0;

    for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
      const values = rows[rowIndex];
      if (values.every((value) => value === "")) {
        continue;
      }
      dataRows += 1;

      if (values.length !== requiredColumns.length) {
        errors.push(`${filePath}:${rowIndex + 1} 应有 ${requiredColumns.length} 列，实际为 ${values.length} 列`);
        continue;
      }

      if (fixCounts) {
        const sourceLanguage = values[0];
        const targetLanguage = values[1];
        const chineseText =
          sourceLanguage === "zh-CN" ? values[2] : targetLanguage === "zh-CN" ? values[3] : null;
        if (chineseText !== null) {
          const expectedCount = String(countHanCharacters(chineseText));
          if (values[4] !== expectedCount) {
            values[4] = expectedCount;
            correctedCounts += 1;
          }
        }
      }

      const record = Object.fromEntries(requiredColumns.map((column, index) => [column, values[index]]));
      const validation = validateRecord(record, {
        filePath,
        lineNumber: rowIndex + 1,
        fileRule,
        minZhChars,
      });
      errors.push(...validation.errors);

      const previousLocation = seenRecords.get(validation.duplicateKey);
      if (previousLocation) {
        errors.push(`${filePath}:${rowIndex + 1} 与 ${previousLocation} 完全重复`);
      } else {
        seenRecords.set(validation.duplicateKey, `${filePath}:${rowIndex + 1}`);
      }

      const summaryKey = `${validation.direction} / ${validation.domain}`;
      counts.set(summaryKey, (counts.get(summaryKey) ?? 0) + 1);
      const completeKey = `${path.basename(filePath)} / ${validation.direction} / ${validation.domain}`;
      completeCounts.set(completeKey, (completeCounts.get(completeKey) ?? 0) + 1);
    }

    if (fixCounts && correctedCounts > 0) {
      fs.writeFileSync(filePath, serializeCsv(rows), "utf8");
    }

    summary.push({
      file: path.relative(process.cwd(), filePath),
      rows: dataRows,
      fixed: correctedCounts,
      distribution: [...counts.entries()].map(([key, count]) => `${key}: ${count}`).join("; ") || "-",
    });
  }

  if (requireComplete) {
    for (const domain of ["education", "technology", "finance"]) {
      for (const direction of ["zh-CN->en", "en->zh-CN"]) {
        const key = `zh_en.csv / ${direction} / ${domain}`;
        const actual = completeCounts.get(key) ?? 0;
        if (actual < 5001) {
          errors.push(`${key} 至少应为 5001 条，实际为 ${actual} 条`);
        }
      }
    }
  }

  console.table(summary);
  if (errors.length > 0) {
    console.error(`\n校验失败，共 ${errors.length} 个问题：`);
    for (const error of errors) {
      console.error(`- ${error}`);
    }
    process.exitCode = 1;
    return;
  }

  const totalRows = summary.reduce((total, item) => total + item.rows, 0);
  console.log(`\n校验通过：${files.length} 个文件，${totalRows} 条数据。`);
}

try {
  main();
} catch (error) {
  console.error(`校验失败：${error.message}`);
  process.exitCode = 1;
}
