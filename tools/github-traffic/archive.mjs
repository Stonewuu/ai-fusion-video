import { readFile, rename, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";
import path from "node:path";

const metrics = ["clones", "views"];
const repositoryName = /^[\w.-]+\/[\w.-]+$/;

function isDay(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }
  const time = Date.parse(value + "T00:00:00Z");
  return Number.isFinite(time) && new Date(time).toISOString().slice(0, 10) === value;
}

function counts(value) {
  if (!Number.isSafeInteger(value?.count) || !Number.isSafeInteger(value?.uniques) ||
      value.uniques < 0 || value.count < value.uniques) {
    throw new Error("流量记录中的次数或去重人数无效。");
  }
  return { count: value.count, uniques: value.uniques };
}

function readResponse(metric, response) {
  const totals = counts(response);
  if (!Array.isArray(response[metric])) {
    throw new Error("流量接口缺少按日统计。");
  }
  const byDate = new Map();
  for (const item of response[metric]) {
    const date = item.timestamp?.slice(0, 10);
    if (!isDay(date) || item.timestamp !== date + "T00:00:00Z" || byDate.has(date)) {
      throw new Error("流量接口返回了无效或重复的 UTC 日期。");
    }
    byDate.set(date, counts(item));
  }
  const dates = [...byDate.keys()].sort();
  return { byDate, totals: { ...totals, first: dates.at(0) ?? null, last: dates.at(-1) ?? null } };
}

export function newArchive(repository) {
  if (!repositoryName.test(repository)) {
    throw new Error("仓库名称须包含所有者和仓库名。");
  }
  return { version: 1, repository, days: [], snapshots: [] };
}

function assertDates(rows, key) {
  let previous = "";
  for (const row of rows) {
    const date = row[key];
    if (!isDay(date) || date <= previous) {
      throw new Error("归档中的日期无效、重复或顺序有误。");
    }
    previous = date;
  }
}

export function validateArchive(archive, repository) {
  if (archive?.version !== 1 || archive.repository !== repository ||
      !repositoryName.test(repository) || !Array.isArray(archive.days) || !Array.isArray(archive.snapshots)) {
    throw new Error("历史文件的仓库或格式不匹配。");
  }
  assertDates(archive.days, "date");
  assertDates(archive.snapshots, "observedOn");
  for (const row of archive.days) {
    for (const metric of metrics) {
      if (row[metric] !== null) {
        counts(row[metric]);
      }
    }
  }
  for (const snapshot of archive.snapshots) {
    for (const metric of metrics) {
      const summary = snapshot[metric];
      counts(summary);
      if (!(summary.first === null && summary.last === null) &&
          !(isDay(summary.first) && isDay(summary.last) && summary.first <= summary.last)) {
        throw new Error("归档中的统计起止日期无效。");
      }
    }
  }
}

export function mergeTraffic(archive, repository, observedOn, responses) {
  validateArchive(archive, repository);
  const previous = archive.snapshots.at(-1);
  if (!isDay(observedOn) || (previous && observedOn < previous.observedOn)) {
    throw new Error("本次采集日期早于最近的归档，或日期格式无效。");
  }
  const days = new Map(archive.days.map(row => [row.date, { ...row }]));
  const snapshot = { observedOn };
  for (const metric of metrics) {
    const { byDate, totals } = readResponse(metric, responses[metric]);
    snapshot[metric] = totals;
    for (const [date, value] of byDate) {
      const row = days.get(date) ?? { date, clones: null, views: null };
      row[metric] = value;
      days.set(date, row);
    }
  }
  // GitHub 单独返回整个窗口的去重人数，不能用每天的人数重新计算。
  const snapshots = previous?.observedOn === observedOn ? archive.snapshots.slice(0, -1) : [...archive.snapshots];
  snapshots.push(snapshot);
  return { ...archive, days: [...days.values()].sort((a, b) => a.date.localeCompare(b.date)), snapshots };
}

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function run() {
  const [mode, repository, input, output, observedOn = new Date().toISOString().slice(0, 10)] = process.argv.slice(2);
  if (!["init", "update"].includes(mode) || !input || !output) {
    throw new Error("用法：node archive.mjs init|update 所有者/仓库 接口数据目录 历史文件 [UTC日期]");
  }
  const archive = mode === "init" ? newArchive(repository) : await readJson(output);
  const [clones, views] = await Promise.all(metrics.map(metric => readJson(path.join(input, metric + ".json"))));
  const next = mergeTraffic(archive, repository, observedOn, { clones, views });
  const json = JSON.stringify(next, null, 2) + "\n";
  if (mode === "init") {
    // 首次初始化不能覆盖已有历史。
    await writeFile(output, json, { flag: "wx" });
  } else {
    const temporary = output + "." + randomUUID() + ".tmp";
    await writeFile(temporary, json);
    await rename(temporary, output);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await run();
}
