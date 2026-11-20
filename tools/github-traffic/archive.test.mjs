import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeTraffic, newArchive, validateArchive } from "./archive.mjs";

const repo = "owner/sample";
const entry = (date, count, uniques) => ({ timestamp: date + "T00:00:00Z", count, uniques });
const response = (key, count, uniques, rows) => ({ count, uniques, [key]: rows });
const first = {
  clones: response("clones", 12, 4, [entry("2026-01-01", 5, 3), entry("2026-01-02", 7, 3)]),
  views: response("views", 4, 2, [entry("2026-01-01", 4, 2)])
};

test("跨日重复访客以 GitHub 的窗口总数为准，缺少记录的指标保留为空", () => {
  const archive = mergeTraffic(newArchive(repo), repo, "2026-01-03", first);
  assert.equal(archive.snapshots[0].clones.uniques, 4);
  assert.deepEqual(archive.days[1], { date: "2026-01-02", clones: { count: 7, uniques: 3 }, views: null });
});

test("同日重跑合并修订值、保留窗口外历史，晚于归档的采集才可写入", () => {
  const original = mergeTraffic(newArchive(repo), repo, "2026-01-03", first);
  assert.deepEqual(mergeTraffic(original, repo, "2026-01-03", first), original);
  const updated = {
    clones: response("clones", 8, 3, [entry("2026-01-02", 8, 3)]),
    views: response("views", 0, 0, [entry("2026-01-04", 0, 0)])
  };
  const later = mergeTraffic(original, repo, "2026-01-05", updated);
  assert.equal(later.snapshots.length, 2);
  assert.equal(later.days.length, 3);
  assert.deepEqual(later.days[0], original.days[0]);
  assert.equal(later.days[1].clones.count, 8);
  assert.equal(later.days[2].views.count, 0);
  assert.equal(original.days[1].clones.count, 7);
  assert.throws(() => mergeTraffic(later, repo, "2026-01-02", first));
});

test("异常响应或不同仓库的文件不能覆盖已有历史", () => {
  const original = mergeTraffic(newArchive(repo), repo, "2026-01-03", first);
  const invalid = { ...first, views: { count: 3, uniques: 2 } };
  assert.throws(() => mergeTraffic(original, repo, "2026-01-04", invalid));
  assert.throws(() => mergeTraffic(original, "other/repo", "2026-01-04", first));
  assert.throws(() => validateArchive({ ...original, days: [...original.days, original.days[0]] }, repo));
});

test("两个接口都没有按日记录时仍能保留 GitHub 返回的零值快照", () => {
  const empty = { clones: response("clones", 0, 0, []), views: response("views", 0, 0, []) };
  const result = mergeTraffic(newArchive(repo), repo, "2026-01-03", empty);
  assert.deepEqual(result.days, []);
  assert.deepEqual(result.snapshots[0].views, { count: 0, uniques: 0, first: null, last: null });
});
