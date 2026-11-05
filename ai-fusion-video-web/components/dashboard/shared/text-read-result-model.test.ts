import assert from "node:assert/strict";
import test from "node:test";
import { readTextPage, savedTextResult, scriptContentSummaries, textReadSummary } from "./text-read-result-model";
import { formatToolArgumentScalar, listToolArgumentEntries } from "./tool-call-argument-model";

const middle = {
  status: "success", content: "本段原文", offset: 8000, endOffset: 16000, totalChars: 176165,
  hasMore: true, nextOffset: 16000, resourceType: "script", field: "rawContent",
};

test("分段读取使用正文位置展示范围，不把这一段当作累计处理进度", () => {
  const page = readTextPage(middle, "read_script_content");
  assert.ok(page);
  assert.deepEqual(textReadSummary(page), {
    range: "第 8,001–16,000 字", total: "共 176,165 字", detail: "本次读取 8,000 字 · 后面还有内容",
  });
});

test("只读取最后一段时只报告到达末尾，不宣称全文或剧本解析已经完成", () => {
  const page = readTextPage({ ...middle, offset: 176000, endOffset: 176165, nextOffset: null, hasMore: false }, "read_script_content");
  assert.ok(page);
  const summary = textReadSummary(page);
  assert.equal(summary.range, "第 176,001–176,165 字");
  assert.equal(summary.detail, "本次读取 165 字 · 已到内容末尾");
  assert.doesNotMatch(summary.detail, /全文|解析完成/);
});

test("空正文和在末尾继续读取不产生倒置范围", () => {
  const empty = readTextPage({ ...middle, content: "", offset: 0, endOffset: 0, totalChars: 0, nextOffset: null, hasMore: false }, "read_script_content");
  assert.ok(empty);
  assert.equal(textReadSummary(empty).detail, "原文为空");
  assert.doesNotMatch(textReadSummary(empty).range, /第 1/);
  const end = readTextPage({ ...middle, content: "", offset: 176165, endOffset: 176165, nextOffset: null, hasMore: false }, "read_tool_result");
  assert.ok(end);
  assert.equal(textReadSummary(end).detail, "已到内容末尾");
});

test("矛盾或不完整的分页结果不能被展示为读取成功", () => {
  for (const change of [{ endOffset: 7000 }, { totalChars: 1 }, { nextOffset: 0 }, { hasMore: false },
    { offset: -1 }, { endOffset: 8000, nextOffset: 8000 }, { status: "error" }, { field: "unknown" }]) {
    assert.equal(readTextPage({ ...middle, ...change }, "read_script_content"), null);
  }
});

test("完整结果的读取入口展示字数，不能被识别为空列表", () => {
  assert.equal(savedTextResult({ readTool: "read_tool_result", resultId: "a".repeat(64), totalChars: 95910 }), 95910);
  assert.equal(savedTextResult({ assets: [], total: 0 }), null);
  assert.equal(savedTextResult({ readTool: "read_tool_result", totalChars: 95910 }), null);
});

test("剧本读取入口提供原文字数，缺少入口时不虚构零字", () => {
  assert.deepEqual(scriptContentSummaries({ contentReads: { rawContent: { readTool: "read_script_content", totalChars: 176165 } } }),
    [{ label: "原文", totalChars: 176165 }]);
  assert.deepEqual(scriptContentSummaries({ title: "剧本" }), []);
});

test("读取参数显示中文对象和从一开始的文字位置", () => {
  const entries = listToolArgumentEntries({ resourceType: "episode", resourceId: 42, field: "rawContent", offset: 8000, limit: 4000 }, "read_script_content");
  assert.deepEqual(entries.map((entry) => entry.label), ["资源类型", "剧本分集 ID", "读取内容", "开始位置", "本次读取上限"]);
  assert.equal(formatToolArgumentScalar("resourceType", "episode", "read_script_content"), "剧本分集");
  assert.equal(formatToolArgumentScalar("field", "rawContent", "read_script_content"), "原文");
  assert.equal(formatToolArgumentScalar("offset", 8000, "read_script_content"), "第 8,001 字起");
  assert.equal(formatToolArgumentScalar("limit", 4000, "read_tool_result"), "4,000 字");
});
