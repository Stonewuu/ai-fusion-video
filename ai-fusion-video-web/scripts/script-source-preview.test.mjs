import test from "node:test";
import assert from "node:assert/strict";
import { loadBrowserWorkerModule } from "./helpers/browser-worker-runtime.mjs";

const { previewScriptSource } = loadBrowserWorkerModule("lib/scripts/preview-script-source.ts");

function preview(source, rule = "AUTO", targetLength = 1000) {
  return previewScriptSource({ source, rule, targetLength }, new AbortController().signal);
}

function assertParts(source, result, parts) {
  let start = 0;
  const segments = parts.map(([title, text]) => {
    const end = start + text.length;
    const segment = { title, start, end };
    start = end;
    return segment;
  });
  assert.equal(parts.map(([, text]) => text).join(""), source);
  assert.equal(result.totalChars, source.length);
  assert.deepEqual(result.segments, segments);
}

test("自动识别保留前言与卷名，区分无正文的目录和重复出现的章节", async () => {
  const intro = "目录\r\n第一章 归来......1\r\n第一卷 春\r\n第一章 归来\r\n";
  const first = "第一卷 春\r\n第一章 归来\r\n她回来了🌕。\r\n";
  const repeat = "第一章 归来\r\n";
  const last = "第二卷 秋\r\n## 【第２章】\r\n继续前行。";
  const source = intro + first + repeat + last;
  const result = await preview(source);
  assertParts(source, result, [
    ["前言", intro], ["第一卷 春 · 第一章 归来", first],
    ["第一卷 春 · 第一章 归来", repeat], ["第二卷 秋 · 【第2章】", last],
  ]);
  assert.equal(result.headingCount, 3);
});

test("自动识别优先分集标题，指定章、集或字数会使用对应边界", async () => {
  const pre = "序章\n往事。\n";
  const ch = "Chapter IV: Dawn\n故事。\n";
  const ep = "第 ３ 話 乌云\n开场。\n";
  const en = "EP02: Rain\n继续。";
  const source = pre + ch + ep + en;
  const byEpisode = [["序章", pre + ch], ["第 3 話 乌云", ep], ["EP02: Rain", en]];
  assertParts(source, await preview(source), byEpisode);
  assertParts(source, await preview(source, "EPISODE"), byEpisode);
  assertParts(source, await preview(source, "CHAPTER"), [["序章", pre], ["Chapter IV: Dawn", ch + ep + en]]);
  assertParts(source, await preview(source, "LENGTH"), [["第 1 集", source]]);
  await assert.rejects(preview("只有正文。", "CHAPTER"), /未识别到所选标题/);
});

test("长段落先按换行、再按最后一个句末拆分，保留 CRLF 和补充平面字符", async () => {
  const cases = [
    ["文".repeat(600) + "\r\n", "句。".repeat(399)],
    ["文".repeat(999), "\r\n" + "完".repeat(30)],
    ["文".repeat(999), "🌕" + "完".repeat(30)],
    ["文".repeat(800) + "。" + "续".repeat(198) + "！", "余".repeat(100)],
  ];
  for (const [first, last] of cases) {
    const source = first + last;
    const result = await preview(source, "LENGTH");
    assertParts(source, result, [["第 1 集（1）", first], ["第 1 集（2）", last]]);
    assert.equal(result.splitCount, 1);
  }
});

test("标题里的排版空格可忽略，粘贴内容中的段落分隔符不会拼出章节标题", async () => {
  const intro = "第\u2028一章\n留在前言。\n";
  const chapter = "第\u3000兩百\u3000回 归途\n正文。";
  const source = intro + chapter;
  assertParts(source, await preview(source), [["前言", intro], ["第 兩百 回 归途", chapter]]);
});

test("分集超过保存上限时给出调整方法，取消识别时结束后台计算", async () => {
  await assert.rejects(preview("第一章\n文。\n".repeat(20_001)), /20000 集/);
  const controller = new AbortController();
  const pending = previewScriptSource({ source: "第一章\n文。", rule: "AUTO", targetLength: 1000 }, controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
});
