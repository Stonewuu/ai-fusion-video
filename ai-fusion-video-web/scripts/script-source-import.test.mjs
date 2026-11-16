import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
const source = await readFile(
  new URL("../lib/scripts/source-import.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2020,
  },
});
const {
  deleteSegment,
  duplicateTitles,
  mergeSegment,
  splitSegment,
  titleKey,
  readSourceFile,
} = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

test("分界编辑只改变范围，所有原文保持完整且顺序不变", () => {
  const source = "第一章\n正文🌕。\n第二章\n原文。";
  const segments = [{ start: 0, end: source.length, title: "故事" }];
  const split = splitSegment(segments, 0, source.indexOf("第二章"));
  assert.equal(
    split.map((item) => source.slice(item.start, item.end)).join(""),
    source,
  );
  assert.deepEqual(mergeSegment(split, 0), segments);
  assert.throws(() => splitSegment(segments, 0, 0), /光标/);
  assert.throws(
    () =>
      mergeSegment(
        [
          { start: 0, end: 60000, title: "一" },
          { start: 60000, end: 120000, title: "二" },
        ],
        0,
      ),
    /10 万/,
  );
  assert.equal(
    mergeSegment(
      [
        { start: 0, end: 12000, title: "一" },
        { start: 12000, end: 42000, title: "二" },
      ],
      0,
    )[0].end,
    42000,
  );
});

test("删除前言后只保留章节原文，原预览仍可用于撤销", () => {
  const preface = "----------\r\n正文卷\r\n----------\r\n";
  const first = "第一章\r\n出发🌕。\r\n";
  const last = "第二章\r\n归来。";
  const source = preface + first + last;
  const segments = [
    { start: 0, end: preface.length, title: "前言" },
    {
      start: preface.length,
      end: preface.length + first.length,
      title: "第一章",
    },
    {
      start: preface.length + first.length,
      end: source.length,
      title: "第二章",
    },
  ];
  const original = structuredClone(segments);
  const result = deleteSegment(source, segments, 0);

  assert.equal(result.source, first + last);
  assert.deepEqual(result.segments, [
    { start: 0, end: first.length, title: "第一章" },
    { start: first.length, end: first.length + last.length, title: "第二章" },
  ]);
  assert.deepEqual(segments, original);
  assert.equal(source, preface + first + last);
});

test("删除中间分集后再合并，不会恢复已删除的文字", () => {
  const first = "第一章\n保留开头。\n";
  const unwanted = "广告\n这部分不导入。\n";
  const last = "第二章\n保留结尾。";
  const source = first + unwanted + last;
  const result = deleteSegment(
    source,
    [
      { start: 0, end: first.length, title: "第一章" },
      {
        start: first.length,
        end: first.length + unwanted.length,
        title: "广告",
      },
      {
        start: first.length + unwanted.length,
        end: source.length,
        title: "第二章",
      },
    ],
    1,
  );
  const merged = mergeSegment(result.segments, 0);

  assert.equal(result.source, first + last);
  assert.deepEqual(merged, [
    { start: 0, end: first.length + last.length, title: "第一章" },
  ]);
  assert.equal(
    result.source.slice(merged[0].start, merged[0].end),
    first + last,
  );
});

test("手动分集后可以删除末集和最后一集，字符边界保持完整", () => {
  const kept = "🌕保留的正文。\n";
  const source = kept + "附录\n不需要的文字🌕。";
  const split = splitSegment(
    [{ start: 0, end: source.length, title: "故事" }],
    0,
    kept.length,
  );
  const result = deleteSegment(source, split, 1);

  assert.equal(result.source, kept);
  assert.deepEqual(result.segments, [
    { start: 0, end: kept.length, title: "故事" },
  ]);
  assert.deepEqual(deleteSegment(result.source, result.segments, 0), {
    source: "",
    segments: [],
  });
});

test("超过旧文件和字数上限的小说可以完整读取", async () => {
  const text = "长篇小说正文。".repeat(600_000);
  const file = new File([text], "长篇小说.txt");
  assert.ok(file.size > 10 * 1024 * 1024);
  assert.equal(await readSourceFile(file, "utf-8"), text);
});

test("空文件与编码错误分别提示处理办法", async () => {
  await assert.rejects(
    readSourceFile(new File(["  \n"], "空文件.txt"), "utf-8"),
    /没有可导入的文字/,
  );
  await assert.rejects(
    readSourceFile(new File([new Uint8Array([255])], "编码.txt"), "utf-8"),
    /切换文件编码/,
  );
});

test("重名检测忽略全角、空格和英文大小写，保留原始标题", () => {
  const titles = ["第 １ 章", "第1章", "Chapter I", "chapter i", "第二章"];
  assert.deepEqual(
    [...duplicateTitles(titles)],
    [titleKey("第1章"), titleKey("Chapter I")],
  );
  assert.equal(titles[0], "第 １ 章");
});
