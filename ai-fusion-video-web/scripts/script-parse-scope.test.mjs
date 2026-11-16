import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(
  new URL("../lib/scripts/parse-scope.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2020,
  },
});
const { selectParseEpisodes, parseEpisodesMessage } = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

test("一键选择全部未完成分集，几百集不截断并跳过已完成和无原文项", () => {
  const episodes = Array.from({ length: 500 }, (_, index) => ({
    id: index + 1000,
    episodeNumber: index + 1,
    rawContentLength: 3000,
    parsingStatus: index < 20 ? 2 : 0,
  }));
  episodes.push({
    id: 2000,
    episodeNumber: 501,
    rawContentLength: 0,
    parsingStatus: 0,
  });
  const selected = selectParseEpisodes(episodes, "pending", 1, 1, false);
  assert.equal(selected.length, 480);
  assert.equal(selected[0].id, 1020);
  assert.equal(selected.at(-1).id, 1499);
  const message = parseEpisodesMessage(selected, false, "pending");
  assert.ok(message.includes("全部有原文且尚未解析完成"));
  assert.ok(!message.includes("1020"));
});

test("集数范围使用集号筛选，发给智能体的对象仍使用数据库 ID", () => {
  const episodes = [
    { id: 900, episodeNumber: 3, rawContentLength: 200, parsingStatus: 0 },
    { id: 4, episodeNumber: 4, rawContentLength: 200, parsingStatus: 2 },
    { id: 42, episodeNumber: 5, rawContentLength: 200, parsingStatus: 3 },
    { id: 5, episodeNumber: 12, rawContentLength: 200, parsingStatus: 0 },
  ];
  const selected = selectParseEpisodes(episodes, "range", 3, 5, false);
  assert.deepEqual(
    selected.map((episode) => episode.id),
    [900, 42],
  );
  assert.ok(
    parseEpisodesMessage(selected, false, "range").includes("[900, 42]"),
  );
  assert.deepEqual(
    selectParseEpisodes(episodes, "range", 3, 5, true).map(
      (episode) => episode.id,
    ),
    [900, 4, 42],
  );
});

test("重新解析全部包含已完成分集并明确只替换首批，非法范围阻止提交", () => {
  const episodes = [
    { id: 8, episodeNumber: 1, rawContentLength: 100, parsingStatus: 2 },
  ];
  assert.equal(selectParseEpisodes(episodes, "all", 1, 1, true).length, 1);
  assert.ok(
    parseEpisodesMessage(episodes, true, "all").includes("首批保存时替换"),
  );
  for (const [from, to] of [
    [0, 1],
    [2, 1],
    [1.5, 3],
    [1, NaN],
  ]) {
    assert.throws(
      () => selectParseEpisodes(episodes, "range", from, to, false),
      /开始集数|结束集数/,
    );
  }
});

test("范围错误指出具体字段，反向范围同时标记两端", () => {
  for (const [from, to, field, message] of [
    [0, 1, "from", "开始集数须为大于 0 的整数"],
    [1, 0, "to", "结束集数须为大于 0 的整数"],
    [700, 600, "range", "开始集数不能大于结束集数"],
  ]) {
    assert.throws(
      () => selectParseEpisodes([], "range", from, to, false),
      (error) => error.field === field && error.message === message,
    );
  }
});
