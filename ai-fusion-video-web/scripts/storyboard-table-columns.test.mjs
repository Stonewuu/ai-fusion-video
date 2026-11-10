import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL(
  "../app/(dashboard)/projects/[id]/storyboards/_components/storyboard-table-columns.ts", import.meta.url,
), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { resolveStoryboardColWidths, loadSavedColWidths, saveColWidths } = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

test("旧记录升级未调整的列，保留加宽列，并抬高过窄的列", () => {
  const legacy = [48, 80, 92, 80, 420, 160, 64, 50, 90, 80, 240, 200, 100, 100];
  const result = resolveStoryboardColWidths(legacy);
  assert.equal(result[4], 420);
  assert.equal(result[6], 88);
  assert.equal(result[7], 64);
  assert.equal(result[9], 160);
  assert.equal(result[10], 320);
  assert.equal(legacy[7], 50);
});

test("新记录再次读取时不把用户设置误认为旧默认值", (t) => {
  let raw = null;
  const previous = globalThis.localStorage;
  globalThis.localStorage = {
    getItem() {
      return raw;
    },
    setItem(_key, value) {
      raw = value;
    },
  };
  t.after(() => {
    if (previous === undefined) {
      delete globalThis.localStorage;
    } else {
      globalThis.localStorage = previous;
    }
  });
  const widths = resolveStoryboardColWidths(null);
  widths[2] = 92;
  widths[10] = 240;
  saveColWidths(widths);
  assert.deepEqual(loadSavedColWidths(), widths);
});

test("损坏的本地列宽不会产生无效网格或压缩已保存的有效列", () => {
  const widths = resolveStoryboardColWidths(null);
  widths[4] = 420;
  widths[6] = "64";
  widths[7] = -1;
  widths[9] = null;
  widths[10] = Number.POSITIVE_INFINITY;
  const result = resolveStoryboardColWidths({ version: 2, widths });
  assert.equal(result[4], 420);
  assert.equal(result[6], 88);
  assert.equal(result[7], 72);
  assert.equal(result[9], 160);
  assert.equal(result[10], 320);
});
