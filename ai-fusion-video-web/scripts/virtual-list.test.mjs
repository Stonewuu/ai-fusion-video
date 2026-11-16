import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(
  new URL("../lib/virtual-list.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2020,
  },
});
const { retainVirtualItems } = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

test("滚动后保留离屏的编辑和拖动项，并维持键盘遍历顺序", () => {
  assert.deepEqual(
    retainVirtualItems([50, 51, 52], [2, 98], 100),
    [2, 50, 51, 52, 98],
  );
});
test("焦点和拖动命中同一可见项时不会重复挂载", () => {
  const visible = [10, 11, 12];
  assert.deepEqual(retainVirtualItems(visible, [11, 11], 20), [10, 11, 12]);
  assert.deepEqual(visible, [10, 11, 12]);
});
test("场次完全离开视口时仅保留正在操作的镜头，删除后不保留失效索引", () => {
  assert.deepEqual(retainVirtualItems([], [-1, 4, 20], 10), [4]);
  assert.deepEqual(retainVirtualItems([], [0], 0), []);
});
