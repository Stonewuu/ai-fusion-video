import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../components/auth/auth-dot-renderer.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { createAuthDotRenderer } = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

function fixture(t) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const canvas = { width: 300, height: 150 };
  const paints = [];
  const clears = [];
  const renderer = createAuthDotRenderer(canvas, {
    clearRect(...args) {
      clears.push(args);
    },
    fillRect(...args) {
      paints.push(args);
    },
  });
  renderer.resize(400, 300);
  t.after(() => renderer.dispose());
  return { canvas, paints, clears, renderer };
}

test("动态点阵每次只更新局部，静止期间不安排逐帧绘制", (t) => {
  const { paints, clears, renderer } = fixture(t);
  const initial = paints.length;
  assert.ok(initial > 0);
  t.mock.timers.tick(5000);
  assert.equal(paints.length, initial);
  renderer.setAnimating(true);
  t.mock.timers.tick(99);
  assert.equal(paints.length, initial);
  t.mock.timers.tick(1);
  assert.ok(paints.length > initial);
  assert.ok(paints.length - initial < initial / 10);
  assert.ok(clears.every(([, , width, height]) => width === 3 && height === 3));
});

test("页面隐藏或成功过渡时暂停，恢复可见后继续，卸载后释放画布并停止更新", (t) => {
  const { canvas, paints, renderer } = fixture(t);
  renderer.setAnimating(true);
  t.mock.timers.tick(100);
  renderer.setAnimating(false);
  const paused = paints.length;
  t.mock.timers.tick(10000);
  assert.equal(paints.length, paused);
  renderer.setAnimating(true);
  t.mock.timers.tick(100);
  assert.ok(paints.length > paused);
  renderer.dispose();
  const disposed = paints.length;
  renderer.setAnimating(true);
  t.mock.timers.tick(10000);
  assert.equal(paints.length, disposed);
  assert.equal(canvas.width, 0);
  assert.equal(canvas.height, 0);
});

test("背景尺寸与可视区域一致，重复尺寸通知不重建点阵", (t) => {
  const { canvas, paints, renderer } = fixture(t);
  const initial = paints.length;
  renderer.resize(400, 300);
  assert.equal(paints.length, initial);
  renderer.resize(390, 844);
  assert.equal(canvas.width, 390);
  assert.equal(canvas.height, 844);
  assert.ok(paints.length > initial);
});
