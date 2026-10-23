import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const require = createRequire(import.meta.url);
const source = fs.readFileSync(path.resolve(import.meta.dirname, "../lib/canvas/node-run-store.ts"), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const loadedModule = { exports: {} };
vm.runInNewContext(compiled, { module: loadedModule, exports: loadedModule.exports, require, Map });
const { createCanvasNodeRunStore } = loadedModule.exports;
const result = (id, changes = {}) => ({
  id: `run-${id}`, nodeId: id, parentNodeRunId: null, revision: 1, status: "RUNNING",
  derivedState: "CURRENT", contentUnavailable: false, outputs: { values: {} }, ...changes,
});

test("a 1000-node poll only changes the node whose result advanced", () => {
  const store = createCanvasNodeRunStore();
  const rows = Array.from({ length: 1000 }, (_, i) => result(String(i)));
  store.getState().update(rows);
  const previous = store.getState().nodes;
  const polled = structuredClone(rows);
  polled[500].revision++;
  polled[500].status = "SUCCEEDED";
  polled[500].outputs.values.text = [{ type: "text", text: "真实结果" }];
  store.getState().update(polled);
  const next = store.getState().nodes;
  assert.equal([...next].filter(([id, node]) => node !== previous.get(id)).length, 1);
  assert.equal(next.get("500").outputs.values.text[0].text, "真实结果");
  let notifications = 0;
  store.subscribe(() => notifications++);
  store.getState().update(structuredClone(polled));
  assert.equal(notifications, 0);
  assert.equal(store.getState().nodes, next);
});

test("batch children do not replace the top-level node result", () => {
  const store = createCanvasNodeRunStore();
  const parent = result("a");
  store.getState().update([result("a", { id: "child", parentNodeRunId: "parent" }), parent]);
  assert.equal(store.getState().nodes.get("a"), parent);
});

test("derived changes and redacted results replace cached content without a revision change", () => {
  const store = createCanvasNodeRunStore();
  store.getState().update([result("a")]);
  const stale = result("a", { derivedState: "STALE" });
  store.getState().update([stale]);
  assert.equal(store.getState().nodes.get("a"), stale);
  const redacted = { ...stale, contentUnavailable: true, outputs: { values: {} } };
  store.getState().update([redacted]);
  assert.equal(store.getState().nodes.get("a"), redacted);
});

test("switching runs or losing access removes the previous run result", () => {
  const store = createCanvasNodeRunStore();
  store.getState().update([result("a")]);
  const replaced = result("a", { id: "another-run-a" });
  store.getState().update([replaced]);
  assert.equal(store.getState().nodes.get("a"), replaced);
  store.getState().update([]);
  assert.equal(store.getState().nodes.size, 0);
  assert.equal(store.getState().children.size, 0);
});

test("a completed batch child refreshes previews before its parent finishes without rerendering other nodes", () => {
  const store = createCanvasNodeRunStore();
  const parent = result("batch");
  const child = result("batch", { id: "child", parentNodeRunId: parent.id, itemIndex: 0, attempt: 1 });
  store.getState().update([parent, child]);
  const completed = { ...child, revision: 2, status: "SUCCEEDED", outputs: { values: { images: [{ mediaId: "image", type: "image" }] } } };
  store.getState().update([structuredClone(parent), completed]);
  assert.equal(store.getState().nodes.get("batch"), parent);
  assert.equal(store.getState().children.get(parent.id)[0], completed);
  let changes = 0;
  store.subscribe(() => changes++);
  store.getState().update(structuredClone([parent, completed]));
  assert.equal(changes, 0);
  store.getState().update([parent, completed, { ...child, id: "retry", attempt: 2 }]);
  assert.equal(store.getState().children.get(parent.id).length, 1);
  assert.equal(store.getState().children.get(parent.id)[0].id, "retry");
});
