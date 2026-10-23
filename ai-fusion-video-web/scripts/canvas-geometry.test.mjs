import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";
import { getNodesBounds } from "@xyflow/react";

const require = createRequire(import.meta.url);
const { adoptUserNodes } = createRequire(require.resolve("@xyflow/react"))("@xyflow/system");
const source = fs.readFileSync(path.resolve(import.meta.dirname, "../lib/canvas/flow-geometry.ts"), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
const loadedModule = { exports: {} };
vm.runInNewContext(compiled, { module: loadedModule, exports: loadedModule.exports, require, Map, Set, Error });
const { canvasGraphHierarchy, canvasNodeGeometry } = loadedModule.exports;
const node = (id, parentId = null, kind = "text") => ({
  id, kind, resource: null,
  layout: { x: 100, y: 200, width: 300, height: null, parentId, collapsed: false },
});
const capability = { inputs: [{ id: "prompt" }, { id: "images" }], outputs: [{ id: "text" }] };

test("a 1000-node unordered hierarchy preserves every child and puts ancestors first", () => {
  const nodes = [];
  for (let group = 0; group < 20; group++) {
    nodes.push(node(`g${group}`, null, "group"));
    for (let child = 0; child < 49; child++) nodes.push(node(`g${group}:${child}`, `g${group}`));
  }
  const original = [...nodes].reverse();
  const result = canvasGraphHierarchy(original);
  assert.equal(result.byId.size, 1000);
  assert.equal(result.ordered.length, 1000);
  for (let group = 0; group < 20; group++) {
    assert.equal(result.childCounts.get(`g${group}`), 49);
    assert.ok(result.ordered.findIndex(n => n.id === `g${group}`) < result.ordered.findIndex(n => n.id === `g${group}:0`));
  }
  assert.equal(original[0].id, "g19:48");
});

test("invalid group ancestry fails explicitly instead of hanging during projection", () => {
  assert.throws(() => canvasGraphHierarchy([node("a", "b"), node("b", "a")]), /循环/);
  assert.throws(() => canvasGraphHierarchy([node("a", "missing")]), /不存在/);
});

test("unmounted nodes participate in React Flow bounds and expose every edge anchor", () => {
  const first = { ...node("a"), position: { x: 100, y: 200 }, ...canvasNodeGeometry(node("a"), capability, false) };
  const second = { ...node("b"), position: { x: 1000, y: 800 }, ...canvasNodeGeometry(node("b"), capability, false) };
  const bounds = getNodesBounds([first, second]);
  assert.equal(bounds.x, 100);
  assert.equal(bounds.width, 1200);
  assert.ok(bounds.height >= 600 + 3 * 32);
  assert.deepEqual(Array.from(first.handles, h => h.id), ["annotation-out", "annotation-in", "prompt", "images", "text"]);
  assert.ok(first.handles.every(h => Number.isFinite(h.x) && Number.isFinite(h.y)));
  assert.equal(first.handles.find(h => h.id === "images").type, "target");
  assert.equal(first.handles.find(h => h.id === "text").type, "source");
});

test("measured content dimensions replace initial bounds, while group collapse is explicit", () => {
  assert.equal(canvasNodeGeometry(node("a"), capability, false, { width: 320, height: 173 }).initialHeight, 173);
  const group = node("g", null, "group");
  group.layout.height = 900;
  assert.equal(canvasNodeGeometry(group, { inputs: [], outputs: [] }, false).initialHeight, 900);
  assert.equal(canvasNodeGeometry(group, capability, true).initialHeight, 64);
  assert.equal(canvasNodeGeometry(group, capability, true).handles.length, 2);
  assert.equal(canvasNodeGeometry(node("a"), capability, true).handles.length, 5);
});

test("unrelated graph updates retain measured DOM port anchors instead of replacing them with estimates", () => {
  const documentNode = node("measured");
  const projected = { ...documentNode, position: { x: 100, y: 200 }, ...canvasNodeGeometry(documentNode, capability, false) };
  const lookup = new Map();
  const parents = new Map();
  adoptUserNodes([projected], lookup, parents);
  const measured = { width: 300, height: 241 };
  const actual = { source: [{ id: "text", x: 295, y: 208, width: 10, height: 10, position: "right" }], target: [] };
  lookup.get("measured").internals.handleBounds = actual;
  const updated = { ...projected, measured, ...canvasNodeGeometry(documentNode, capability, false, measured) };
  adoptUserNodes([updated], lookup, parents);
  assert.equal(lookup.get("measured").internals.handleBounds, actual);
  assert.equal(lookup.get("measured").internals.handleBounds.source[0].y, 208);
});
