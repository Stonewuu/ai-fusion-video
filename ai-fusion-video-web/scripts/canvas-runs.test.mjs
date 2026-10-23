import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/canvas/run-state.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`)(loaded, loaded.exports);
const { parseRunRequest, runRequestKey, runFinished } = loaded.exports;
const scope = { userId: 7, teamId: null };
const canvasId = "00000000-0000-4000-8000-000000000001";
const request = { version: 1, canvasId, title: "已确认的运行范围", createdAt: "2026-10-13T05:00:00Z",
  cost: { nodes: 4, images: 2, videoSeconds: 10, batchItems: 1, amount: "未配置" },
  request: { requestId: "original-start", planId: "00000000-0000-4000-8000-000000000002", planHash: "a".repeat(64), scope } };

test("recovery preserves the original immutable plan request and all quantity limits", () => {
  assert.deepEqual(parseRunRequest(JSON.stringify(request), canvasId, scope), request);
  assert.equal(parseRunRequest(JSON.stringify(request), canvasId, scope).request.requestId, "original-start");
});
test("recovery rejects another canvas, account, team and a damaged plan receipt", () => {
  assert.throws(() => parseRunRequest(JSON.stringify(request), "another-canvas", scope));
  assert.throws(() => parseRunRequest(JSON.stringify(request), canvasId, { userId: 8, teamId: null }));
  assert.throws(() => parseRunRequest(JSON.stringify(request), canvasId, { userId: 7, teamId: 2 }));
  assert.throws(() => parseRunRequest(JSON.stringify({ ...request, request: { ...request.request, planHash: "damaged" } }), canvasId, scope));
  assert.throws(() => parseRunRequest(JSON.stringify({ ...request, cost: { ...request.cost, images: -1 } }), canvasId, scope));
  assert.throws(() => parseRunRequest("{broken", canvasId, scope));
  assert.notEqual(runRequestKey(canvasId, scope), runRequestKey(canvasId, { userId: 7, teamId: 2 }));
});
test("selection, uncertainty and a cancellation request keep the run observable", () => {
  for (const state of ["WAITING_SELECTION", "WAITING_CONFIRMATION", "CANCEL_REQUESTED", "UNKNOWN_SUBMISSION", "NEEDS_REVIEW", "RUNNING"])
    assert.equal(runFinished(state), false);
  for (const state of ["SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED"])
    assert.equal(runFinished(state), true);
});
