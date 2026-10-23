import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const compiled = ts.transpileModule(
  fs.readFileSync(
    new URL("../lib/canvas/import-state.ts", import.meta.url),
    "utf8",
  ),
  {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
    },
  },
).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`)(
  loaded,
  loaded.exports,
);
const { parseImportRequest, importRequestKey, importFinished } = loaded.exports;
const scope = { userId: 71, teamId: null };
const canvasId = "123e4567-e89b-42d3-a456-426614174000";
const source = {
  title: "首帧",
  source: { type: "shot", id: 1, version: 3, field: "firstFrameImageUrl" },
};
const saved = {
  version: 1,
  canvasId,
  createdAt: "2026-10-14T00:00:00Z",
  request: {
    requestId: "fixed-import-request",
    scope,
    sources: [source],
    position: { x: 80, y: 160 },
  },
};
test("request recovery preserves source versions, ordering, destination and original identity", () => {
  assert.deepEqual(
    parseImportRequest(JSON.stringify(saved), canvasId, scope),
    saved,
  );
  const onlyLibrary = {
    ...saved,
    request: { ...saved.request, position: null },
  };
  assert.deepEqual(
    parseImportRequest(JSON.stringify(onlyLibrary), canvasId, scope),
    onlyLibrary,
  );
});
test("another account, team or canvas cannot resume the stored request", () => {
  for (const context of [
    { userId: 72, teamId: null },
    { userId: 71, teamId: 1 },
  ]) {
    assert.throws(() =>
      parseImportRequest(JSON.stringify(saved), canvasId, context),
    );
    assert.notEqual(
      importRequestKey(canvasId, scope),
      importRequestKey(canvasId, context),
    );
  }
  assert.throws(() =>
    parseImportRequest(JSON.stringify(saved), "another-canvas", scope),
  );
});
test("malformed admission records and oversized or incomplete source lists never submit", () => {
  for (const request of [
    { ...saved.request, requestId: "" },
    { ...saved.request, sources: [] },
    { ...saved.request, sources: Array(51).fill(source) },
    {
      ...saved.request,
      sources: [{ ...source, source: { ...source.source, version: null } }],
    },
    {
      ...saved.request,
      sources: [{ ...source, source: { ...source.source, id: -1 } }],
    },
    { ...saved.request, position: { x: 1000001, y: 0 } },
  ])
    assert.throws(() =>
      parseImportRequest(
        JSON.stringify({ ...saved, request }),
        canvasId,
        scope,
      ),
    );
  assert.throws(() => parseImportRequest("{broken", canvasId, scope));
});
test("waiting for another receiver is still active and partial failure is retryable terminal work", () => {
  for (const state of ["QUEUED", "WAITING", "RUNNING"])
    assert.equal(importFinished(state), false);
  for (const state of ["SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED"])
    assert.equal(importFinished(state), true);
});
