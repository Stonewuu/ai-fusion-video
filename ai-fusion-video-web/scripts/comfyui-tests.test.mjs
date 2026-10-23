import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(
  new URL("../lib/comfyui-test-request.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.CommonJS,
  },
}).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`)(
  loaded,
  loaded.exports,
);
const {
  comfyUiTestInput,
  comfyUiTestKey,
  readPendingComfyUiTest,
  comfyUiTestStatus,
} = loaded.exports;
const scope = { userId: 7, teamId: null };
const original = {
  schemaVersion: 1,
  createdAt: "2026-10-13T04:00:00Z",
  request: {
    requestId: "fixed-request",
    versionId: 42,
    workflowHash: "a".repeat(64),
    scope,
    inputs: { prompt: "原始输入", referenceImageUrls: ["media://original"] },
  },
};

test("trial recovery retains frozen request and rejects another user, team or workflow version", () => {
  assert.deepEqual(
    readPendingComfyUiTest(JSON.stringify(original), scope, 42),
    original,
  );
  for (const changedScope of [
    { ...scope, userId: 9 },
    { ...scope, teamId: 2 },
  ]) {
    assert.throws(() =>
      readPendingComfyUiTest(JSON.stringify(original), changedScope, 42),
    );
    assert.notEqual(
      comfyUiTestKey(scope, 42),
      comfyUiTestKey(changedScope, 42),
    );
  }
  assert.throws(() =>
    readPendingComfyUiTest(JSON.stringify(original), scope, 43),
  );
});

test("invalid local records never silently become a new trial", () => {
  assert.throws(() => readPendingComfyUiTest("{", scope, 42));
  for (const patch of [
    { schemaVersion: 2 },
    { createdAt: "invalid" },
    { request: { ...original.request, requestId: "" } },
    { request: { ...original.request, workflowHash: "changed" } },
    { request: { ...original.request, inputs: [] } },
  ]) {
    assert.throws(() =>
      readPendingComfyUiTest(
        JSON.stringify({ ...original, ...patch }),
        scope,
        42,
      ),
    );
  }
});

test("media input lines preserve Data URI and URL commas; numbers and booleans are strict", () => {
  const data = "data:image/png;base64,aGVsbG8=";
  assert.deepEqual(
    comfyUiTestInput(`${data}\nhttps://example.com/a,b.png`, "uploaded_image"),
    [data, "https://example.com/a,b.png"],
  );
  assert.equal(comfyUiTestInput("false", "boolean"), false);
  assert.throws(() => comfyUiTestInput("yes", "boolean"));
  assert.throws(() => comfyUiTestInput("1.5", "integer"));
  assert.throws(() => comfyUiTestInput("Infinity", "number"));
});

test("cancel request and unresolved remote status remain distinct from confirmed cancellation", () => {
  assert.equal(
    comfyUiTestStatus({
      status: "WAITING",
      phase: "POLLING",
      cancelRequested: true,
    }),
    "等待远端取消",
  );
  assert.equal(
    comfyUiTestStatus({
      status: "NEEDS_REVIEW",
      phase: "POLLING",
      cancelRequested: true,
    }),
    "需要核实",
  );
  assert.equal(
    comfyUiTestStatus({ status: "CANCELLED", cancelRequested: true }),
    "已取消",
  );
});
