import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

function loadTs(relativePath) {
  const source = fs.readFileSync(
    new URL(relativePath, import.meta.url),
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
  return loaded.exports;
}
const {
  comfyUiTestInput,
  comfyUiTestInputs,
  comfyUiTestKey,
  readPendingComfyUiTest,
  comfyUiTestStatus,
} = loadTs("../lib/comfyui-test-request.ts");
const {
  parseInputBindingsJson,
  flattenInputBindings,
  groupInputBindings,
  updateComfyUiInputBinding,
} = loadTs(
  "../app/(dashboard)/settings/ai-models/_components/comfyui-workflow-support.ts",
);
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

test("optional media bindings survive editing and serialization without changing older bindings", () => {
  const bindings = {
    referenceImages: [
      {
        nodeId: "1",
        inputName: "image",
        valueType: "uploaded_image",
        index: 0,
      },
      {
        nodeId: "2",
        inputName: "image",
        valueType: "uploaded_image",
        index: 1,
        optional: true,
      },
    ],
    referenceAudios: [
      {
        nodeId: "3",
        inputName: "audio",
        valueType: "uploaded_audio",
        index: 0,
        optional: true,
      },
    ],
    referenceVideos: [
      {
        nodeId: "4",
        inputName: "video",
        valueType: "uploaded_video",
        index: 0,
        optional: true,
      },
    ],
  };
  const saved = JSON.stringify(
    groupInputBindings(
      flattenInputBindings(parseInputBindingsJson(JSON.stringify(bindings))),
    ),
  );
  assert.deepEqual(JSON.parse(saved), bindings);
  assert.equal(
    Object.hasOwn(JSON.parse(saved).referenceImages[0], "optional"),
    false,
  );
});

test("switching a media binding to a scalar removes its optional behavior", () => {
  const row = {
    businessField: "referenceImages",
    nodeId: "1",
    inputName: "image",
    valueType: "uploaded_image",
    optional: true,
  };
  const nodes = [{ id: "2", classType: "Text", inputNames: ["text"] }];
  const switched = updateComfyUiInputBinding(
    row,
    { businessField: "prompt", valueType: "string", nodeId: "2" },
    nodes,
  );
  assert.equal(switched.inputName, "text");
  assert.equal(Object.hasOwn(switched, "optional"), false);
  assert.equal(row.optional, true);
  const disabled = updateComfyUiInputBinding(row, { optional: false }, nodes);
  assert.equal(disabled.optional, false);
});

test("trial requests omit absent media and preserve supplied audio, video and false values", () => {
  assert.deepEqual(
    comfyUiTestInputs(
      {
        referenceImages: " ",
        referenceAudios: "media://audio-one\nhttps://example.com/a,b.wav",
        referenceVideos: "media://video-one",
        generateAudio: "false",
        seed: "0",
      },
      [
        { name: "referenceImages", type: "uploaded_image" },
        { name: "referenceAudios", type: "uploaded_audio" },
        { name: "referenceVideos", type: "uploaded_video" },
        { name: "generateAudio", type: "boolean" },
        { name: "seed", type: "integer" },
      ],
    ),
    {
      referenceAudios: ["media://audio-one", "https://example.com/a,b.wav"],
      referenceVideos: ["media://video-one"],
      generateAudio: false,
      seed: 0,
    },
  );
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
    "自动处理中",
  );
  assert.equal(
    comfyUiTestStatus({ status: "CANCELLED", cancelRequested: true }),
    "已取消",
  );
});
