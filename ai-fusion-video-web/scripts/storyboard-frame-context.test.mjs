import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/storyboard/frame-prompts.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
const loaded = { exports: {} };
vm.runInThisContext("(function(module,exports){" + compiled + "\n})")(loaded, loaded.exports);
const { buildBatchFrameContext, buildDefaultFramePrompt } = loaded.exports;

test("同场首尾帧合并为一次任务，并保持各自选中范围", () => {
  const context = buildBatchFrameContext(10, [4097, 4098, 4097], [4098, 4100]);
  assert.equal(context.frameType, "both");
  assert.deepEqual(context.selectedStoryboardItemIds, [4097, 4098, 4100]);
  assert.deepEqual(context.frameRequests, [
    { frameType: "first", storyboardItemIds: [4097, 4098] },
    { frameType: "last", storyboardItemIds: [4098, 4100] },
  ]);
});

test("仅选尾帧不会隐式增加首帧任务，空选择不发起工作流", () => {
  const context = buildBatchFrameContext(10, [], [4100]);
  assert.equal(context.frameType, "last");
  assert.deepEqual(context.frameRequests, [{ frameType: "last", storyboardItemIds: [4100] }]);
  assert.equal(buildBatchFrameContext(10, [], []), null);
});

test("重新生成时分别沿用已经修订的首帧和尾帧提示词", () => {
  const shot = { id: 42, firstFramePrompt: "帘子仅开窄缝", lastFramePrompt: "帘已掀开，招手结束" };
  assert.equal(buildDefaultFramePrompt(shot, null, "first"), "帘子仅开窄缝");
  assert.equal(buildDefaultFramePrompt(shot, null, "last"), "帘已掀开，招手结束");
});

test("尾帧没有提示词时从本镜头构建，不误用首帧的动作起点", () => {
  const result = buildDefaultFramePrompt({ id: 42, firstFramePrompt: "手刚触帘边", lastFramePrompt: null, content: "掀帘招手" }, null, "last");
  assert.ok(result.includes("镜头结束时"));
  assert.ok(result.includes("掀帘招手"));
  assert.ok(!result.includes("手刚触帘边"));
});
