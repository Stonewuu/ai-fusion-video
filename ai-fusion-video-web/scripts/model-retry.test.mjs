import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const nodeRequire = createRequire(import.meta.url);
const modules = new Map();
function load(relative) {
  if (modules.has(relative)) {
    return modules.get(relative).exports;
  }
  const compiled = ts.transpileModule(fs.readFileSync(path.join(root, relative), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  modules.set(relative, loaded);
  const require = (name) => {
    if (name.startsWith("node:")) {
      return nodeRequire(name);
    }
    return load(`${name.startsWith("@/") ? name.slice(2) : path.posix.join(path.posix.dirname(relative), name)}.ts`);
  };
  vm.runInThisContext(`(function(require,module,exports){${compiled}\n})`, { filename: relative })(require, loaded, loaded.exports);
  return loaded.exports;
}

const { createInitialPipelineState, reducePipelineEvent } = load("components/dashboard/agent-pipeline/state.ts");
const { createPipelineEventHandler } = load("lib/store/pipeline-event-handler.ts");
const { formatSubAgentResult } = load("lib/ai/sub-agent-result.ts");
const event = (sequence, outputType, values = {}) => ({ schemaVersion: 1, runId: "same-run", sequence, outputType, ...values });

// 两处时间线必须一致处理重试，不能把失败尝试的半段内容保留下来。
for (const surface of ["对话", "运行通知"]) {
  for (const child of [false, true]) {
    test(`${surface}在${child ? "子任务" : "主任务"}重试时保留成功操作并继续原运行`, async () => {
      let state = createInitialPipelineState();
      let store = { tasks: [{ id: "task", status: "running", state: { ...state, status: "running" } }] };
      let settlements = 0;
      const receive = surface === "对话"
        ? (value) => { state = reducePipelineEvent(state, value); }
        : createPipelineEventHandler({ notifySettlement: () => { settlements++; }, scheduleStatusSync: () => {} },
          "task", (update) => { store = { ...store, ...update(store) }; });
      let sequence = 0;
      const parent = child ? { parentToolCallId: "parent" } : {};
      if (child) {
        receive(event(++sequence, "TOOL_CALL", { replyId: "parent-reply", toolCalls: [{ id: "parent", name: "storyboard_asset_preprocessor", arguments: "{}" }] }));
      }
      receive(event(++sequence, "TOOL_CALL", { ...parent, replyId: "successful-reply", toolCalls: [{ id: "saved", name: "save_asset", arguments: "{}" }] }));
      receive(event(++sequence, "TOOL_FINISHED", { ...parent, replyId: "successful-reply", toolCallId: "saved", toolName: "save_asset", toolResult: "已保存", toolStatus: "success" }));
      receive(event(++sequence, "CONTENT", { ...parent, replyId: "failed-reply", content: "未完成的回答" }));
      receive(event(++sequence, "REASONING", { ...parent, replyId: "failed-reply", reasoningContent: "未完成的思考" }));
      receive(event(++sequence, "TOOL_CALL_STARTED", { ...parent, replyId: "failed-reply", toolCalls: [{ id: "unfinished", name: "save_asset", arguments: "" }] }));
      receive(event(++sequence, "MODEL_RETRY", { ...parent, replyId: "failed-reply", content: "模型请求暂时中断，2 秒后重试（1/3）" }));
      await new Promise((resolve) => setTimeout(resolve, 25));
      const during = surface === "对话" ? state : store.tasks[0].state;
      assert.notEqual(during.status, "error");
      assert.equal(settlements, 0);
      const items = child ? during.timeline[0].children : during.timeline;
      assert.equal(items.filter((item) => item.type === "tool").length, 1);
      assert.equal(items[0].id, "saved");
      assert.equal(items[0].status, "done");
      assert.equal(items[1].type, "retry");
      assert.ok(!JSON.stringify(items).includes("未完成"));
      if (child) {
        assert.equal(during.timeline[0].id, "parent");
        assert.equal(during.timeline[0].status, "calling");
      }
      receive(event(++sequence, "CONTENT", { ...parent, replyId: "retry-reply", content: "已完成" }));
      receive(event(++sequence, "DONE"));
      const finished = surface === "对话" ? state : store.tasks[0].state;
      assert.equal(finished.status, "done");
      assert.equal(finished.runId, "same-run");
      assert.ok(JSON.stringify(finished.timeline).includes("已完成"));
    });
  }
}

test("子任务失败显示可读原因，并保留正常结果", () => {
  const result = `Error: ${JSON.stringify({ executionMode: "PLATFORM_MANAGED_CHILD_RUN", parentRunId: "internal-parent", childRunId: "internal-child", status: "FAILED", error: "Request failed" })}`;
  assert.equal(formatSubAgentResult(result), "模型请求中断，请稍后继续。");
  assert.equal(formatSubAgentResult(JSON.stringify({ executionMode: "PLATFORM_MANAGED_CHILD_RUN", status: "COMPLETED", result: { message: "已保存 8 个场景" } })), "已保存 8 个场景");
  assert.equal(formatSubAgentResult("已完成"), "已完成");
});

load("components/dashboard/agent-pipeline/state.test.ts");

for (const surface of ["对话", "运行通知"]) {
  for (const child of [false, true]) {
    test(`${surface}用完整消息替换${child ? "子任务" : "主任务"}草稿并保持工具顺序`, async () => {
      let state = createInitialPipelineState();
      let store = { tasks: [{ id: "task", status: "running", state: { ...state, status: "running" } }] };
      const receive = surface === "对话"
        ? (value) => {
          state = reducePipelineEvent(state, value);
        }
        : createPipelineEventHandler({ notifySettlement: () => undefined, scheduleStatusSync: () => undefined },
          "task", (update) => {
            store = { ...store, ...update(store) };
          });
      let sequence = 0;
      const send = (output, values) => receive({ ...event(++sequence, output, values), schemaVersion: 2 });
      if (child) {
        send("TOOL_CALL", { replyId: "parent-reply", toolCalls: [{ id: "parent", name: "writer", arguments: "{}" }] });
      }
      const parent = child ? { parentToolCallId: "parent", sourceRunId: "child" } : {};
      send("REASONING", { ...parent, replyId: "reply", reasoningContent: "思考片段" });
      send("CONTENT", { ...parent, replyId: "reply", content: "已显示" });
      send("TOOL_CALL", { ...parent, replyId: "reply", toolCalls: [{ id: "lookup", name: "lookup", arguments: "{}" }] });
      const full = { ...parent, replyId: "reply", messageId: "saved-message", contentMode: "replace",
        content: "已显示的完整回复", reasoningContent: "完整思考", reasoningDurationMs: 42 };
      send("CONTENT", full);
      send("CONTENT", full);
      await new Promise((resolve) => setTimeout(resolve, 25));
      const updated = surface === "对话" ? state : store.tasks[0].state;
      const timeline = child ? updated.timeline[0].children : updated.timeline;
      assert.deepEqual(timeline.map((item) => item.type), ["reasoning", "content", "tool"]);
      assert.equal(timeline[0].text, "完整思考");
      assert.equal(timeline[0].durationMs, 42);
      assert.equal(timeline[1].text, "已显示的完整回复");
      assert.equal(timeline[2].id, "lookup");
    });
  }
}

test("子运行的审批使用子运行标识，展示流仍属于父运行", () => {
  const started = reducePipelineEvent(createInitialPipelineState(), {
    ...event(1, "TOOL_CALL", { replyId: "reply", toolCalls: [{ id: "write", name: "save", arguments: "{}" }] }), schemaVersion: 2,
  });
  const state = reducePipelineEvent(started, {
    ...event(2, "USER_CONFIRMATION_REQUIRED", { replyId: "reply", controlRunId: "child-run", expiresAt: "2026-11-24T12:00:00Z",
      pendingToolCalls: [{ toolCallId: "write", toolName: "save", input: {} }] }), schemaVersion: 2,
  });
  assert.equal(state.runId, "same-run");
  assert.equal(state.pendingConfirmation.runId, "child-run");
});
