import test from "node:test";
import assert from "node:assert/strict";
import { createTypeScriptLoader } from "./helpers/load-typescript.mjs";

const load = createTypeScriptLoader();
const { createInitialPipelineState, reducePipelineEvent } = load("components/dashboard/agent-pipeline/state.ts");
const { createPipelineEventHandler } = load("lib/store/pipeline-event-handler.ts");
const { messagesToTimeline } = load("components/dashboard/notification-panel/history.ts");

function parallelConversation() {
  const events = [];
  const messages = [];
  let order = 0;
  const emit = (outputType, fields = {}) => events.push({
    schemaVersion: 2, runId: "root", sequence: events.length + 1, outputType, ...fields,
  });
  const save = (fields) => messages.push({ id: ++order, messageOrder: order, conversationId: "conversation", ...fields });
  const names = ["青松", "海蓝", "赤砂"];
  save({ role: "user", content: "请并行处理三个分集" });
  for (const [index, name] of names.entries()) {
    emit("TOOL_CALL", { replyId: "parent-reply", toolCalls: [{ id: `child-${index}`, name: "episode_scene_writer", arguments: JSON.stringify({ name }) }] });
    save({ role: "tool", toolCallId: `child-${index}`, toolName: "episode_scene_writer", toolStatus: "running", content: JSON.stringify({ name }) });
  }
  for (const [index, name] of names.entries()) {
    const owner = { sourceRunId: `run-${index}`, parentToolCallId: `child-${index}`, replyId: `reply-${index}` };
    emit("REASONING", { ...owner, reasoningContent: `${name}思考` });
    emit("CONTENT", { ...owner, content: `${name}草稿` });
    emit("TOOL_CALL", { ...owner, toolCalls: [{ id: "lookup", name: "get_script_episode", arguments: JSON.stringify({ index }) }] });
  }
  for (const index of [2, 0, 1]) {
    const name = names[index];
    const owner = { sourceRunId: `run-${index}`, parentToolCallId: `child-${index}`, replyId: `reply-${index}` };
    emit("CONTENT", { ...owner, messageId: `message-${index}`, contentMode: "replace", content: `${name}完整回复`, reasoningContent: `${name}完整思考` });
    save({ role: "assistant", parentToolCallId: owner.parentToolCallId, content: `${name}完整回复`, reasoningContent: `${name}完整思考` });
    save({ role: "tool", parentToolCallId: owner.parentToolCallId, toolCallId: "lookup", toolName: "get_script_episode", toolStatus: "running", content: JSON.stringify({ index }) });
    emit("TOOL_FINISHED", { ...owner, toolCallId: "lookup", toolName: "get_script_episode", toolStatus: "success", toolResult: `${name}查询结果` });
    save({ role: "tool", parentToolCallId: owner.parentToolCallId, toolCallId: "lookup", toolName: "get_script_episode", toolStatus: "success", content: `${name}查询结果` });
    emit("DONE", owner);
    emit("TOOL_FINISHED", { toolCallId: owner.parentToolCallId, toolName: "episode_scene_writer", toolStatus: "success", toolResult: `${name}完成` });
    save({ role: "tool", toolCallId: owner.parentToolCallId, toolName: "episode_scene_writer", toolStatus: "success", content: `${name}完成` });
  }
  emit("DONE");
  return { events, messages, names };
}

function verifyGroups(timeline, names) {
  assert.equal(timeline.length, 3);
  names.forEach((name, index) => {
    const parent = timeline[index];
    assert.equal(parent.id, `child-${index}`);
    assert.equal(parent.status, "done");
    assert.equal(parent.result, `${name}完成`);
    assert.deepEqual(parent.children.map((item) => item.type), ["reasoning", "content", "tool"]);
    assert.equal(parent.children[0].text, `${name}完整思考`);
    assert.equal(parent.children[1].text, `${name}完整回复`);
    assert.equal(parent.children[2].result, `${name}查询结果`);
    assert.equal(parent.children[2].status, "done");
  });
}

test("三个并行子任务乱序完成时，两处实时界面与历史重建保持相同归属", async () => {
  const fixture = parallelConversation();
  let assistant = createInitialPipelineState();
  let store = { tasks: [{ id: "task", status: "running", state: { ...assistant, status: "running" } }] };
  const receive = createPipelineEventHandler({ notifySettlement: () => undefined, scheduleStatusSync: () => undefined },
    "task", (update) => {
      store = { ...store, ...update(store) };
    });
  for (const event of fixture.events) {
    assistant = reducePipelineEvent(assistant, event);
    receive(event);
    if (event.outputType === "DONE" && event.parentToolCallId) {
      assert.equal(assistant.status, "running");
    }
  }
  await new Promise((resolve) => setTimeout(resolve, 30));
  verifyGroups(assistant.timeline, fixture.names);
  verifyGroups(store.tasks[0].state.timeline, fixture.names);
  verifyGroups(messagesToTimeline(fixture.messages.filter((message) => message.role !== "user")), fixture.names);
});

test("新页面从头重建，并行子任务的重复序号不会产生重复消息", () => {
  const fixture = parallelConversation();
  let reopened = createInitialPipelineState();
  for (const event of fixture.events) {
    reopened = reducePipelineEvent(reopened, event);
    reopened = reducePipelineEvent(reopened, event);
  }
  verifyGroups(reopened.timeline, fixture.names);
  assert.equal(reopened.lastSequence, fixture.events.length);
});

test("Redis 已过期时两处界面只凭数据库历史恢复三个子任务及最终状态", async () => {
  const fixture = parallelConversation();
  const history = { schemaVersion: 2, runId: "root", sequence: 0, outputType: "HISTORY",
    runStatus: "COMPLETED", finished: true, historyMessages: fixture.messages.filter((message) => message.role !== "user"),
    pendingEvents: [{ outputType: "DONE" }] };
  const reopened = reducePipelineEvent({ ...createInitialPipelineState(), runId: "root", lastSequence: 999 }, history);
  verifyGroups(reopened.timeline, fixture.names);
  assert.equal(reopened.status, "done");
  assert.equal(reopened.lastSequence, 0);
  let store = { tasks: [{ id: "task", status: "running", state: { ...createInitialPipelineState(), status: "running", lastSequence: 999 } }] };
  const settled = [];
  const receive = createPipelineEventHandler({ notifySettlement: (_, status) => settled.push(status), scheduleStatusSync: () => undefined },
    "task", (update) => {
      store = { ...store, ...update(store) };
    });
  receive(history);
  verifyGroups(store.tasks[0].state.timeline, fixture.names);
  assert.equal(store.tasks[0].status, "done");
  assert.deepEqual(settled, ["done"]);
});

test("运行中重开恢复已有消息和未完成草稿，单个子任务取消不改变其他子任务", () => {
  const history = { schemaVersion: 2, runId: "root", sequence: 30, outputType: "HISTORY",
    runStatus: "RUNNING", finished: false,
    historyMessages: [0, 1, 2].map((index) => ({ id: index + 1, messageOrder: index + 1, role: "tool",
      toolCallId: `child-${index}`, toolName: "episode_scene_writer", toolStatus: "running", content: "{}" })),
    pendingEvents: [{ outputType: "CONTENT", parentToolCallId: "child-0", replyId: "reply-0", content: "已输出" },
      { outputType: "CANCELLED", parentToolCallId: "child-1", agentName: "子任务1" }] };
  let state = reducePipelineEvent(createInitialPipelineState(), history);
  assert.deepEqual(state.timeline.map((item) => item.status), ["calling", "cancelled", "calling"]);
  assert.equal(state.timeline[0].children[0].text, "已输出");
  state = reducePipelineEvent(state, { schemaVersion: 2, runId: "root", sequence: 31, outputType: "CONTENT",
    parentToolCallId: "child-0", replyId: "reply-0", content: "继续输出" });
  assert.equal(state.timeline[0].children[0].text, "已输出继续输出");
  state = reducePipelineEvent(state, { schemaVersion: 2, runId: "root", sequence: 32, outputType: "CANCELLED" });
  assert.equal(state.status, "cancelled");
  assert.deepEqual(state.timeline.map((item) => item.status), ["cancelled", "cancelled", "cancelled"]);
  assert.equal(state.timeline[0].children[0].text, "已输出继续输出");
});

test("父任务取消时已经完成的子任务保持完成，重开历史也保留相同状态和参数", () => {
  const messages = [0, 1, 2].map((index) => ({ id: index + 1, messageOrder: index + 1, role: "tool",
    toolCallId: `child-${index}`, toolName: "episode_scene_writer", toolStatus: "cancelled",
    content: JSON.stringify({ scriptEpisodeId: index }), childRunStatus: index < 2 ? "COMPLETED" : "CANCELLED" }));
  const history = { schemaVersion: 2, runId: "root", sequence: 0, outputType: "HISTORY",
    runStatus: "CANCELLED", finished: true, historyMessages: messages, pendingEvents: [{ outputType: "CANCELLED" }] };
  const restored = reducePipelineEvent(createInitialPipelineState(), history);
  assert.deepEqual(restored.timeline.map((item) => item.status), ["done", "done", "cancelled"]);
  assert.deepEqual(restored.timeline.map((item) => JSON.parse(item.arguments).scriptEpisodeId), [0, 1, 2]);
  let live = createInitialPipelineState();
  live = reducePipelineEvent(live, { schemaVersion: 2, runId: "root", sequence: 1, outputType: "TOOL_CALL", replyId: "reply",
    toolCalls: messages.map((message) => ({ id: message.toolCallId, name: message.toolName, arguments: message.content })) });
  for (const index of [0, 1]) {
    live = reducePipelineEvent(live, { schemaVersion: 2, runId: "root", sequence: index + 2,
      outputType: "DONE", parentToolCallId: `child-${index}`, agentName: "子任务" });
  }
  live = reducePipelineEvent(live, { schemaVersion: 2, runId: "root", sequence: 4, outputType: "CANCELLED" });
  assert.deepEqual(live.timeline.map((item) => item.status), restored.timeline.map((item) => item.status));
});

test("Pipeline 续跑接收当前运行快照时保留上一轮历史", () => {
  const prefix = [{ type: "content", text: "上一轮已经保存的回复" }];
  let store = { tasks: [{ id: "task", status: "running", state: { ...createInitialPipelineState(),
    status: "running", timeline: prefix, historyPrefix: prefix } }] };
  const receive = createPipelineEventHandler({ notifySettlement: () => undefined, scheduleStatusSync: () => undefined },
    "task", (update) => {
      store = { ...store, ...update(store) };
    });
  receive({ schemaVersion: 2, runId: "next-run", sequence: 0, outputType: "HISTORY", runStatus: "COMPLETED", finished: true,
    historyMessages: [{ id: 2, messageOrder: 2, role: "assistant", content: "本轮新回复" }],
    pendingEvents: [{ outputType: "DONE" }] });
  assert.deepEqual(store.tasks[0].state.timeline.map((item) => item.text), ["上一轮已经保存的回复", "本轮新回复"]);
  assert.equal(store.tasks[0].status, "done");
});
