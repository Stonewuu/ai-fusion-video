import assert from "node:assert/strict";
import test from "node:test";
import type { AgentMessage } from "@/lib/api/ai-assistant";
import { messagesToTimeline } from "./history";

function message(
  id: number,
  values: Partial<AgentMessage>
): AgentMessage {
  return {
    id,
    conversationId: "conversation-1",
    role: "assistant",
    content: "",
    messageOrder: id,
    ...values,
  };
}

test("renders a persisted child validation error without a call projection", () => {
  const validationError =
    "Error: Parameter validation failed for tool 'save_storyboard_scene_shots'";
  const timeline = messagesToTimeline([
    message(1, {
      role: "tool",
      toolCallId: "parent-1",
      toolName: "episode_scene_writer",
      toolStatus: "running",
      content: "{\"scriptEpisodeId\":1}",
    }),
    message(2, {
      role: "tool",
      toolCallId: "call_00_X8zmhhAzrGWbLIf9j7iz6837",
      parentToolCallId: "parent-1",
      toolName: "save_storyboard_scene_shots",
      toolStatus: "error",
      content: validationError,
    }),
  ]);

  assert.equal(timeline.length, 1);
  const parent = timeline[0];
  assert.equal(parent?.type, "tool");
  if (parent?.type !== "tool") throw new Error("Expected parent tool item");
  assert.equal(parent.children?.length, 1);
  const child = parent.children?.[0];
  assert.equal(child?.type, "tool");
  if (child?.type !== "tool") throw new Error("Expected child tool item");
  assert.equal(child.status, "error");
  assert.equal(child.arguments, "");
  assert.equal(child.result, validationError);
});

test("merges a child call that is persisted after its result", () => {
  const timeline = messagesToTimeline([
    message(1, {
      role: "tool",
      toolCallId: "child-1",
      parentToolCallId: "parent-1",
      toolName: "get_project",
      toolStatus: "success",
      content: "{\"projectId\":1}",
    }),
    message(2, {
      role: "tool",
      toolCallId: "parent-1",
      toolName: "episode_scene_writer",
      toolStatus: "running",
      content: "{\"scriptEpisodeId\":1}",
    }),
    message(3, {
      role: "tool",
      toolCallId: "child-1",
      parentToolCallId: "parent-1",
      toolName: "get_project",
      toolStatus: "running",
      content: "{\"projectId\":1,\"detailLevel\":\"full\"}",
    }),
  ]);

  const parent = timeline[0];
  assert.equal(parent?.type, "tool");
  if (parent?.type !== "tool") throw new Error("Expected parent tool item");
  assert.equal(parent.children?.length, 1);
  const child = parent.children?.[0];
  assert.equal(child?.type, "tool");
  if (child?.type !== "tool") throw new Error("Expected child tool item");
  assert.equal(child.status, "done");
  assert.equal(
    child.arguments,
    "{\"projectId\":1,\"detailLevel\":\"full\"}"
  );
  assert.equal(child.result, "{\"projectId\":1}");
});

test("merges a root call that is persisted after its result", () => {
  const timeline = messagesToTimeline([
    message(1, {
      role: "tool",
      toolCallId: "tool-1",
      toolName: "get_project",
      toolStatus: "success",
      content: "{\"projectId\":1}",
    }),
    message(2, {
      role: "tool",
      toolCallId: "tool-1",
      toolName: "get_project",
      toolStatus: "running",
      content: "{\"projectId\":1,\"detailLevel\":\"full\"}",
    }),
  ]);

  assert.equal(timeline.length, 1);
  const tool = timeline[0];
  assert.equal(tool?.type, "tool");
  if (tool?.type !== "tool") throw new Error("Expected tool item");
  assert.equal(tool.status, "done");
  assert.equal(
    tool.arguments,
    "{\"projectId\":1,\"detailLevel\":\"full\"}"
  );
  assert.equal(tool.result, "{\"projectId\":1}");
});
