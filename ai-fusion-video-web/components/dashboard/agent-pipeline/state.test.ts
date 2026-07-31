import assert from "node:assert/strict";
import test from "node:test";
import type { AiChatStreamEvent } from "@/lib/api/ai-pipeline";
import { createInitialPipelineState, reducePipelineEvent } from "./state";

function event(
  sequence: number,
  outputType: AiChatStreamEvent["outputType"],
  values: Partial<AiChatStreamEvent> = {},
): AiChatStreamEvent {
  return {
    schemaVersion: 1,
    runId: "run-1",
    sequence,
    outputType,
    ...values,
  };
}

function toolCallState() {
  return reducePipelineEvent(
    createInitialPipelineState(),
    event(1, "TOOL_CALL", {
      replyId: "reply-1",
      toolCalls: [{
        id: "tool-1",
        name: "update_script_info",
        arguments: "{\"scriptId\":1}",
      }],
    }),
  );
}

function pendingConfirmationState() {
  return reducePipelineEvent(
    toolCallState(),
    event(2, "USER_CONFIRMATION_REQUIRED", {
      replyId: "reply-1",
      expiresAt: "2026-07-30T12:00:00Z",
      pendingToolCalls: [{
        toolCallId: "tool-1",
        toolName: "update_script_info",
        argumentsPreview: "{\"scriptId\":1}",
      }],
    }),
  );
}

test("permission request moves the exact tool call into approval state", () => {
  const state = pendingConfirmationState();
  assert.equal(state.timeline[0]?.type, "tool");
  if (state.timeline[0]?.type !== "tool") throw new Error("Expected tool timeline item");
  assert.equal(state.timeline[0].status, "awaiting_approval");
  assert.equal(state.pendingConfirmation?.replyId, "reply-1");
});

test("approved decision is visible until the tool result arrives", () => {
  const state = reducePipelineEvent(
    pendingConfirmationState(),
    event(3, "USER_CONFIRM_RESULT", {
      replyId: "reply-1",
      decisions: [{ toolCallId: "tool-1", approved: true }],
    }),
  );
  assert.equal(state.timeline[0]?.type, "tool");
  if (state.timeline[0]?.type !== "tool") throw new Error("Expected tool timeline item");
  assert.equal(state.timeline[0].status, "approved");
  assert.equal(state.pendingConfirmation, undefined);
});

test("rejected decision is terminal and never remains calling", () => {
  const state = reducePipelineEvent(
    pendingConfirmationState(),
    event(3, "USER_CONFIRM_RESULT", {
      replyId: "reply-1",
      decisions: [{ toolCallId: "tool-1", approved: false }],
    }),
  );
  assert.equal(state.timeline[0]?.type, "tool");
  if (state.timeline[0]?.type !== "tool") throw new Error("Expected tool timeline item");
  assert.equal(state.timeline[0].status, "rejected");
  assert.notEqual(state.timeline[0].status, "calling");
});

test("confirmation result without exact decisions fails explicitly", () => {
  assert.throws(
    () => reducePipelineEvent(
      pendingConfirmationState(),
      event(3, "USER_CONFIRM_RESULT", { replyId: "reply-1" }),
    ),
    /has no decisions/,
  );
});

test("parallel tool confirmations merge into one exact batch", () => {
  const called = reducePipelineEvent(
    createInitialPipelineState(),
    event(1, "TOOL_CALL", {
      replyId: "reply-batch",
      toolCalls: [
        { id: "tool-1", name: "get_project_script", arguments: "{\"projectId\":1}" },
        { id: "tool-2", name: "get_project_storyboard", arguments: "{\"projectId\":1}" },
      ],
    }),
  );
  const firstPending = reducePipelineEvent(
    called,
    event(2, "USER_CONFIRMATION_REQUIRED", {
      replyId: "reply-batch",
      expiresAt: "2026-07-30T12:00:00Z",
      pendingToolCalls: [{
        toolCallId: "tool-1",
        toolName: "get_project_script",
        argumentsPreview: "{\"projectId\":1}",
      }],
    }),
  );
  const batchPending = reducePipelineEvent(
    firstPending,
    event(3, "USER_CONFIRMATION_REQUIRED", {
      replyId: "reply-batch",
      expiresAt: "2026-07-30T12:00:00Z",
      pendingToolCalls: [{
        toolCallId: "tool-2",
        toolName: "get_project_storyboard",
        argumentsPreview: "{\"projectId\":1}",
      }],
    }),
  );

  assert.deepEqual(
    batchPending.pendingConfirmation?.toolCalls.map((toolCall) => toolCall.toolCallId),
    ["tool-1", "tool-2"],
  );
  assert.deepEqual(
    batchPending.timeline.map((item) => item.type === "tool" ? item.status : item.type),
    ["awaiting_approval", "awaiting_approval"],
  );

  const rejected = reducePipelineEvent(
    batchPending,
    event(4, "USER_CONFIRM_RESULT", {
      replyId: "reply-batch",
      decisions: [
        { toolCallId: "tool-1", approved: false },
        { toolCallId: "tool-2", approved: false },
      ],
    }),
  );
  assert.deepEqual(
    rejected.timeline.map((item) => item.type === "tool" ? item.status : item.type),
    ["rejected", "rejected"],
  );
});

test("cancelled terminal clears a pending batch and cancels every tool", () => {
  const called = reducePipelineEvent(
    createInitialPipelineState(),
    event(1, "TOOL_CALL", {
      replyId: "reply-batch",
      toolCalls: [
        { id: "tool-1", name: "get_project", arguments: "{\"projectId\":1}" },
        {
          id: "tool-2",
          name: "get_project_script",
          arguments: "{\"projectId\":1}",
        },
      ],
    }),
  );
  const pending = reducePipelineEvent(
    called,
    event(2, "USER_CONFIRMATION_REQUIRED", {
      replyId: "reply-batch",
      expiresAt: "2026-07-30T12:00:00Z",
      pendingToolCalls: [
        {
          toolCallId: "tool-1",
          toolName: "get_project",
          argumentsPreview: "{\"projectId\":1}",
        },
        {
          toolCallId: "tool-2",
          toolName: "get_project_script",
          argumentsPreview: "{\"projectId\":1}",
        },
      ],
    }),
  );

  const cancelled = reducePipelineEvent(pending, event(3, "CANCELLED"));

  assert.equal(cancelled.pendingConfirmation, undefined);
  assert.deepEqual(
    cancelled.timeline.map((item) => item.type === "tool" ? item.status : item.type),
    ["cancelled", "cancelled"],
  );
});
