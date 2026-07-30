import type { AiChatStreamEvent } from "@/lib/api/ai-pipeline";
import type {
  AgentPipelineState,
  SubTimelineItem,
  TimelineItem,
} from "./types";
import {
  cancelCallingTimelineTools,
  finishedToolTimelineStatus,
  type ToolTimelineStatus,
} from "@/lib/store/pipeline-timeline";

export function createInitialPipelineState(): AgentPipelineState {
  return {
    status: "idle",
    reasoningText: "",
    timeline: [],
    lastSequence: 0,
  };
}

export function createPendingPipelineState(): AgentPipelineState {
  return {
    status: "reasoning",
    reasoningText: "",
    timeline: [],
    lastSequence: 0,
  };
}

function appendReasoningToSubTimeline(
  children: SubTimelineItem[],
  reasoningContent: string,
  startedAtMs?: number
): SubTimelineItem[] {
  const last = children[children.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...children.slice(0, -1),
      {
        ...last,
        text: last.text + reasoningContent,
        startedAtMs: last.startedAtMs ?? startedAtMs,
      },
    ];
  }

  return [
    ...children,
    {
      type: "reasoning",
      text: reasoningContent,
      ...(startedAtMs !== undefined ? { startedAtMs } : {}),
    },
  ];
}

function updateLastSubTimelineReasoningDuration(
  children: SubTimelineItem[],
  durationMs: number
): SubTimelineItem[] {
  for (let index = children.length - 1; index >= 0; index--) {
    const item = children[index];
    if (item.type === "reasoning") {
      return children.map((child, childIndex) =>
        childIndex === index && child.type === "reasoning"
          ? { ...child, durationMs }
          : child
      );
    }
  }

  return children;
}

function appendReasoningToTimeline(
  timeline: TimelineItem[],
  reasoningContent: string,
  startedAtMs?: number
): TimelineItem[] {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...timeline.slice(0, -1),
      {
        ...last,
        text: last.text + reasoningContent,
        startedAtMs: last.startedAtMs ?? startedAtMs,
      },
    ];
  }

  return [
    ...timeline,
    {
      type: "reasoning",
      text: reasoningContent,
      ...(startedAtMs !== undefined ? { startedAtMs } : {}),
    },
  ];
}

function updateLastTimelineReasoningDuration(
  timeline: TimelineItem[],
  durationMs: number
): TimelineItem[] {
  for (let index = timeline.length - 1; index >= 0; index--) {
    const item = timeline[index];
    if (item.type === "reasoning") {
      return timeline.map((timelineItem, timelineIndex) =>
        timelineIndex === index && timelineItem.type === "reasoning"
          ? { ...timelineItem, durationMs }
          : timelineItem
      );
    }
  }

  return timeline;
}

function updateToolStatus(
  timeline: TimelineItem[],
  toolCallId: string,
  status: ToolTimelineStatus
): TimelineItem[] {
  return timeline.map((item) =>
    item.type === "tool" && item.id === toolCallId
      ? { ...item, status }
      : item
  );
}

type ToolStatusUpdate = {
  status: ToolTimelineStatus;
  expectedStatus: ToolTimelineStatus;
  expectedName?: string;
};

function updateConfirmationToolStatuses(
  timeline: TimelineItem[],
  parentToolCallId: string | undefined,
  updates: ReadonlyMap<string, ToolStatusUpdate>,
): TimelineItem[] {
  if (updates.size === 0) {
    throw new Error("Tool confirmation must contain at least one decision");
  }

  const updateChildren = (children: SubTimelineItem[]): SubTimelineItem[] => {
    for (const [toolCallId, update] of updates) {
      const matches = children.filter(
        (child) => child.type === "tool" && child.id === toolCallId,
      );
      if (matches.length !== 1) {
        throw new Error(`Expected one child tool call for confirmation: ${toolCallId}`);
      }
      const match = matches[0];
      if (match.type !== "tool" || match.status !== update.expectedStatus) {
        throw new Error(`Invalid confirmation state for child tool call: ${toolCallId}`);
      }
      if (update.expectedName !== undefined && match.name !== update.expectedName) {
        throw new Error(`Confirmation tool name mismatch: ${toolCallId}`);
      }
    }
    return children.map((child) => {
      if (child.type !== "tool") return child;
      const update = updates.get(child.id);
      return update ? { ...child, status: update.status } : child;
    });
  };

  if (parentToolCallId) {
    const parents = timeline.filter(
      (item) => item.type === "tool" && item.id === parentToolCallId,
    );
    if (parents.length !== 1 || parents[0].type !== "tool" || !parents[0].children) {
      throw new Error(`Confirmation parent tool call is missing: ${parentToolCallId}`);
    }
    return timeline.map((item) => item.type === "tool" && item.id === parentToolCallId
      ? { ...item, children: updateChildren(item.children!) }
      : item);
  }

  for (const [toolCallId, update] of updates) {
    const matches = timeline.filter(
      (item) => item.type === "tool" && item.id === toolCallId,
    );
    if (matches.length !== 1 || matches[0].type !== "tool") {
      throw new Error(`Expected one tool call for confirmation: ${toolCallId}`);
    }
    const match = matches[0];
    if (match.status !== update.expectedStatus) {
      throw new Error(`Invalid confirmation state for tool call: ${toolCallId}`);
    }
    if (update.expectedName !== undefined && match.name !== update.expectedName) {
      throw new Error(`Confirmation tool name mismatch: ${toolCallId}`);
    }
  }
  return timeline.map((item) => {
    if (item.type !== "tool") return item;
    const update = updates.get(item.id);
    return update ? { ...item, status: update.status } : item;
  });
}

function requireConfirmationDecisions(
  event: AiChatStreamEvent,
  pending: NonNullable<AgentPipelineState["pendingConfirmation"]>,
): Map<string, boolean> {
  if (!event.replyId || event.replyId !== pending.replyId) {
    throw new Error("USER_CONFIRM_RESULT replyId does not match the pending confirmation");
  }
  if (event.parentToolCallId !== pending.parentToolCallId) {
    throw new Error("USER_CONFIRM_RESULT parent tool identity does not match");
  }
  if (!event.decisions?.length) {
    throw new Error("USER_CONFIRM_RESULT has no decisions");
  }
  const decisions = new Map<string, boolean>();
  for (const decision of event.decisions) {
    if (!decision.toolCallId || typeof decision.approved !== "boolean") {
      throw new Error("USER_CONFIRM_RESULT contains an invalid decision");
    }
    if (decisions.has(decision.toolCallId)) {
      throw new Error(`USER_CONFIRM_RESULT contains a duplicate decision: ${decision.toolCallId}`);
    }
    decisions.set(decision.toolCallId, decision.approved);
  }
  const pendingIds = new Set(pending.toolCalls.map((toolCall) => toolCall.toolCallId));
  if (
    pendingIds.size !== pending.toolCalls.length
    || decisions.size !== pendingIds.size
    || [...decisions.keys()].some((toolCallId) => !pendingIds.has(toolCallId))
  ) {
    throw new Error("USER_CONFIRM_RESULT decisions do not match the pending tool calls");
  }
  return decisions;
}

function finishToolCall(
  timeline: TimelineItem[],
  parentToolCallId: string | undefined,
  toolCallId: string,
  status: Extract<ToolTimelineStatus, "done" | "error" | "cancelled">,
  result: string | null | undefined,
): TimelineItem[] {
  const validPreviousStatus = (value: ToolTimelineStatus) =>
    value === "calling" || value === "approved";

  if (parentToolCallId) {
    const parents = timeline.filter(
      (item) => item.type === "tool" && item.id === parentToolCallId,
    );
    if (parents.length !== 1 || parents[0].type !== "tool" || !parents[0].children) {
      throw new Error(`Finished tool parent is missing: ${parentToolCallId}`);
    }
    const children = parents[0].children;
    const matches = children.filter(
      (child) => child.type === "tool" && child.id === toolCallId,
    );
    if (
      matches.length !== 1
      || matches[0].type !== "tool"
      || !validPreviousStatus(matches[0].status)
    ) {
      throw new Error(`Finished child tool has no valid in-progress call: ${toolCallId}`);
    }
    return timeline.map((item) => item.type === "tool" && item.id === parentToolCallId
      ? {
          ...item,
          children: item.children!.map((child) =>
            child.type === "tool" && child.id === toolCallId
              ? { ...child, status, result }
              : child),
        }
      : item);
  }

  const matches = timeline.filter(
    (item) => item.type === "tool" && item.id === toolCallId,
  );
  if (
    matches.length !== 1
    || matches[0].type !== "tool"
    || !validPreviousStatus(matches[0].status)
  ) {
    throw new Error(`Finished tool has no valid in-progress call: ${toolCallId}`);
  }
  return timeline.map((item) => item.type === "tool" && item.id === toolCallId
    ? { ...item, status, result }
    : item);
}

function appendToToolChildren(
  timeline: TimelineItem[],
  parentToolCallId: string,
  updater: (children: SubTimelineItem[]) => SubTimelineItem[]
): TimelineItem[] {
  return timeline.map((item) =>
    item.type === "tool" && item.id === parentToolCallId
      ? { ...item, children: updater(item.children ?? []) }
      : item
  );
}

function appendContentToSubTimeline(
  children: SubTimelineItem[],
  content: string
): SubTimelineItem[] {
  const updated = [...children];
  const last = updated[updated.length - 1];
  if (last && last.type === "content") {
    return [
      ...updated.slice(0, -1),
      { ...last, text: last.text + content },
    ];
  }

  return [...updated, { type: "content", text: content }];
}

function appendContentToTimeline(
  timeline: TimelineItem[],
  content: string
): TimelineItem[] {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "content") {
    return [
      ...timeline.slice(0, -1),
      { ...last, text: last.text + content },
    ];
  }

  return [...timeline, { type: "content", text: content }];
}

function validTimestamp(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

function validDuration(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

export function reducePipelineEvent(
  prev: AgentPipelineState,
  event: AiChatStreamEvent
): AgentPipelineState {
  if (prev.runId && prev.runId !== event.runId) {
    throw new Error("Pipeline event belongs to a different run");
  }
  if (event.sequence <= prev.lastSequence) {
    return prev;
  }
  const next: AgentPipelineState = {
    ...prev,
    timeline: [...prev.timeline],
    runId: event.runId,
    lastSequence: event.sequence,
    error: undefined,
  };

  if (event.conversationId) {
    next.conversationId = event.conversationId;
  }

  const isSubAgent = !!event.parentToolCallId;
  const eventReasoningDurationMs = validDuration(event.reasoningDurationMs);

  if (eventReasoningDurationMs !== undefined) {
    if (isSubAgent) {
      next.timeline = appendToToolChildren(
        next.timeline,
        event.parentToolCallId!,
        (children) =>
          updateLastSubTimelineReasoningDuration(
            children,
            eventReasoningDurationMs
          )
      );
    } else {
      next.reasoningDurationMs = eventReasoningDurationMs;
      next.timeline = updateLastTimelineReasoningDuration(
        next.timeline,
        eventReasoningDurationMs
      );
    }
  }

  switch (event.outputType) {
    case "REASONING":
      if (event.reasoningContent) {
        const reasoningStartTime = validTimestamp(event.reasoningStartTime);
        if (isSubAgent) {
          next.timeline = appendToToolChildren(
            next.timeline,
            event.parentToolCallId!,
            (children) =>
              appendReasoningToSubTimeline(
                children,
                event.reasoningContent!,
                reasoningStartTime
              )
          );
        } else {
          next.status = prev.status === "cancelling" ? "cancelling" : "reasoning";
          const last = next.timeline[next.timeline.length - 1];
          if (!last || last.type !== "reasoning") {
            next.reasoningText = "";
            next.reasoningDurationMs = undefined;
            next.reasoningStartTime = reasoningStartTime;
          } else if (next.reasoningStartTime === undefined) {
            next.reasoningStartTime = reasoningStartTime;
          }
          next.reasoningText += event.reasoningContent;
          next.timeline = appendReasoningToTimeline(
            next.timeline,
            event.reasoningContent,
            reasoningStartTime
          );
        }
      }
      return next;

    case "CONTENT":
      next.status = prev.status === "cancelling" ? "cancelling" : "running";
      if (event.content) {
        if (isSubAgent) {
          next.timeline = appendToToolChildren(
            next.timeline,
            event.parentToolCallId!,
            (children) =>
              appendContentToSubTimeline(
                children,
                event.content!
              )
          );
        } else {
          next.timeline = appendContentToTimeline(next.timeline, event.content);
        }
      }
      return next;

    case "TOOL_CALL":
      next.status = prev.status === "cancelling" ? "cancelling" : "running";
      if (!event.replyId || !event.toolCalls?.length) {
        throw new Error("TOOL_CALL event has no replyId or tool calls");
      }
      {
        for (const toolCall of event.toolCalls) {
          if (isSubAgent) {
            next.timeline = appendToToolChildren(
              next.timeline,
              event.parentToolCallId!,
              (children) => {
                if (
                  children.some(
                    (child) => child.type === "tool" && child.id === toolCall.id
                  )
                ) {
                  return children;
                }
                return [
                  ...children,
                  {
                    type: "tool",
                    id: toolCall.id,
                    name: toolCall.name,
                    arguments: toolCall.arguments,
                    batchId: event.replyId,
                    status: "calling",
                  },
                ];
              }
            );
          } else if (
            !next.timeline.some(
              (item) => item.type === "tool" && item.id === toolCall.id
            )
          ) {
            next.timeline.push({
              type: "tool",
              id: toolCall.id,
              name: toolCall.name,
              arguments: toolCall.arguments,
              batchId: event.replyId,
              status: "calling",
              agentName: event.agentName,
            });
          }
        }
      }
      return next;

    case "TOOL_FINISHED":
      if (!event.toolCallId) {
        throw new Error("TOOL_FINISHED event has no toolCallId");
      }
      next.timeline = finishToolCall(
        next.timeline,
        event.parentToolCallId,
        event.toolCallId,
        finishedToolTimelineStatus(event.toolStatus),
        event.toolResult,
      );
      return next;

    case "SUB_AGENT_FINISHED":
      if (isSubAgent) {
        next.timeline = updateToolStatus(
          next.timeline,
          event.parentToolCallId!,
          "done"
        );
      }
      return next;

    case "USER_CONFIRMATION_REQUIRED":
      if (!event.replyId || !event.pendingToolCalls?.length || !event.expiresAt) {
        throw new Error("Invalid USER_CONFIRMATION_REQUIRED event");
      }
      {
        const existing = prev.pendingConfirmation;
        if (existing && (
          existing.runId !== event.runId
          || existing.replyId !== event.replyId
          || existing.parentToolCallId !== event.parentToolCallId
          || existing.expiresAt !== event.expiresAt
        )) {
          throw new Error("Concurrent tool confirmation batches have different identities");
        }
        const existingToolCalls = new Map(
          (existing?.toolCalls ?? []).map((toolCall) => [toolCall.toolCallId, toolCall]),
        );
        const updates = new Map<string, ToolStatusUpdate>();
        const appendedToolCalls: NonNullable<AiChatStreamEvent["pendingToolCalls"]> = [];
        for (const toolCall of event.pendingToolCalls) {
          if (updates.has(toolCall.toolCallId)) {
            throw new Error(`Duplicate pending tool call: ${toolCall.toolCallId}`);
          }
          const existingToolCall = existingToolCalls.get(toolCall.toolCallId);
          if (existingToolCall && (
            existingToolCall.toolName !== toolCall.toolName
            || existingToolCall.argumentsPreview !== toolCall.argumentsPreview
          )) {
            throw new Error(`Pending tool call changed within its batch: ${toolCall.toolCallId}`);
          }
          updates.set(toolCall.toolCallId, {
            status: "awaiting_approval",
            expectedStatus: existingToolCall ? "awaiting_approval" : "calling",
            expectedName: toolCall.toolName,
          });
          if (!existingToolCall) appendedToolCalls.push(toolCall);
        }
        if (existing
          && (existing.submitting || Object.keys(existing.decisions).length > 0)
          && appendedToolCalls.length > 0) {
          throw new Error("Tool confirmation batch changed after a decision was submitted");
        }
        next.timeline = updateConfirmationToolStatuses(
          next.timeline,
          event.parentToolCallId,
          updates,
        );
        next.pendingConfirmation = existing
          ? {
              ...existing,
              toolCalls: [...existing.toolCalls, ...appendedToolCalls],
            }
          : {
              runId: event.runId,
              replyId: event.replyId,
              ...(event.parentToolCallId
                ? { parentToolCallId: event.parentToolCallId }
                : {}),
              toolCalls: event.pendingToolCalls,
              expiresAt: event.expiresAt,
              decisions: {},
              submitting: false,
            };
      }
      next.status = prev.status === "cancelling" ? "cancelling" : "running";
      return next;

    case "USER_CONFIRM_RESULT":
      if (!prev.pendingConfirmation) {
        throw new Error("USER_CONFIRM_RESULT has no pending confirmation");
      }
      {
        const decisions = requireConfirmationDecisions(event, prev.pendingConfirmation);
        const updates = new Map<string, ToolStatusUpdate>();
        for (const [toolCallId, approved] of decisions) {
          updates.set(toolCallId, {
            status: approved ? "approved" : "rejected",
            expectedStatus: "awaiting_approval",
          });
        }
        next.timeline = updateConfirmationToolStatuses(
          next.timeline,
          event.parentToolCallId,
          updates,
        );
      }
      next.pendingConfirmation = undefined;
      next.status = prev.status === "cancelling" ? "cancelling" : "running";
      return next;

    case "DONE":
      if (event.parentToolCallId || event.agentName) return next;
      next.status = "done";
      next.pendingConfirmation = undefined;
      if (event.content) {
        next.timeline = appendContentToTimeline(next.timeline, event.content);
      }
      return next;

    case "ERROR":
      if (isSubAgent) {
        next.timeline = updateToolStatus(
          next.timeline,
          event.parentToolCallId!,
          "error"
        );
        next.timeline = appendToToolChildren(
          next.timeline,
          event.parentToolCallId!,
          (children) => [
            ...children,
            {
              type: "content",
              text: `❌ ${event.agentName || "子Agent"} 出错: ${event.error || "未知错误"}`,
            },
          ]
        );
      } else {
        next.status = "error";
        next.pendingConfirmation = undefined;
        next.error = event.error || "未知错误";
      }
      return next;

    case "CANCELLED":
      if (!event.parentToolCallId && !event.agentName) {
        next.status = "cancelled";
        next.pendingConfirmation = undefined;
        next.timeline = cancelCallingTimelineTools(next.timeline);
      }
      return next;

    default:
      return next;
  }
}
