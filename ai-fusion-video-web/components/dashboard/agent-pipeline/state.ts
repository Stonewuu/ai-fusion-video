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
  reasoningContent: string
): SubTimelineItem[] {
  const last = children[children.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...children.slice(0, -1),
      { ...last, text: last.text + reasoningContent },
    ];
  }

  return [...children, { type: "reasoning", text: reasoningContent }];
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
  reasoningContent: string
): TimelineItem[] {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...timeline.slice(0, -1),
      { ...last, text: last.text + reasoningContent },
    ];
  }

  return [...timeline, { type: "reasoning", text: reasoningContent }];
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
  content: string,
  reasoningDurationMs?: number
): SubTimelineItem[] {
  let updated = [...children];
  if (reasoningDurationMs) {
    updated = updateLastSubTimelineReasoningDuration(updated, reasoningDurationMs);
  }
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

  switch (event.outputType) {
    case "REASONING":
      if (event.reasoningContent) {
        if (isSubAgent) {
          next.timeline = appendToToolChildren(
            next.timeline,
            event.parentToolCallId!,
            (children) =>
              appendReasoningToSubTimeline(children, event.reasoningContent!)
          );
        } else {
          next.status = prev.status === "cancelling" ? "cancelling" : "reasoning";
          next.reasoningText += event.reasoningContent;
          next.timeline = appendReasoningToTimeline(
            next.timeline,
            event.reasoningContent
          );
        }
      }
      return next;

    case "CONTENT":
      next.status = prev.status === "cancelling" ? "cancelling" : "running";
      if (event.reasoningDurationMs && !isSubAgent) {
        next.reasoningDurationMs = event.reasoningDurationMs;
        next.timeline = updateLastTimelineReasoningDuration(
          next.timeline,
          event.reasoningDurationMs
        );
      }
      if (event.content) {
        if (isSubAgent) {
          next.timeline = appendToToolChildren(
            next.timeline,
            event.parentToolCallId!,
            (children) =>
              appendContentToSubTimeline(
                children,
                event.content!,
                event.reasoningDurationMs
              )
          );
        } else {
          next.timeline = appendContentToTimeline(next.timeline, event.content);
        }
      }
      return next;

    case "TOOL_CALL":
      next.status = prev.status === "cancelling" ? "cancelling" : "running";
      if (event.toolCalls) {
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
              status: "calling",
              agentName: event.agentName,
            });
          }
        }
      }
      return next;

    case "TOOL_FINISHED":
      if (event.toolCallId) {
        const toolItemStatus = finishedToolTimelineStatus(event.toolStatus);
        if (isSubAgent) {
          next.timeline = appendToToolChildren(
            next.timeline,
            event.parentToolCallId!,
            (children) =>
              children.map((child) =>
                child.type === "tool" && child.id === event.toolCallId
                  ? { ...child, status: toolItemStatus, result: event.toolResult }
                  : child
              )
          );
        } else {
          next.timeline = next.timeline.map((item) =>
            item.type === "tool" && item.id === event.toolCallId
              ? { ...item, status: toolItemStatus, result: event.toolResult }
              : item
          );
        }
      }
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

    case "DONE":
      if (event.parentToolCallId || event.agentName) return next;
      next.status = "done";
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
        next.error = event.error || "未知错误";
      }
      return next;

    case "CANCELLED":
      if (!event.parentToolCallId && !event.agentName) {
        next.status = "cancelled";
        next.timeline = cancelCallingTimelineTools(next.timeline);
      }
      return next;

    default:
      return next;
  }
}
