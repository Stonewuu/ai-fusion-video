import type {
  AgentConversation,
  AgentMessage,
} from "@/lib/api/ai-assistant";
import type {
  SubTimelineItem,
  TimelineItem,
} from "@/lib/store/pipeline-store";
import { isSubAgentTool } from "./constants";

function pushReasoningToTimeline(
  timeline: TimelineItem[],
  text: string,
  durationMs?: number
) {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "reasoning") {
    last.text += text;
    if (durationMs !== undefined) {
      last.durationMs = durationMs;
    }
    return;
  }

  timeline.push({
    type: "reasoning",
    text,
    ...(durationMs !== undefined ? { durationMs } : {}),
  });
}

function pushContentToTimeline(timeline: TimelineItem[], text: string) {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "content") {
    last.text = last.text.endsWith("\n\n")
      ? last.text + text
      : last.text.endsWith("\n")
        ? `${last.text}\n${text}`
        : `${last.text}\n\n${text}`;
    return;
  }

  timeline.push({ type: "content", text });
}

function updateLastTimelineReasoningDuration(
  timeline: TimelineItem[],
  durationMs: number
) {
  for (let index = timeline.length - 1; index >= 0; index--) {
    const item = timeline[index];
    if (item.type === "reasoning") {
      item.durationMs = durationMs;
      return;
    }
  }
}

function pushReasoningToSubTimeline(
  children: SubTimelineItem[],
  text: string,
  durationMs?: number
) {
  const last = children[children.length - 1];
  if (last && last.type === "reasoning") {
    last.text += text;
    if (durationMs !== undefined) {
      last.durationMs = durationMs;
    }
    return;
  }

  children.push({
    type: "reasoning",
    text,
    ...(durationMs !== undefined ? { durationMs } : {}),
  });
}

function pushContentToSubTimeline(children: SubTimelineItem[], text: string) {
  const last = children[children.length - 1];
  if (last && last.type === "content") {
    last.text = last.text.endsWith("\n\n")
      ? last.text + text
      : last.text.endsWith("\n")
        ? `${last.text}\n${text}`
        : `${last.text}\n\n${text}`;
    return;
  }

  children.push({ type: "content", text });
}

function updateLastSubTimelineReasoningDuration(
  children: SubTimelineItem[],
  durationMs: number
) {
  for (let index = children.length - 1; index >= 0; index--) {
    const item = children[index];
    if (item.type === "reasoning") {
      item.durationMs = durationMs;
      return;
    }
  }
}

function normalizeToolResult(
  toolName: string | undefined,
  content: string | undefined
): string | undefined {
  if (!content || !toolName || !isSubAgentTool(toolName)) {
    return content;
  }

  try {
    const value: unknown = JSON.parse(content);
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return content;
    }
    const result = (value as Record<string, unknown>).result;
    if (typeof result === "string" && result.trim()) {
      return result;
    }
    const error = (value as Record<string, unknown>).error;
    if (typeof error === "string" && error.trim()) {
      return error;
    }
  } catch {
    // 非 JSON 的子 Agent 结果直接按原文本展示。
  }

  return content;
}

export function messagesToTimeline(messages: AgentMessage[]): TimelineItem[] {
  const timeline: TimelineItem[] = [];
  const toolIndexMap = new Map<string, number>();
  const pendingParentUpdates = new Map<
    string,
    Array<(children: SubTimelineItem[]) => void>
  >();

  const registerTool = (toolCallId: string, index: number) => {
    toolIndexMap.set(toolCallId, index);
    const pendingUpdates = pendingParentUpdates.get(toolCallId);
    if (!pendingUpdates?.length) return;

    const parentItem = timeline[index];
    if (parentItem.type !== "tool") return;
    if (!parentItem.children) {
      parentItem.children = [];
    }
    for (const update of pendingUpdates) {
      update(parentItem.children);
    }
    pendingParentUpdates.delete(toolCallId);
  };

  const appendToParentChildren = (
    parentToolCallId: string,
    updater: (children: SubTimelineItem[]) => void
  ) => {
    const parentIdx = toolIndexMap.get(parentToolCallId);
    if (parentIdx === undefined) {
      const pendingUpdates = pendingParentUpdates.get(parentToolCallId) ?? [];
      pendingUpdates.push(updater);
      pendingParentUpdates.set(parentToolCallId, pendingUpdates);
      return;
    }

    const parentItem = timeline[parentIdx];
    if (parentItem.type !== "tool") return;

    if (!parentItem.children) {
      parentItem.children = [];
    }

    updater(parentItem.children);
  };

  for (const msg of messages) {
    if (msg.role === "tool") {
      const toolCallId = msg.toolCallId || `hist-tool-${msg.id}`;

      if (msg.parentToolCallId) {
        appendToParentChildren(msg.parentToolCallId, (children) => {
          if (msg.toolStatus === "running") {
            children.push({
              type: "tool",
              id: toolCallId,
              name: msg.toolName || "tool",
              arguments: msg.content || "",
              status: "calling",
            });
            return;
          }

          const existingChild = children.find(
            (child) => child.type === "tool" && child.id === toolCallId
          );
          if (existingChild && existingChild.type === "tool") {
            existingChild.status =
              msg.toolStatus === "error" ? "error" : "done";
            existingChild.result = normalizeToolResult(
              msg.toolName,
              msg.content
            );
            return;
          }

          children.push({
            type: "tool",
            id: toolCallId,
            name: msg.toolName || "tool",
            arguments: "",
            status: msg.toolStatus === "error" ? "error" : "done",
            result: normalizeToolResult(msg.toolName, msg.content),
          });
        });
        continue;
      }

      if (msg.toolStatus === "running") {
        const idx = timeline.length;
        timeline.push({
          type: "tool",
          id: toolCallId,
          name: msg.toolName || "tool",
          arguments: msg.content || "",
          status: "calling",
        });
        registerTool(toolCallId, idx);
        continue;
      }

      const existingIdx = toolIndexMap.get(toolCallId);
      if (existingIdx !== undefined) {
        const existingItem = timeline[existingIdx];
        if (existingItem.type === "tool") {
          existingItem.status =
            msg.toolStatus === "error" ? "error" : "done";
          existingItem.result = normalizeToolResult(msg.toolName, msg.content);
        }
        continue;
      }

      const idx = timeline.length;
      timeline.push({
        type: "tool",
        id: toolCallId,
        name: msg.toolName || "tool",
        arguments: "",
        status: msg.toolStatus === "error" ? "error" : "done",
        result: normalizeToolResult(msg.toolName, msg.content),
      });
      registerTool(toolCallId, idx);
      continue;
    }

    if (msg.parentToolCallId) {
      appendToParentChildren(msg.parentToolCallId, (children) => {
        if (msg.reasoningContent) {
          pushReasoningToSubTimeline(
            children,
            msg.reasoningContent,
            msg.reasoningDurationMs
          );
        } else if (msg.reasoningDurationMs !== undefined) {
          updateLastSubTimelineReasoningDuration(
            children,
            msg.reasoningDurationMs
          );
        }

        if (msg.content) {
          if (msg.reasoningDurationMs !== undefined) {
            updateLastSubTimelineReasoningDuration(
              children,
              msg.reasoningDurationMs
            );
          }
          pushContentToSubTimeline(children, msg.content);
        }
      });
      continue;
    }

    if (msg.reasoningContent) {
      pushReasoningToTimeline(
        timeline,
        msg.reasoningContent,
        msg.reasoningDurationMs
      );
    } else if (msg.reasoningDurationMs !== undefined) {
      updateLastTimelineReasoningDuration(timeline, msg.reasoningDurationMs);
    }

    if (msg.content) {
      if (msg.reasoningDurationMs !== undefined) {
        updateLastTimelineReasoningDuration(timeline, msg.reasoningDurationMs);
      }
      pushContentToTimeline(timeline, msg.content);
    }
  }

  return timeline;
}

export function resolveHistoryErrorMessage(
  conversation: AgentConversation,
  messages: AgentMessage[]
): string | undefined {
  const isErrorConversation =
    conversation.status === "error" || conversation.status === "failed";
  if (!isErrorConversation) {
    return undefined;
  }

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    const content = message.content?.trim();
    if (!content || message.role === "user") {
      continue;
    }
    return content;
  }

  return "任务执行失败";
}
