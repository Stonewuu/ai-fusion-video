import type { AiChatStreamEvent } from "@/lib/api/ai-pipeline";
import type { AiChatStreamEvent as GenericStreamEvent } from "@/lib/api/ai-assistant";
import {
  cancelCallingTimelineTools,
  finishedToolTimelineStatus,
  type SubTimelineItem,
  type TimelineItem,
  type ToolTimelineStatus,
} from "@/lib/store/pipeline-timeline";
import type {
  InvalidationType,
  PipelineTask,
  PipelineState,
  PipelineStoreState,
} from "@/lib/store/pipeline-store";

export interface PipelineEventLifecycle {
  notifySettlement: (
    id: string,
    status: "done" | "error" | "cancelled",
    onComplete?: () => void,
    onSettled?: (status: "done" | "error" | "cancelled") => void,
  ) => void;
  scheduleStatusSync: () => void;
}

export const TOOL_INVALIDATION_MAP: Record<string, InvalidationType> = {
  // assets 相关的工具
  create_asset: "assets",
  add_asset_item: "assets",
  update_asset: "assets",
  update_asset_item: "assets",
  batch_create_assets: "assets",
  batch_create_asset_items: "assets",
  delete_asset_resource: "assets",
  update_asset_image: "assets",
  generate_image: "assets",

  // scripts 相关的工具
  save_script_episode: "scripts",
  save_script_scene_items: "scripts",
  update_script: "scripts",
  update_script_info: "scripts",
  manage_script_scenes: "scripts",
  update_script_scene: "scripts",
  update_script_scene_item: "scripts",
  manage_script_scene_items: "scripts",
  delete_script_child_resource: "scripts",

  // storyboards 相关的工具
  save_storyboard_episode: "storyboards",
  save_storyboard_scene_shots: "storyboards",
  insert_storyboard_item: "storyboards",
  update_storyboard_item: "storyboards",
  update_storyboard_item_video: "storyboards",
  update_storyboard_item_frame: "storyboards",
  update_storyboard_scene: "storyboards",
  delete_storyboard_child_resource: "storyboards",
  generate_video: "storyboards",
  script_to_storyboard: "storyboards",
};

function isMainAgentTerminalEvent(event: GenericStreamEvent): boolean {
  return !event.parentToolCallId && !event.agentName && (
    event.outputType === "DONE" ||
    event.outputType === "ERROR" ||
    event.outputType === "CANCELLED"
  );
}

function isDurablePipelineEvent(
  event: GenericStreamEvent
): event is AiChatStreamEvent {
  return (
    event.schemaVersion === 1 &&
    typeof event.runId === "string" &&
    event.runId.length > 0 &&
    typeof event.sequence === "number" &&
    Number.isSafeInteger(event.sequence) &&
    event.sequence > 0
  );
}

let idCounter = 0;
export function generatePipelineTaskId(): string {
  return `pipeline-${Date.now()}-${++idCounter}`;
}

type ContentMergeMode = "stream" | "paragraph";

function mergeContentText(
  existingText: string,
  incomingText: string,
  mode: ContentMergeMode
): string {
  if (!existingText) return incomingText;
  if (!incomingText) return existingText;
  if (mode === "stream") {
    return existingText + incomingText;
  }
  if (existingText.endsWith("\n\n")) {
    return existingText + incomingText;
  }
  if (existingText.endsWith("\n")) {
    return `${existingText}\n${incomingText}`;
  }
  return `${existingText}\n\n${incomingText}`;
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

/** 在 timeline 中找到指定 tool ID 的节点，并向其 children 追加子事件。 */
function appendToToolChildren(
  timeline: TimelineItem[],
  parentToolCallId: string,
  updater: (children: SubTimelineItem[]) => SubTimelineItem[]
): TimelineItem[] {
  const found = timeline.some(
    (item) => item.type === "tool" && item.id === parentToolCallId
  );

  if (!found) {
    throw new Error(
      `Pipeline child event has no parent tool call: ${parentToolCallId}`,
    );
  }

  return timeline.map((item) => {
    if (item.type === "tool" && item.id === parentToolCallId) {
      return {
        ...item,
        children: updater(item.children ?? []),
      };
    }
    return item;
  });
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

/** 处理 SSE 事件的通用逻辑（支持子 Agent 嵌套） */
export function createPipelineEventHandler(
  lifecycle: PipelineEventLifecycle,
  id: string,
  set: (fn: (s: PipelineStoreState) => Partial<PipelineStoreState>) => void,
  onComplete?: () => void,
  onSettled?: (status: "done" | "error" | "cancelled") => void,
  contentMergeMode: ContentMergeMode = "stream",
  durableEvents = true
) {
  // 事件队列 + rAF 节流，避免高频 set() 导致 Maximum update depth exceeded
  const eventQueue: GenericStreamEvent[] = [];
  let rafScheduled = false;
  let acceptedRunId: string | undefined;
  let acceptedSequence = 0;
  let terminalNotified = false;

  function flushEvents() {
    rafScheduled = false;
    const batch = eventQueue.splice(0);
    if (batch.length === 0) return;

    // 收集本批次需要触发的 invalidation 类型
    const invalidations: InvalidationType[] = [];
    let hasTerminalEvent = false;
    for (const event of batch) {
      if (isMainAgentTerminalEvent(event)) {
        hasTerminalEvent = true;
      }
    }


    set((s) => {
      const tasks = s.tasks.map((t) => {
        if (t.id !== id) return t;

        const next: PipelineState = {
          ...t.state,
          timeline: [...t.state.timeline],
        };

        for (const event of batch) {
          if (durableEvents) {
            if (!isDurablePipelineEvent(event)) {
              throw new Error("Pipeline event has no durable identity");
            }
            if (next.runId && next.runId !== event.runId) {
              throw new Error("Pipeline event belongs to a different run");
            }
            if (event.sequence <= next.lastSequence) {
              continue;
            }
            next.runId = event.runId;
            next.lastSequence = event.sequence;
          }
          next.error = undefined;
          if (event.conversationId) {
            next.conversationId = event.conversationId;
          }

          const isSubAgent = !!event.parentToolCallId;
          const eventReasoningDurationMs = validDuration(
            event.reasoningDurationMs
          );

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
                const reasoningStartTime = validTimestamp(
                  event.reasoningStartTime
                );
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
              
              break;

            case "CONTENT":
              if (event.content) {
                const content = event.content;
                if (isSubAgent) {
                  next.timeline = appendToToolChildren(
                    next.timeline,
                    event.parentToolCallId!,
                    (children) => {
                      const updatedChildren = [...children];

                      const last =
                        updatedChildren[updatedChildren.length - 1];
                      if (last && last.type === "content") {
                        return [
                          ...updatedChildren.slice(0, -1),
                          {
                            ...last,
                            text: mergeContentText(
                              last.text,
                              content,
                              contentMergeMode
                            ),
                          },
                        ];
                      }
                      return [
                        ...updatedChildren,
                        { type: "content" as const, text: content },
                      ];
                    }
                  );
                } else {
                  const last = next.timeline[next.timeline.length - 1];
                  if (
                    last &&
                    last.type === "content" &&
                    !next.separateNextRootContent
                  ) {
                    next.timeline[next.timeline.length - 1] = {
                      ...last,
                      text: mergeContentText(
                        last.text,
                        content,
                        contentMergeMode
                      ),
                    };
                  } else {
                    next.timeline.push({
                      type: "content",
                      text: content,
                    });
                  }
                  next.separateNextRootContent = false;
                }
              }
              break;

            case "TOOL_CALL":
              if (event.toolCalls) {
                for (const tc of event.toolCalls) {
                  if (isSubAgent) {
                    next.timeline = appendToToolChildren(
                      next.timeline,
                      event.parentToolCallId!,
                      (children) => {
                        const exists = children.some(
                          (c) => c.type === "tool" && c.id === tc.id
                        );
                        if (exists) return children;
                        return [
                          ...children,
                          {
                            type: "tool" as const,
                            id: tc.id,
                            name: tc.name,
                            arguments: tc.arguments,
                            batchId: event.replyId,
                            status: "calling" as const,
                          },
                        ];
                      }
                    );
                  } else {
                    const exists = next.timeline.some(
                      (item) => item.type === "tool" && item.id === tc.id
                    );
                    if (!exists) {
                      next.timeline.push({
                        type: "tool",
                        id: tc.id,
                        name: tc.name,
                        arguments: tc.arguments,
                        batchId: event.replyId,
                        status: "calling",
                        agentName: event.agentName,
                      });
                    }
                  }
                }
              }
              break;

            case "TOOL_FINISHED":
              if (event.toolCallId) {
                const toolStatus = finishedToolTimelineStatus(event.toolStatus);

                if (isSubAgent) {
                  next.timeline = appendToToolChildren(
                    next.timeline,
                    event.parentToolCallId!,
                    (children) =>
                      children.map((c) =>
                        c.type === "tool" && c.id === event.toolCallId
                          ? {
                              ...c,
                              status: toolStatus,
                              result: event.toolResult,
                            }
                          : c
                      )
                  );
                } else {
                  const exists = next.timeline.some(
                    (item) =>
                      item.type === "tool" && item.id === event.toolCallId
                  );
                  if (exists) {
                    next.timeline = next.timeline.map((item) =>
                      item.type === "tool" && item.id === event.toolCallId
                        ? {
                            ...item,
                            status: toolStatus,
                            result: event.toolResult,
                            // 补充工具名（占位节点可能为 unknown_sub_agent）
                            ...(event.toolName ? { name: event.toolName } : {}),
                          }
                        : item
                    );
                  } else if (event.toolName) {
                    // 容错：TOOL_CALL 已被裁剪，补创建已完成工具节点
                    next.timeline.push({
                      type: "tool",
                      id: event.toolCallId,
                      name: event.toolName,
                      arguments: "",
                      status: toolStatus,
                      result: event.toolResult,
                    });
                  }
                }
              }
              // 收集 invalidation
              if (
                event.toolName &&
                event.toolStatus !== "error" &&
                TOOL_INVALIDATION_MAP[event.toolName]
              ) {
                invalidations.push(TOOL_INVALIDATION_MAP[event.toolName]);
              }
              break;

            case "SUB_AGENT_FINISHED":
              if (isSubAgent) {
                next.timeline = updateToolStatus(
                  next.timeline,
                  event.parentToolCallId!,
                  "done"
                );
              }
              break;

            case "DONE":
              if (!isMainAgentTerminalEvent(event)) break;
              next.status = "done";
              next.cancelRequested = false;
              if (event.content) {
                const last = next.timeline[next.timeline.length - 1];
                if (
                  last &&
                  last.type === "content" &&
                  !next.separateNextRootContent
                ) {
                  next.timeline[next.timeline.length - 1] = {
                    ...last,
                    text: mergeContentText(
                      last.text,
                      event.content,
                      contentMergeMode
                    ),
                  };
                } else {
                  next.timeline.push({ type: "content", text: event.content });
                }
                next.separateNextRootContent = false;
              }
              break;

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
                      type: "content" as const,
                        text: `${event.agentName || "子Agent"} 出错：${event.error || "未知错误"}`,
                    },
                  ]
                );
              } else if (event.agentName) {
                next.timeline.push({
                  type: "content",
                  text: `${event.agentName} 出错：${event.error || "未知错误"}`,
                });
              } else {
                next.status = "error";
                next.cancelRequested = false;
                next.error = event.error || "未知错误";
              }
              break;

            case "CANCELLED":
              if (isMainAgentTerminalEvent(event)) {
                next.status = "cancelled";
                next.cancelRequested = false;
                next.timeline = cancelCallingTimelineTools(next.timeline);
              }
              break;
          }
        } // end for batch

        if (next.cancelRequested) {
          next.timeline = cancelCallingTimelineTools(next.timeline);
        }

        const newStatus: PipelineTask["status"] =
          next.status === "done"
            ? "done"
            : next.status === "error"
              ? "error"
              : next.status === "cancelled"
                ? "cancelled"
                : "running";

        const isFinished = newStatus !== "running" && t.status === "running";
        return {
          ...t,
          status: newStatus,
          state: next,
          ...(isFinished ? { finishedAt: Date.now() } : {}),
        };
      });

      // 合并 invalidation 到同一次 set
      const newInvalidation = { ...s.invalidation };
      if (hasTerminalEvent) {
        newInvalidation.assets = (newInvalidation.assets || 0) + 1;
        newInvalidation.scripts = (newInvalidation.scripts || 0) + 1;
        newInvalidation.storyboards = (newInvalidation.storyboards || 0) + 1;
      }
      for (const inv of invalidations) {
        newInvalidation[inv] = (newInvalidation[inv] || 0) + 1;
      }

      return { tasks, invalidation: newInvalidation };
    });

    // 完成/错误/取消时触发后续回调
    for (const event of batch) {
      if (terminalNotified || !isMainAgentTerminalEvent(event)) continue;
      if (event.outputType === "DONE" && isMainAgentTerminalEvent(event)) {
        terminalNotified = true;
        lifecycle.notifySettlement(id, "done", onComplete, onSettled);
      }
      if (
        (event.outputType === "ERROR" || event.outputType === "CANCELLED") &&
        isMainAgentTerminalEvent(event)
      ) {
        terminalNotified = true;
        lifecycle.notifySettlement(
          id,
          event.outputType === "CANCELLED" ? "cancelled" : "error",
          onComplete,
          onSettled,
        );
      }
    }
    lifecycle.scheduleStatusSync();
  }

  return (event: GenericStreamEvent) => {
    if (durableEvents) {
      if (!isDurablePipelineEvent(event)) {
        throw new Error("Pipeline event has no durable identity");
      }
      if (acceptedRunId && event.runId !== acceptedRunId) {
        throw new Error("Pipeline stream switched to a different run");
      }
      if (event.sequence <= acceptedSequence) return;
      acceptedRunId = event.runId;
      acceptedSequence = event.sequence;
    }
    eventQueue.push(event);
    // A hidden browser tab suspends requestAnimationFrame. Commit the entire
    // pending batch as soon as the durable terminal event arrives so task
    // metadata can never become terminal while its timeline is still stale.
    if (isMainAgentTerminalEvent(event)) {
      flushEvents();
      return;
    }
    if (!rafScheduled) {
      rafScheduled = true;
      if (typeof requestAnimationFrame !== "undefined") {
        requestAnimationFrame(flushEvents);
      } else {
        setTimeout(flushEvents, 16);
      }
    }
  };
}
