"use client";

import { create } from "zustand";
import {
  pipelineStream,
  reconnectPipelineStream,
  cancelPipeline,
  getPipelineStatus,
  listRunningPipelines,
  type AiChatReq,
  type AiChatStreamEvent,
  type PipelineRunStatus,
} from "@/lib/api/ai-pipeline";
import type { AiChatStreamEvent as GenericStreamEvent } from "@/lib/api/ai-assistant";
import {
  reconnectTaskStream,
  getTaskStreamStatus,
} from "@/lib/api/task-stream";

// ========== 数据失效映射 ==========

/** 数据失效类型 */
export type InvalidationType = "assets" | "scripts" | "storyboards";

/** 工具名 → 影响的数据类型 */
const TOOL_INVALIDATION_MAP: Record<string, InvalidationType> = {
  // assets 相关的工具
  create_asset: "assets",
  add_asset_item: "assets",
  update_asset: "assets",
  batch_create_assets: "assets",
  batch_create_asset_items: "assets",
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

  // storyboards 相关的工具
  save_storyboard_episode: "storyboards",
  save_storyboard_scene_shots: "storyboards",
  insert_storyboard_item: "storyboards",
  update_storyboard_item_video: "storyboards",
  update_storyboard_item_frame: "storyboards",
  generate_video: "storyboards",
};

// ========== 类型 ==========

/** 子 Agent 时间线中的元素 */
export type SubTimelineItem =
  | {
      type: "tool";
      id: string;
      name: string;
      arguments: string;
      status: "calling" | "done" | "error";
      result?: string;
    }
  | { type: "content"; text: string }
  | { type: "reasoning"; text: string; durationMs?: number };

/** 时间线中的每个元素（与 agent-pipeline.tsx 一致） */
export type TimelineItem =
  | {
      type: "tool";
      id: string;
      name: string;
      arguments: string;
      status: "calling" | "done" | "error";
      result?: string;
      /** 如果此工具是子 Agent 调用，agentName 标识来源 */
      agentName?: string;
      /** 子 Agent 的嵌套时间线（推理、内容、工具调用） */
      children?: SubTimelineItem[];
    }
  | { type: "reasoning"; text: string; durationMs?: number }
  | { type: "content"; text: string };

export interface PipelineState {
  status: "running" | "done" | "error" | "cancelled";
  reasoningText: string;
  reasoningDurationMs?: number;
  timeline: TimelineItem[];
  runId?: string;
  lastSequence: number;
  conversationId?: string;
  error?: string;
  /** 页面恢复的 Pipeline 在打开详情前不回放 journal 内容。 */
  contentLoaded?: boolean;
  contentLoading?: boolean;
  /** 仅在详情选中时建立前端 reconnect SSE，不影响后端 Pipeline。 */
  reconnectOnSelect?: boolean;
  /** 页面列表/轮询得到的后端 run 状态。 */
  runStatus?: PipelineRunStatus;
}

export interface PipelineTask {
  id: string;
  label: string;
  projectId: number;
  status: "running" | "done" | "error" | "cancelled";
  state: PipelineState;
  createdAt: number;
  cancellable?: boolean;
  /** 任务结束时间（done/error/cancelled 时记录） */
  finishedAt?: number;
}

// ========== Store ==========

interface PipelineStoreState {
  tasks: PipelineTask[];
  notificationOpen: boolean;
  /** 是否显示大面板（任务中心） */
  panelExpanded: boolean;
  /** 当前展开详情的 pipeline id */
  expandedTaskId: string | null;
  /** 是否已恢复过运行中 Pipeline 列表 */
  runningPipelinesRestored: boolean;
  /** 数据失效计数器 —— 页面监听对应 key 触发刷新 */
  invalidation: Record<InvalidationType, number>;

  // actions
  addPipeline: (config: {
    label: string;
    projectId: number;
    request: AiChatReq;
    onComplete?: () => void;
  }) => string;
  attachTaskStream: (config: {
    label: string;
    projectId: number;
    taskId: string;
    cancellable?: boolean;
    onComplete?: () => void;
    onSettled?: (status: "done" | "error" | "cancelled") => void;
  }) => string;
  /**
   * 添加一个非 AI 任务（如视频合成），不走 SSE 流。
   * 调用方负责后续通过 markSimpleTask 推进状态。
   */
  addSimpleTask: (config: {
    label: string;
    projectId: number;
    initialNote?: string;
    onComplete?: () => void;
  }) => string;
  /** 推进非 AI 任务的状态（追加结果文本/错误，标记完成或失败） */
  markSimpleTask: (
    id: string,
    update: {
      status: "done" | "error";
      resultText?: string;
      errorText?: string;
    }
  ) => void;
  cancelPipeline: (id: string) => Promise<void>;
  removePipeline: (id: string) => void;
  clearCompleted: () => void;
  setNotificationOpen: (open: boolean) => void;
  setPanelExpanded: (expanded: boolean) => void;
  setExpandedTaskId: (id: string | null) => void;
  /** 首次打开恢复任务详情时，从 journal 起点加载完整内容。 */
  loadPipelineContent: (id: string) => void;
  /** 仅终止前端内容流；不会取消后端 Pipeline。 */
  disconnectPipelineContent: (id: string) => void;
  /** 页面加载时调用：查询 running 对话并轮询状态，不建立 SSE。 */
  restoreRunningPipelines: () => void;
}

// 简单任务的完成回调（不放在 zustand state 里避免序列化问题）
const simpleTaskCallbacks = new Map<string, () => void>();

// 存储 AbortController 的 map（不放在 zustand state 里避免序列化问题）
const abortControllers = new Map<string, AbortController>();
const PIPELINE_STATUS_POLL_INTERVAL_MS = 3000;
let pipelineStatusPollTimer: ReturnType<typeof setTimeout> | null = null;

function toTerminalTaskStatus(
  status: PipelineRunStatus
): PipelineTask["status"] | null {
  if (status === "COMPLETED") return "done";
  if (status === "FAILED") return "error";
  if (status === "CANCELLED") return "cancelled";
  return null;
}

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
function generateId(): string {
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

/**
 * 获取当前运行中 pipeline 的 conversationId 集合
 * 供 notification-panel 过滤历史列表使用
 */
export function getRunningConversationIds(): Set<string> {
  const tasks = usePipelineStore.getState().tasks;
  const ids = new Set<string>();
  for (const t of tasks) {
    if (t.status === "running" && t.state.conversationId) {
      ids.add(t.state.conversationId);
    }
  }
  return ids;
}

/**
 * 在 timeline 中找到指定 tool ID 的节点，并向其 children 追加子事件。
 * 如果找不到（重连场景下 TOOL_CALL 可能被 Redis Stream 裁剪），
 * 自动创建一个占位工具节点来承载后续子事件。
 */
function appendToToolChildren(
  timeline: TimelineItem[],
  parentToolCallId: string,
  updater: (children: SubTimelineItem[]) => SubTimelineItem[]
): TimelineItem[] {
  const found = timeline.some(
    (item) => item.type === "tool" && item.id === parentToolCallId
  );

  if (!found) {
    // 容错：创建占位父工具节点（TOOL_CALL 事件已被裁剪）
    const placeholder: TimelineItem = {
      type: "tool",
      id: parentToolCallId,
      name: "unknown_sub_agent",
      arguments: "",
      status: "calling",
      children: updater([]),
    };
    return [...timeline, placeholder];
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
  status: "calling" | "done" | "error"
): TimelineItem[] {
  return timeline.map((item) =>
    item.type === "tool" && item.id === toolCallId
      ? { ...item, status }
      : item
  );
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
  return [
    ...children,
    {
      type: "reasoning",
      text: reasoningContent,
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
  reasoningContent: string
): TimelineItem[] {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...timeline.slice(0, -1),
      { ...last, text: last.text + reasoningContent },
    ];
  }
  return [
    ...timeline,
    {
      type: "reasoning",
      text: reasoningContent,
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
function createEventHandler(
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

          switch (event.outputType) {
            case "REASONING":
              if (event.reasoningContent) {
                if (isSubAgent) {
                  next.timeline = appendToToolChildren(
                    next.timeline,
                    event.parentToolCallId!,
                    (children) =>
                      appendReasoningToSubTimeline(
                        children,
                        event.reasoningContent!
                      )
                  );
                } else {
                  next.reasoningText += event.reasoningContent;
                  next.timeline = appendReasoningToTimeline(
                    next.timeline,
                    event.reasoningContent
                  );
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

            case "CONTENT":
              if (event.reasoningDurationMs && !isSubAgent) {
                next.reasoningDurationMs = event.reasoningDurationMs;
                next.timeline = updateLastTimelineReasoningDuration(
                  next.timeline,
                  event.reasoningDurationMs
                );
              }
              if (event.content) {
                const content = event.content;
                if (isSubAgent) {
                  next.timeline = appendToToolChildren(
                    next.timeline,
                    event.parentToolCallId!,
                    (children) => {
                      let updatedChildren = [...children];
                      if (event.reasoningDurationMs) {
                        updatedChildren = updateLastSubTimelineReasoningDuration(
                          updatedChildren,
                          event.reasoningDurationMs
                        );
                      }

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
                  if (last && last.type === "content") {
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
                const toolStatus =
                  event.toolStatus === "error"
                    ? ("error" as const)
                    : ("done" as const);

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
              if (event.content) {
                const last = next.timeline[next.timeline.length - 1];
                if (last && last.type === "content") {
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
                next.error = event.error || "未知错误";
              }
              break;

            case "CANCELLED":
              if (isMainAgentTerminalEvent(event)) {
                next.status = "cancelled";
              }
              break;
          }
        } // end for batch

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
        abortControllers.delete(id);
        onComplete?.();
        onSettled?.("done");
      }
      if (
        (event.outputType === "ERROR" || event.outputType === "CANCELLED") &&
        isMainAgentTerminalEvent(event)
      ) {
        terminalNotified = true;
        abortControllers.delete(id);
        onSettled?.(
          event.outputType === "CANCELLED" ? "cancelled" : "error"
        );
      }
    }
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

function settleTaskIfRunning(
  set: (fn: (s: PipelineStoreState) => Partial<PipelineStoreState>) => void,
  id: string,
  status: "done" | "error" | "cancelled",
  options?: {
    error?: string;
    onComplete?: () => void;
    onSettled?: (status: "done" | "error" | "cancelled") => void;
  }
) {
  let transitioned = false;
  set((s) => {
    const nextTasks = s.tasks.map((t) => {
      if (t.id !== id || t.status !== "running") {
        return t;
      }
      transitioned = true;
      return {
        ...t,
        status,
        finishedAt: Date.now(),
        state: {
          ...t.state,
          status,
          ...(status === "error"
            ? { error: options?.error || t.state.error || "任务失败" }
            : {}),
        },
      };
    });

    if (transitioned) {
      return {
        tasks: nextTasks,
        invalidation: {
          assets: (s.invalidation.assets || 0) + 1,
          scripts: (s.invalidation.scripts || 0) + 1,
          storyboards: (s.invalidation.storyboards || 0) + 1,
        },
      };
    }
    return { tasks: nextTasks };
  });
  abortControllers.delete(id);
  if (transitioned) {
    if (status === "done") {
      options?.onComplete?.();
    }
    options?.onSettled?.(status);
  }
}

export const usePipelineStore = create<PipelineStoreState>()((set, get) => ({
  tasks: [],
  notificationOpen: false,
  panelExpanded: false,
  expandedTaskId: null,
  runningPipelinesRestored: false,
  invalidation: { assets: 0, scripts: 0, storyboards: 0 },

  addSimpleTask: ({ label, projectId, initialNote, onComplete }) => {
    const id = generateId();
    const initialState: PipelineState = {
      status: "running",
      reasoningText: "",
      timeline: initialNote ? [{ type: "content", text: initialNote }] : [],
      lastSequence: 0,
    };
    const task: PipelineTask = {
      id,
      label,
      projectId,
      status: "running",
      state: initialState,
      createdAt: Date.now(),
      cancellable: false,
    };
    set((s) => ({ tasks: [...s.tasks, task] }));
    if (onComplete) {
      simpleTaskCallbacks.set(id, onComplete);
    }
    return id;
  },

  markSimpleTask: (id, update) => {
    let finished = false;
    set((s) => {
      const nextTasks = s.tasks.map((t) => {
        if (t.id !== id) return t;
        const isFinishingFromRunning = t.status === "running";
        if (isFinishingFromRunning) finished = true;
        const newTimeline = [...t.state.timeline];
        if (update.resultText) {
          newTimeline.push({ type: "content", text: update.resultText });
        }
        return {
          ...t,
          status: update.status,
          state: {
            ...t.state,
            status: update.status,
            timeline: newTimeline,
            error: update.errorText,
          },
          ...(isFinishingFromRunning ? { finishedAt: Date.now() } : {}),
        };
      });

      if (finished) {
        return {
          tasks: nextTasks,
          invalidation: {
            assets: (s.invalidation.assets || 0) + 1,
            scripts: (s.invalidation.scripts || 0) + 1,
            storyboards: (s.invalidation.storyboards || 0) + 1,
          },
        };
      }
      return { tasks: nextTasks };
    });
    if (update.status === "done") {
      const cb = simpleTaskCallbacks.get(id);
      if (cb) {
        try {
          cb();
        } catch (e) {
          console.error("[simpleTask] onComplete error", e);
        }
        simpleTaskCallbacks.delete(id);
      }
    } else {
      simpleTaskCallbacks.delete(id);
    }
  },

  addPipeline: ({ label, projectId, request, onComplete }) => {
    const id = generateId();
    const initialState: PipelineState = {
      status: "running",
      reasoningText: "",
      timeline: [],
      lastSequence: 0,
    };

    const task: PipelineTask = {
      id,
      label,
      projectId,
      status: "running",
      state: initialState,
      createdAt: Date.now(),
      cancellable: true,
    };

    set((s) => ({ tasks: [...s.tasks, task] }));

    const handleEvent = createEventHandler(id, set, onComplete);

    // 启动 SSE 流
    const controller = pipelineStream(request, {
      onEvent: handleEvent,
      onError: (err) => {
        // 传输失败不是 Agent 终态；保留 cursor 供 durable reconnect。
        set((s) => ({
          tasks: s.tasks.map((t) =>
            t.id === id && t.status === "running"
              ? {
                  ...t,
                  state: {
                    ...t.state,
                    error: `Pipeline 连接中断：${err.message}`,
                  },
                }
              : t
          ),
        }));
        abortControllers.delete(id);
      },
      onComplete: () => {
        // 业务终态已由 journal terminal event 在 handleEvent 中处理。
        abortControllers.delete(id);
      },
    });

    abortControllers.set(id, controller);

    return id;
  },

  attachTaskStream: ({
    label,
    projectId,
    taskId,
    cancellable = false,
    onComplete,
    onSettled,
  }) => {
    const id = generateId();
    const task: PipelineTask = {
      id,
      label,
      projectId,
      status: "running",
      state: {
        status: "running",
        reasoningText: "",
        timeline: [{ type: "content", text: "正在连接任务流……" }],
        lastSequence: 0,
        conversationId: taskId,
      },
      createdAt: Date.now(),
      cancellable,
    };

    set((s) => ({ tasks: [...s.tasks, task] }));

    void (async () => {
      try {
        const streamStatus = await getTaskStreamStatus(taskId);
        if (streamStatus === "NONE") {
          settleTaskIfRunning(set, id, "error", {
            error: "任务流不存在",
            onSettled,
          });
          return;
        }

        const handleEvent = createEventHandler(
          id,
          set,
          onComplete,
          onSettled,
          "paragraph",
          false
        );

        set((s) => ({
          tasks: s.tasks.map((t) =>
            t.id === id
              ? { ...t, state: { ...t.state, timeline: [] } }
              : t
          ),
        }));

        const controller = reconnectTaskStream(taskId, {
          onEvent: handleEvent,
          onError: (err) => {
            settleTaskIfRunning(set, id, "error", {
              error: err.message,
              onSettled,
            });
          },
          onComplete: () => {
            if (streamStatus === "ERROR") {
              settleTaskIfRunning(set, id, "error", { onSettled });
            } else if (streamStatus === "COMPLETED") {
              settleTaskIfRunning(set, id, "done", {
                onComplete,
                onSettled,
              });
            } else {
              settleTaskIfRunning(set, id, "error", {
                error: "任务流在未提供终态事件时结束",
                onSettled,
              });
            }
          },
        });

        abortControllers.set(id, controller);
      } catch (err) {
        settleTaskIfRunning(set, id, "error", {
          error: err instanceof Error ? err.message : "连接任务流失败",
          onSettled,
        });
      }
    })();

    return id;
  },

  cancelPipeline: async (id: string) => {
    const task = get().tasks.find((t) => t.id === id);
    try {
      if (!task?.state.runId) {
        throw new Error("Pipeline 尚未返回 runId，无法提交取消请求");
      }
      await cancelPipeline({ runId: task.state.runId });
      // 保持 SSE 连接，等待服务端持久化并发送 CANCELLED terminal event。
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      set((s) => ({
        tasks: s.tasks.map((item) =>
          item.id === id
            ? {
                ...item,
                state: { ...item.state, error: `取消请求失败：${message}` },
              }
            : item
        ),
      }));
      throw error;
    }
  },

  removePipeline: (id: string) => {
    abortControllers.get(id)?.abort();
    abortControllers.delete(id);
    simpleTaskCallbacks.delete(id);
    set((s) => ({
      tasks: s.tasks.filter((t) => t.id !== id),
      expandedTaskId: s.expandedTaskId === id ? null : s.expandedTaskId,
    }));
  },

  clearCompleted: () => {
    const removableIds = get().tasks
      .filter((t) => t.status !== "running")
      .map((t) => t.id);
    for (const id of removableIds) {
      abortControllers.delete(id);
      simpleTaskCallbacks.delete(id);
    }
    set((s) => ({
      tasks: s.tasks.filter((t) => t.status === "running"),
      expandedTaskId:
        s.expandedTaskId &&
        s.tasks.find((t) => t.id === s.expandedTaskId)?.status === "running"
          ? s.expandedTaskId
          : null,
    }));
  },

  setNotificationOpen: (open: boolean) => {
    set({ notificationOpen: open });
    // 关闭通知时同时关闭大面板
    if (!open) set({ panelExpanded: false });
  },

  setPanelExpanded: (expanded: boolean) => {
    set({ panelExpanded: expanded });
    // 打开大面板时确保 notificationOpen 为 true
    if (expanded) set({ notificationOpen: true });
  },

  setExpandedTaskId: (id: string | null) => {
    set({ expandedTaskId: id });
  },

  loadPipelineContent: (id: string) => {
    const task = get().tasks.find((candidate) => candidate.id === id);
    if (
      !task?.state.runId ||
      !task.state.reconnectOnSelect ||
      task.status !== "running" ||
      task.state.contentLoading
    ) {
      return;
    }

    abortControllers.get(id)?.abort();
    abortControllers.delete(id);
    const afterSequence = task.state.lastSequence;
    set((s) => ({
      tasks: s.tasks.map((candidate) =>
        candidate.id === id
          ? {
              ...candidate,
              state: {
                ...candidate.state,
                error: undefined,
                contentLoading: true,
                ...(afterSequence === 0
                  ? {
                      reasoningText: "",
                      reasoningDurationMs: undefined,
                      timeline: [],
                    }
                  : {}),
              },
            }
          : candidate
      ),
    }));

    const handleEvent = createEventHandler(id, set);
    const controller = reconnectPipelineStream(
      task.state.runId,
      afterSequence,
      {
        onEvent: handleEvent,
        onError: (error) => {
          set((s) => ({
            tasks: s.tasks.map((candidate) =>
              candidate.id === id
                ? {
                    ...candidate,
                    state: {
                      ...candidate.state,
                      contentLoading: false,
                      error: `Pipeline 内容加载中断：${error.message}`,
                    },
                  }
                : candidate
            ),
          }));
          abortControllers.delete(id);
        },
        onComplete: () => {
          set((s) => ({
            tasks: s.tasks.map((candidate) =>
              candidate.id === id
                ? {
                    ...candidate,
                    state: {
                      ...candidate.state,
                      contentLoaded: true,
                      contentLoading: false,
                    },
                  }
                : candidate
            ),
          }));
          abortControllers.delete(id);
        },
      }
    );
    abortControllers.set(id, controller);
  },

  disconnectPipelineContent: (id: string) => {
    const task = get().tasks.find((candidate) => candidate.id === id);
    if (!task?.state.reconnectOnSelect) return;

    // 仅关闭当前浏览器的 SSE；不调用 cancelPipeline，不影响服务端 run。
    abortControllers.get(id)?.abort();
    abortControllers.delete(id);
    set((s) => ({
      tasks: s.tasks.map((candidate) =>
        candidate.id === id
          ? {
              ...candidate,
              state: { ...candidate.state, contentLoading: false },
            }
          : candidate
      ),
    }));
  },

  /** 页面加载时调用：只查询 durable root run 的元数据和状态，不建立 SSE。 */
  restoreRunningPipelines: () => {
    if (get().runningPipelinesRestored) return;
    set({ runningPipelinesRestored: true });

    void (async () => {
      try {
        const runningRuns = await listRunningPipelines();
        for (const run of runningRuns) {
          const id = `reconnect-${run.runId}`;
          const existing = get().tasks.find(
            (task) =>
              task.state.runId === run.runId ||
              task.state.conversationId === run.conversationId
          );
          if (existing) continue;

          const placeholder: PipelineTask = {
            id,
            label: run.title || "AI 任务",
            projectId: run.projectId,
            status: "running",
            state: {
              status: "running",
              reasoningText: "",
              timeline: [],
              runId: run.runId,
              lastSequence: 0,
              conversationId: run.conversationId,
              contentLoaded: false,
              contentLoading: false,
              reconnectOnSelect: true,
              runStatus: run.status,
            },
            createdAt: new Date(run.startedAt).getTime(),
            cancellable: true,
          };
          set((s) => ({ tasks: [...s.tasks, placeholder] }));
        }

        const pollStatuses = async () => {
          pipelineStatusPollTimer = null;
          const candidates = get().tasks.filter(
            (task) =>
              task.state.reconnectOnSelect &&
              task.status === "running" &&
              !task.state.contentLoading &&
              task.state.runId
          );

          await Promise.all(
            candidates.map(async (task) => {
              try {
                const status = await getPipelineStatus({
                  runId: task.state.runId!,
                });
                if (status.runId !== task.state.runId) {
                  throw new Error("Pipeline status returned a different runId");
                }

                const terminalStatus = toTerminalTaskStatus(status.status);
                set((s) => {
                  const current = s.tasks.find(
                    (candidate) => candidate.id === task.id
                  );
                  if (
                    !current ||
                    current.state.contentLoading ||
                    current.status !== "running"
                  ) {
                    return {};
                  }

                  if (terminalStatus && !s.panelExpanded) {
                    return {
                      tasks: s.tasks.filter(
                        (candidate) => candidate.id !== task.id
                      ),
                    };
                  }

                  const tasks = s.tasks.map((candidate) =>
                    candidate.id === task.id
                      ? {
                          ...candidate,
                          status: terminalStatus ?? candidate.status,
                          ...(terminalStatus ? { finishedAt: Date.now() } : {}),
                          state: {
                            ...candidate.state,
                            runStatus: status.status,
                            status: terminalStatus ?? candidate.state.status,
                            ...(terminalStatus === "error"
                              ? {
                                  error:
                                    status.terminalEvent?.error || "任务失败",
                                }
                              : {}),
                          },
                        }
                      : candidate
                  );

                  return terminalStatus
                    ? {
                        tasks,
                        invalidation: {
                          assets: (s.invalidation.assets || 0) + 1,
                          scripts: (s.invalidation.scripts || 0) + 1,
                          storyboards: (s.invalidation.storyboards || 0) + 1,
                        },
                      }
                    : { tasks };
                });
              } catch (error) {
                console.error(
                  `[Pipeline] 查询 ${task.state.runId} 状态失败:`,
                  error
                );
              }
            })
          );

          const hasTrackedRuns = get().tasks.some(
            (task) =>
              task.state.reconnectOnSelect && task.status === "running"
          );
          if (hasTrackedRuns) {
            pipelineStatusPollTimer = setTimeout(
              () => void pollStatuses(),
              PIPELINE_STATUS_POLL_INTERVAL_MS
            );
          }
        };

        if (
          get().tasks.some(
            (task) =>
              task.state.reconnectOnSelect && task.status === "running"
          )
        ) {
          if (pipelineStatusPollTimer) {
            clearTimeout(pipelineStatusPollTimer);
          }
          pipelineStatusPollTimer = setTimeout(
            () => void pollStatuses(),
            PIPELINE_STATUS_POLL_INTERVAL_MS
          );
        }
      } catch (err) {
        console.error("[Pipeline] 查询 durable runs 失败:", err);
      }
    })();
  },
}));
