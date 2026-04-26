"use client";

import { create } from "zustand";
import {
  pipelineStream,
  cancelPipeline,
  reconnectPipelineStream,
  getPipelineStatus,
  listRunningPipelines,
  type AiChatReq,
  type AiChatStreamEvent,
} from "@/lib/api/ai-pipeline";

// ========== 数据失效映射 ==========

/** 数据失效类型 */
export type InvalidationType = "assets" | "scripts" | "storyboards";

/** 工具名 → 影响的数据类型 */
const TOOL_INVALIDATION_MAP: Record<string, InvalidationType> = {
  batch_create_assets: "assets",
  batch_create_asset_items: "assets",
  update_asset_image: "assets",
  save_script_episode: "scripts",
  save_script_scene_items: "scripts",
  update_script: "scripts",
  update_script_info: "scripts",
  manage_script_scenes: "scripts",
  update_script_scene: "scripts",
  save_storyboard_episode: "storyboards",
  save_storyboard_scene_shots: "storyboards",
  insert_storyboard_item: "storyboards",
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
  conversationId?: string;
  error?: string;
}

export interface PipelineTask {
  id: string;
  label: string;
  projectId: number;
  status: "running" | "done" | "error" | "cancelled";
  state: PipelineState;
  createdAt: number;
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
  /** 是否已执行过自动重连 */
  reconnected: boolean;
  /** 数据失效计数器 —— 页面监听对应 key 触发刷新 */
  invalidation: Record<InvalidationType, number>;

  // actions
  addPipeline: (config: {
    label: string;
    projectId: number;
    request: AiChatReq;
    onComplete?: () => void;
  }) => string;
  cancelPipeline: (id: string) => void;
  removePipeline: (id: string) => void;
  clearCompleted: () => void;
  setNotificationOpen: (open: boolean) => void;
  setPanelExpanded: (expanded: boolean) => void;
  setExpandedTaskId: (id: string | null) => void;
  /** 页面加载时调用：查询后端 running 对话并尝试 SSE 重连 */
  tryReconnect: () => void;
}

// 存储 AbortController 的 map（不放在 zustand state 里避免序列化问题）
const abortControllers = new Map<string, AbortController>();

function isMainAgentTerminalEvent(event: AiChatStreamEvent): boolean {
  return !event.parentToolCallId && !event.agentName && (
    event.outputType === "DONE" ||
    event.outputType === "ERROR" ||
    event.outputType === "CANCELLED"
  );
}

let idCounter = 0;
function generateId(): string {
  return `pipeline-${Date.now()}-${++idCounter}`;
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
  onComplete?: () => void
) {
  // 事件队列 + rAF 节流，避免高频 set() 导致 Maximum update depth exceeded
  const eventQueue: AiChatStreamEvent[] = [];
  let rafScheduled = false;

  function flushEvents() {
    rafScheduled = false;
    const batch = eventQueue.splice(0);
    if (batch.length === 0) return;

    // 收集本批次需要触发的 invalidation 类型
    const invalidations: InvalidationType[] = [];

    set((s) => {
      const tasks = s.tasks.map((t) => {
        if (t.id !== id) return t;

        const next: PipelineState = {
          ...t.state,
          timeline: [...t.state.timeline],
        };

        for (const event of batch) {
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
                          { ...last, text: last.text + event.content },
                        ];
                      }
                      return [
                        ...updatedChildren,
                        { type: "content" as const, text: event.content! },
                      ];
                    }
                  );
                } else {
                  const last = next.timeline[next.timeline.length - 1];
                  if (last && last.type === "content") {
                    next.timeline[next.timeline.length - 1] = {
                      ...last,
                      text: last.text + event.content,
                    };
                  } else {
                    next.timeline.push({
                      type: "content",
                      text: event.content,
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
              // 收集 invalidation
              if (
                event.toolName &&
                event.toolStatus !== "error" &&
                TOOL_INVALIDATION_MAP[event.toolName]
              ) {
                invalidations.push(TOOL_INVALIDATION_MAP[event.toolName]);
              }
              break;

            case "DONE":
              next.status = "done";
              if (event.content) {
                const last = next.timeline[next.timeline.length - 1];
                if (last && last.type === "content") {
                  next.timeline[next.timeline.length - 1] = {
                    ...last,
                    text: last.text + event.content,
                  };
                } else {
                  next.timeline.push({ type: "content", text: event.content });
                }
              }
              break;

            case "ERROR":
              if (isSubAgent) {
                next.timeline = appendToToolChildren(
                  next.timeline,
                  event.parentToolCallId!,
                  (children) => [
                    ...children,
                    {
                      type: "content" as const,
                      text: `❌ ${event.agentName || "子Agent"} 出错: ${event.error || "未知错误"}`,
                    },
                  ]
                );
              } else if (event.agentName) {
                next.timeline.push({
                  type: "content",
                  text: `❌ ${event.agentName} 出错: ${event.error || "未知错误"}`,
                });
              } else {
                next.status = "error";
                next.error = event.error || "未知错误";
              }
              break;

            case "CANCELLED":
              next.status = "cancelled";
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
      const newInvalidation = invalidations.length > 0
        ? { ...s.invalidation }
        : s.invalidation;
      for (const inv of invalidations) {
        newInvalidation[inv] = (newInvalidation[inv] || 0) + 1;
      }

      return { tasks, invalidation: newInvalidation };
    });

    // 完成/错误/取消时触发后续回调
    for (const event of batch) {
      if (event.outputType === "DONE" && isMainAgentTerminalEvent(event)) {
        abortControllers.delete(id);
        onComplete?.();
      }
      if (
        (event.outputType === "ERROR" || event.outputType === "CANCELLED") &&
        isMainAgentTerminalEvent(event)
      ) {
        abortControllers.delete(id);
      }
    }
  }

  return (event: AiChatStreamEvent) => {
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

export const usePipelineStore = create<PipelineStoreState>()((set, get) => ({
  tasks: [],
  notificationOpen: false,
  panelExpanded: false,
  expandedTaskId: null,
  reconnected: false,
  invalidation: { assets: 0, scripts: 0, storyboards: 0 },

  addPipeline: ({ label, projectId, request, onComplete }) => {
    const id = generateId();
    const initialState: PipelineState = {
      status: "running",
      reasoningText: "",
      timeline: [],
    };

    const task: PipelineTask = {
      id,
      label,
      projectId,
      status: "running",
      state: initialState,
      createdAt: Date.now(),
    };

    set((s) => ({ tasks: [...s.tasks, task] }));

    const handleEvent = createEventHandler(id, set, onComplete);

    // 启动 SSE 流
    const controller = pipelineStream(request, {
      onEvent: handleEvent,
      onError: (err) => {
        // 仅在 running 状态时标记错误，避免覆盖已取消/已完成的状态
        set((s) => ({
          tasks: s.tasks.map((t) =>
            t.id === id && t.status === "running"
              ? {
                  ...t,
                  status: "error" as const,
                  finishedAt: Date.now(),
                  state: {
                    ...t.state,
                    status: "error" as const,
                    error: err.message,
                  },
                }
              : t
          ),
        }));
        abortControllers.delete(id);
      },
      onComplete: () => {
        // SSE 流结束，如果还在 running 则标记为 done
        set((s) => ({
          tasks: s.tasks.map((t) =>
            t.id === id && t.status === "running"
              ? {
                  ...t,
                  status: "done" as const,
                  finishedAt: Date.now(),
                  state: { ...t.state, status: "done" as const },
                }
              : t
          ),
        }));
        abortControllers.delete(id);
        onComplete?.();
      },
    });

    abortControllers.set(id, controller);

    return id;
  },

  cancelPipeline: async (id: string) => {
    const task = get().tasks.find((t) => t.id === id);
    const controller = abortControllers.get(id);

    // 先标记为 cancelled，防止 abort 触发的 onError/onComplete 回调覆盖状态
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id
          ? {
              ...t,
              status: "cancelled" as const,
              finishedAt: Date.now(),
              state: { ...t.state, status: "cancelled" as const },
            }
          : t
      ),
    }));

    controller?.abort();
    abortControllers.delete(id);

    if (task?.state.conversationId) {
      try {
        await cancelPipeline(task.state.conversationId);
      } catch {
        // 忽略取消错误
      }
    }
  },

  removePipeline: (id: string) => {
    abortControllers.get(id)?.abort();
    abortControllers.delete(id);
    set((s) => ({
      tasks: s.tasks.filter((t) => t.id !== id),
      expandedTaskId: s.expandedTaskId === id ? null : s.expandedTaskId,
    }));
  },

  clearCompleted: () => {
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

  /**
   * 页面加载时调用：查询后端 running 对话 → 检查 Redis 流状态 → reconnect SSE
   */
  tryReconnect: () => {
    if (get().reconnected) return;
    set({ reconnected: true });

    // 异步查询后端
    (async () => {
      try {
        const runningConvs = await listRunningPipelines();
        if (runningConvs.length === 0) {
          console.log("[Pipeline] 无运行中任务");
          return;
        }

        console.log(
          `[Pipeline] 后端有 ${runningConvs.length} 个 running 对话，开始重连`
        );

        for (const conv of runningConvs) {
          const conversationId = conv.conversationId;
          const id = `reconnect-${conversationId}`;

          // 检查是否已存在（当前页面已有该任务的 SSE 连接）
          const existing = get().tasks.find(
            (t) => t.state.conversationId === conversationId
          );
          if (existing) continue;

          // 创建占位 task
          const placeholder: PipelineTask = {
            id,
            label: conv.title || "AI 任务",
            projectId: conv.projectId,
            status: "running",
            state: {
              status: "running",
              reasoningText: "",
              timeline: [{ type: "content", text: "正在重连……" }],
              conversationId,
            },
            createdAt: conv.createTime
              ? new Date(conv.createTime).getTime()
              : Date.now(),
          };

          set((s) => ({ tasks: [...s.tasks, placeholder] }));

          // 检查 Redis 流状态
          try {
            const streamStatus = await getPipelineStatus(conversationId);
            console.log(
              `[Pipeline] 对话 ${conversationId} Redis 状态: ${streamStatus}`
            );

            if (streamStatus === "ACTIVE") {
              // 后台仍在运行 → 重连 SSE
              const handleEvent = createEventHandler(id, set);

              // 清空占位文本
              set((s) => ({
                tasks: s.tasks.map((t) =>
                  t.id === id
                    ? { ...t, state: { ...t.state, timeline: [] } }
                    : t
                ),
              }));

              const controller = reconnectPipelineStream(conversationId, {
                onEvent: handleEvent,
                onError: (err) => {
                  set((s) => ({
                    tasks: s.tasks.map((t) =>
                      t.id === id
                        ? {
                            ...t,
                            status: "error" as const,
                            state: {
                              ...t.state,
                              status: "error" as const,
                              error: err.message,
                            },
                          }
                        : t
                    ),
                  }));
                  abortControllers.delete(id);
                },
                onComplete: () => {
                  set((s) => ({
                    tasks: s.tasks.map((t) =>
                      t.id === id && t.status === "running"
                        ? {
                            ...t,
                            status: "done" as const,
                            state: { ...t.state, status: "done" as const },
                          }
                        : t
                    ),
                  }));
                  abortControllers.delete(id);
                },
              });

              abortControllers.set(id, controller);
            } else {
              // Redis 流已结束 → 标记
              const finalStatus: PipelineTask["status"] =
                streamStatus === "COMPLETED" ? "done" : "error";
              set((s) => ({
                tasks: s.tasks.map((t) =>
                  t.id === id
                    ? {
                        ...t,
                        status: finalStatus,
                        state: {
                          ...t.state,
                          status:
                            finalStatus === "done"
                              ? ("done" as const)
                              : ("error" as const),
                          timeline: [
                            {
                              type: "content" as const,
                              text:
                                finalStatus === "done"
                                  ? "任务已完成"
                                  : "任务异常结束",
                            },
                          ],
                        },
                      }
                    : t
                ),
              }));
            }
          } catch (err) {
            console.error(`[Pipeline] 重连 ${conversationId} 失败:`, err);
            set((s) => ({
              tasks: s.tasks.map((t) =>
                t.id === id
                  ? {
                      ...t,
                      status: "error" as const,
                      state: {
                        ...t.state,
                        status: "error" as const,
                        error: "重连失败",
                        timeline: [
                          {
                            type: "content" as const,
                            text: "重连失败，任务可能已结束",
                          },
                        ],
                      },
                    }
                  : t
              ),
            }));
          }
        }
      } catch (err) {
        console.error("[Pipeline] 查询运行中任务失败:", err);
      }
    })();
  },
}));
