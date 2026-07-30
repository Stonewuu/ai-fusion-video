"use client";

import { create } from "zustand";
import {
  continuePipelineStream,
  pipelineStream,
  reconnectPipelineStream,
  cancelPipeline,
  listRunningPipelines,
  type AiChatReq,
  type PipelineRunStatusResponse,
  type PipelineRunStatus,
} from "@/lib/api/ai-pipeline";
import {
  reconnectTaskStream,
  getTaskStreamStatus,
} from "@/lib/api/task-stream";
import {
  cancelCallingTimelineTools,
  restoreOptimisticallyCancelledTimelineTools,
  type TimelineItem,
} from "@/lib/store/pipeline-timeline";
import {
  createPipelineEventHandler,
  generatePipelineTaskId,
  type PipelineEventLifecycle,
} from "@/lib/store/pipeline-event-handler";
import {
  PipelineStatusSynchronizer,
  type PipelineStatusSyncTarget,
} from "@/lib/store/pipeline-status-sync";

export type {
  SubTimelineItem,
  TimelineItem,
  ToolTimelineStatus,
} from "@/lib/store/pipeline-timeline";

// ========== 数据失效映射 ==========

/** 数据失效类型 */
export type InvalidationType = "assets" | "scripts" | "storyboards";

// ========== 类型 ==========

export interface PipelineState {
  status: "running" | "done" | "error" | "cancelled";
  reasoningText: string;
  reasoningStartTime?: number;
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
  /** 取消请求已提交，等待 durable terminal event。 */
  cancelRequested?: boolean;
  /** 新一轮输出需要与上一轮末尾的文本分开渲染。 */
  separateNextRootContent?: boolean;
}

export interface PipelineTask {
  id: string;
  label: string;
  projectId: number | null;
  status: "running" | "done" | "error" | "cancelled";
  state: PipelineState;
  createdAt: number;
  cancellable?: boolean;
  /** 当前浏览器仍持有启动配置，可在失败后继续同一会话。 */
  continuable?: boolean;
  /** 任务结束时间（done/error/cancelled 时记录） */
  finishedAt?: number;
}

// ========== Store ==========

export interface PipelineStoreState {
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
  addPipelineContinuation: (config: {
    label: string;
    projectId: number | null;
    conversationId: string;
    initialTimeline?: TimelineItem[];
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
  continuePipeline: (id: string) => void;
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
const pipelineCompleteCallbacks = new Map<string, () => void>();
const notifiedPipelineSettlements = new Set<string>();

// 存储 AbortController 的 map（不放在 zustand state 里避免序列化问题）
const abortControllers = new Map<string, AbortController>();
const PIPELINE_STATUS_POLL_INTERVAL_MS = 3000;

function schedulePipelineStatusSync(delayMs?: number): void {
  pipelineStatusSynchronizer.schedule(delayMs);
}

function notifyPipelineSettlement(
  id: string,
  status: "done" | "error" | "cancelled",
  onComplete?: () => void,
  onSettled?: (status: "done" | "error" | "cancelled") => void,
): void {
  if (notifiedPipelineSettlements.has(id)) return;
  notifiedPipelineSettlements.add(id);
  abortControllers.delete(id);
  if (status === "done") onComplete?.();
  onSettled?.(status);
}

const pipelineEventLifecycle: PipelineEventLifecycle = {
  notifySettlement: notifyPipelineSettlement,
  scheduleStatusSync: schedulePipelineStatusSync,
};

function toTerminalTaskStatus(
  status: PipelineRunStatus
): Exclude<PipelineTask["status"], "running"> | null {
  if (status === "COMPLETED") return "done";
  if (status === "FAILED") return "error";
  if (status === "CANCELLED") return "cancelled";
  return null;
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
          cancelRequested: false,
          timeline: status === "cancelled"
            ? cancelCallingTimelineTools(t.state.timeline)
            : t.state.timeline,
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
    notifyPipelineSettlement(
      id,
      status,
      options?.onComplete,
      options?.onSettled,
    );
  }
}

function startPipelineContinuationStream(
  id: string,
  conversationId: string,
  set: (fn: (state: PipelineStoreState) => Partial<PipelineStoreState>) => void,
  get: () => PipelineStoreState,
  onComplete?: () => void,
) {
  const handleEvent = createPipelineEventHandler(
    pipelineEventLifecycle,
    id,
    set,
    onComplete,
  );
  const controller = continuePipelineStream(conversationId, {
    onEvent: handleEvent,
    onError: (error) => {
      const current = get().tasks.find((candidate) => candidate.id === id);
      if (current?.status !== "running") return;
      if (!current.state.runId) {
        settleTaskIfRunning(set, id, "error", {
          error: `继续执行失败：${error.message}`,
        });
        return;
      }
      set((state) => ({
        tasks: state.tasks.map((candidate) =>
          candidate.id === id
            ? {
                ...candidate,
                state: {
                  ...candidate.state,
                  error: `Pipeline 连接中断：${error.message}`,
                },
              }
            : candidate,
        ),
      }));
      abortControllers.delete(id);
    },
    onComplete: () => {
      abortControllers.delete(id);
    },
  });
  abortControllers.set(id, controller);
}

export const usePipelineStore = create<PipelineStoreState>()((set, get) => ({
  tasks: [],
  notificationOpen: false,
  panelExpanded: false,
  expandedTaskId: null,
  runningPipelinesRestored: false,
  invalidation: { assets: 0, scripts: 0, storyboards: 0 },

  addSimpleTask: ({ label, projectId, initialNote, onComplete }) => {
    const id = generatePipelineTaskId();
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
    const id = generatePipelineTaskId();
    const initialState: PipelineState = {
      status: "running",
      reasoningText: "",
      timeline: [],
      lastSequence: 0,
      conversationId: request.conversationId,
    };

    const task: PipelineTask = {
      id,
      label,
      projectId,
      status: "running",
      state: initialState,
      createdAt: Date.now(),
      cancellable: true,
      continuable: true,
    };

    set((s) => ({ tasks: [...s.tasks, task] }));
    if (onComplete) pipelineCompleteCallbacks.set(id, onComplete);

    const handleEvent = createPipelineEventHandler(
      pipelineEventLifecycle,
      id,
      set,
      onComplete,
    );

    // 启动 SSE 流
    const controller = pipelineStream(request, {
      onEvent: handleEvent,
      onError: (err) => {
        const current = get().tasks.find((candidate) => candidate.id === id);
        if (current?.status === "running" && !current.state.runId) {
          settleTaskIfRunning(set, id, "error", {
            error: `Pipeline 启动失败：${err.message}`,
          });
          return;
        }
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
        schedulePipelineStatusSync(0);
      },
      onComplete: () => {
        // 业务终态已由 journal terminal event 在 handleEvent 中处理。
        abortControllers.delete(id);
      },
    });

    abortControllers.set(id, controller);

    return id;
  },

  addPipelineContinuation: ({
    label,
    projectId,
    conversationId,
    initialTimeline = [],
  }) => {
    const id = generatePipelineTaskId();
    const task: PipelineTask = {
      id,
      label,
      projectId,
      status: "running",
      state: {
        status: "running",
        reasoningText: "",
        timeline: [...initialTimeline],
        lastSequence: 0,
        conversationId,
        separateNextRootContent: initialTimeline.length > 0,
      },
      createdAt: Date.now(),
      cancellable: true,
      continuable: true,
    };
    set((state) => ({ tasks: [...state.tasks, task] }));
    startPipelineContinuationStream(id, conversationId, set, get);
    return id;
  },

  continuePipeline: (id: string) => {
    const task = get().tasks.find((candidate) => candidate.id === id);
    if (
      !task ||
      (task.status !== "error" && task.status !== "cancelled")
    ) {
      return;
    }
    if (!task.continuable || !task.state.conversationId) {
      throw new Error("当前 Pipeline 缺少继续执行所需的会话配置");
    }
    abortControllers.get(id)?.abort();
    abortControllers.delete(id);
    notifiedPipelineSettlements.delete(id);
    set((state) => ({
      tasks: state.tasks.map((candidate) =>
        candidate.id === id
          ? {
              ...candidate,
              status: "running",
              finishedAt: undefined,
              state: {
                ...candidate.state,
                status: "running",
                reasoningText: "",
                reasoningStartTime: undefined,
                reasoningDurationMs: undefined,
                runId: undefined,
                lastSequence: 0,
                error: undefined,
                runStatus: undefined,
                cancelRequested: false,
                contentLoaded: true,
                contentLoading: false,
                reconnectOnSelect: false,
                separateNextRootContent: true,
              },
            }
          : candidate,
      ),
    }));

    startPipelineContinuationStream(
      id,
      task.state.conversationId,
      set,
      get,
      pipelineCompleteCallbacks.get(id),
    );
  },

  attachTaskStream: ({
    label,
    projectId,
    taskId,
    cancellable = false,
    onComplete,
    onSettled,
  }) => {
    const id = generatePipelineTaskId();
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

        const handleEvent = createPipelineEventHandler(
          pipelineEventLifecycle,
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
    if (!task || task.status !== "running" || task.state.cancelRequested) return;
    if (!task.state.runId) {
      const error = new Error("Pipeline 尚未返回 runId，无法提交取消请求");
      set((s) => ({
        tasks: s.tasks.map((item) =>
          item.id === id
            ? { ...item, state: { ...item.state, error: `取消请求失败：${error.message}` } }
            : item
        ),
      }));
      throw error;
    }
    const previousTimeline = task.state.timeline;
    set((s) => ({
      tasks: s.tasks.map((item) =>
        item.id === id
          ? {
              ...item,
              state: {
                ...item.state,
                cancelRequested: true,
                error: undefined,
                timeline: cancelCallingTimelineTools(item.state.timeline),
              },
            }
          : item
      ),
    }));
    schedulePipelineStatusSync(0);
    try {
      await cancelPipeline({ runId: task.state.runId });
      // 保持 SSE 连接，等待服务端持久化并发送 CANCELLED terminal event。
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      set((s) => ({
        tasks: s.tasks.map((item) =>
          item.id === id
            ? {
                ...item,
                state: item.state.cancelRequested
                  ? {
                      ...item.state,
                      cancelRequested: false,
                      timeline: restoreOptimisticallyCancelledTimelineTools(
                        item.state.timeline,
                        previousTimeline,
                      ),
                      error: `取消请求失败：${message}`,
                    }
                  : item.state,
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
    pipelineCompleteCallbacks.delete(id);
    notifiedPipelineSettlements.delete(id);
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
      pipelineCompleteCallbacks.delete(id);
      notifiedPipelineSettlements.delete(id);
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
      task.state.contentLoaded === true ||
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
                      reasoningStartTime: undefined,
                      reasoningDurationMs: undefined,
                      timeline: [],
                    }
                  : {}),
              },
            }
          : candidate
      ),
    }));

    const handleEvent = createPipelineEventHandler(
      pipelineEventLifecycle,
      id,
      set,
    );
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
          schedulePipelineStatusSync(0);
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
    schedulePipelineStatusSync(0);
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
              cancelRequested: run.status === "CANCEL_REQUESTED",
            },
            createdAt: new Date(run.startedAt).getTime(),
            cancellable: true,
            continuable: true,
          };
          set((s) => ({ tasks: [...s.tasks, placeholder] }));
        }
        schedulePipelineStatusSync(0);
      } catch (err) {
        console.error("[Pipeline] 查询 durable runs 失败:", err);
      }
    })();
  },
}));

function trackedPipelineStatuses(): PipelineStatusSyncTarget[] {
  return usePipelineStore.getState().tasks.flatMap((task) =>
    task.status === "running"
      && !task.state.contentLoading
      && task.state.runId
      ? [{ taskId: task.id, runId: task.state.runId }]
      : []
  );
}

function applyPipelineStatus(
  target: PipelineStatusSyncTarget,
  status: PipelineRunStatusResponse,
): void {
  const terminalStatus = toTerminalTaskStatus(status.status);
  let transitioned = false;
  let removed = false;
  usePipelineStore.setState((state) => {
    const current = state.tasks.find(
      (candidate) => candidate.id === target.taskId,
    );
    if (
      !current
      || current.state.runId !== target.runId
      || current.state.contentLoading
      || current.status !== "running"
    ) {
      return {};
    }

    if (
      terminalStatus
      && current.state.reconnectOnSelect
      && !state.panelExpanded
    ) {
      transitioned = true;
      removed = true;
      return {
        tasks: state.tasks.filter(
          (candidate) => candidate.id !== target.taskId,
        ),
      };
    }

    const tasks = state.tasks.map((candidate) => {
      if (candidate.id !== target.taskId) return candidate;
      transitioned = terminalStatus !== null;
      return {
        ...candidate,
        status: terminalStatus ?? candidate.status,
        ...(terminalStatus ? { finishedAt: Date.now() } : {}),
        state: {
          ...candidate.state,
          runStatus: status.status,
          status: terminalStatus ?? candidate.state.status,
          cancelRequested: status.status === "CANCEL_REQUESTED",
          timeline: terminalStatus === "cancelled"
            || status.status === "CANCEL_REQUESTED"
            ? cancelCallingTimelineTools(candidate.state.timeline)
            : candidate.state.timeline,
          ...(terminalStatus === "error"
            ? { error: status.terminalEvent?.error || "任务失败" }
            : {}),
        },
      };
    });

    return terminalStatus
      ? {
          tasks,
          invalidation: {
            assets: (state.invalidation.assets || 0) + 1,
            scripts: (state.invalidation.scripts || 0) + 1,
            storyboards: (state.invalidation.storyboards || 0) + 1,
          },
        }
      : { tasks };
  });

  if (transitioned && terminalStatus) {
    notifyPipelineSettlement(
      target.taskId,
      terminalStatus,
      pipelineCompleteCallbacks.get(target.taskId),
    );
    if (removed) {
      pipelineCompleteCallbacks.delete(target.taskId);
      notifiedPipelineSettlements.delete(target.taskId);
    }
  }
}

const pipelineStatusSynchronizer = new PipelineStatusSynchronizer({
  intervalMs: PIPELINE_STATUS_POLL_INTERVAL_MS,
  listTargets: trackedPipelineStatuses,
  applyStatus: applyPipelineStatus,
  reportError: (target, error) => {
    console.error(`[Pipeline] 查询 ${target.runId} 状态失败:`, error);
  },
});
