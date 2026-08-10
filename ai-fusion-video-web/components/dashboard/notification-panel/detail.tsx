"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowLeft,
  Ban,
  CheckCircle2,
  ChevronRight,
  Clock,
  Loader2,
  MessageSquare,
  RefreshCw,
  X,
  XCircle,
} from "lucide-react";
import {
  listConversations,
  listMessages,
  type AgentConversation,
  type AgentMessage,
} from "@/lib/api/ai-assistant";
import { PIPELINE_AGENT_TYPES } from "@/lib/api/ai-pipeline";
import { toastApiError } from "@/lib/api/toast-api-error";
import { Button } from "@/components/ui/button";
import {
  usePipelineStore,
  type PipelineTask,
} from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import {
  getAgentTypeName,
  getToolDisplayName,
} from "./constants";
import {
  messagesToTimeline,
  resolveHistoryErrorMessage,
} from "./history";
import {
  ElapsedText,
  useScrollElement,
  useSmartScroll,
} from "./hooks";
import { MessageTimeline } from "./timeline";
import {
  formatDatetime,
  formatElapsed,
  formatTimestamp,
  getElapsedStr,
} from "./utils";

const EMPTY_TIMELINE: PipelineTask["state"]["timeline"] = [];
const EMPTY_MESSAGES: AgentMessage[] = [];

function PipelineContinueAction({ onContinue }: { onContinue: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-border/30 bg-card/50 p-3">
      <div className="min-w-0">
        <p className="text-xs font-medium">执行未完成</p>
        <p className="mt-0.5 text-[11px] text-muted-foreground">
          保留当前进度并让 Agent 继续执行
        </p>
      </div>
      <Button type="button" variant="outline" size="sm" onClick={onContinue}>
        <RefreshCw />
        重试
      </Button>
    </div>
  );
}

function PipelineStateExpiredNotice() {
  return (
    <div className="rounded-xl border border-border/30 bg-card/50 p-3">
      <p className="text-xs font-medium">会话状态已过期</p>
      <p className="mt-0.5 text-[11px] text-muted-foreground">
        该会话状态存储已过期，请发起新会话
      </p>
    </div>
  );
}

function mergeConversationsById(
  current: AgentConversation[],
  incoming: AgentConversation[]
): AgentConversation[] {
  const merged = new Map(current.map((conversation) => [conversation.id, conversation]));
  incoming.forEach((conversation) => merged.set(conversation.id, conversation));
  return Array.from(merged.values());
}

function selectTaskListVersion(state: {
  tasks: PipelineTask[];
}): string {
  return JSON.stringify(
    state.tasks.map((task) => [
      task.id,
      task.label,
      task.status,
      task.createdAt,
      task.finishedAt,
      task.state.cancelRequested,
    ])
  );
}

function PipelineDetailPanel({ taskId }: { taskId: string }) {
  const task = usePipelineStore(
    (state) => state.tasks.find((candidate) => candidate.id === taskId) ?? null
  );
  const loadPipelineContent = usePipelineStore(
    (state) => state.loadPipelineContent
  );
  const disconnectPipelineContent = usePipelineStore(
    (state) => state.disconnectPipelineContent
  );
  const continuePipeline = usePipelineStore((state) => state.continuePipeline);
  const [idleTimelineLength, setIdleTimelineLength] = useState<number | null>(null);
  const timeline = task?.state.timeline ?? EMPTY_TIMELINE;
  const timelineLength = timeline.length;
  const isRunning = task?.status === "running";
  const cancelling = !!task?.state.cancelRequested;
  const isContentPending =
    task?.state.contentLoaded === false && !task.state.error;
  const isIdle = isRunning && !cancelling && idleTimelineLength === timelineLength;
  const canCancel = isRunning && !cancelling && task?.cancellable !== false;
  const showCancel = isRunning && task?.cancellable !== false;
  const [timelineElement, setTimelineElement] = useScrollElement();
  useSmartScroll(
    timelineElement,
    [timeline, isIdle],
    isRunning && !cancelling,
  );

  useEffect(() => {
    loadPipelineContent(taskId);
  }, [loadPipelineContent, task?.state.contentLoaded, task?.status, taskId]);

  useEffect(() => {
    return () => disconnectPipelineContent(taskId);
  }, [disconnectPipelineContent, taskId]);

  useEffect(() => {
    if (!isRunning) return;
    const timer = setTimeout(() => {
      setIdleTimelineLength(timelineLength);
    }, 2000);
    return () => clearTimeout(timer);
  }, [timelineLength, isRunning]);

  if (!task) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        任务已从当前列表中移除
      </div>
    );
  }

  const statusText = {
    running: "运行中",
    done: "已完成",
    error: "出错",
    cancelled: "已取消",
  };

  const statusColor = {
    running: "text-blue-400",
    done: "text-green-400",
    error: "text-destructive",
    cancelled: "text-muted-foreground",
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/20 shrink-0 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="text-sm font-semibold truncate">{task.label}</h4>
          <p className={cn("text-xs mt-0.5", statusColor[task.status])}>
            {cancelling ? "取消中" : statusText[task.status]} · <ElapsedText task={task} />
            <span className="text-muted-foreground/50 ml-1">
              启动于 {formatTimestamp(task.createdAt)}
            </span>
          </p>
        </div>
        {showCancel && (
          <Button
            type="button"
            variant="destructive-ghost"
            size="xs"
            disabled={!canCancel}
            onClick={() => {
              void usePipelineStore
                .getState()
                .cancelPipeline(task.id)
                .catch((error) => {
                  console.error("[Pipeline] 取消请求失败:", error);
                  toastApiError(error, "取消工作流失败");
                });
            }}
            aria-label={cancelling ? "正在取消工作流" : "停止工作流"}
          >
            {cancelling
              ? <Loader2 className="animate-spin motion-reduce:animate-none" />
              : <Ban />}
            {cancelling ? "取消中…" : "停止"}
          </Button>
        )}
      </div>

      <div ref={setTimelineElement} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
        {isContentPending && timeline.length === 0 ? (
          <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            正在加载 Pipeline 内容…
          </div>
        ) : (
          <MessageTimeline
            reasoningText={task.state.reasoningText}
            reasoningStartTime={task.state.reasoningStartTime}
            reasoningDurationMs={task.state.reasoningDurationMs}
            timeline={timeline}
            scrollElement={timelineElement}
            initialScrollToEnd
            streaming={isRunning && !cancelling}
            error={task.state.error}
          />
        )}

        <AnimatePresence>
          {isIdle && task.status === "running" && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.25 }}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary/5 border border-primary/10"
            >
              <Loader2 className="h-3 w-3 animate-spin text-primary/60" />
              <span className="text-xs text-primary/70 font-medium">AI 全力处理中…</span>
            </motion.div>
          )}
        </AnimatePresence>

        {(task.status === "error" || task.status === "cancelled") &&
          task.continuable && (
          <PipelineContinueAction
            onContinue={() => {
              try {
                continuePipeline(task.id);
              } catch (error) {
                console.error("[Pipeline] 继续执行失败:", error);
                toastApiError(error, "继续执行失败");
              }
            }}
          />
        )}
      </div>
    </div>
  );
}

function HistoryDetailPanel({
  conversation,
  onContinue,
}: {
  conversation: AgentConversation;
  onContinue: (
    conversation: AgentConversation,
    timeline: PipelineTask["state"]["timeline"]
  ) => void;
}) {
  const [messageState, setMessageState] = useState<{
    conversationId: string;
    messages: AgentMessage[];
  } | null>(null);
  const loading = messageState?.conversationId !== conversation.conversationId;
  const messages =
    messageState?.conversationId === conversation.conversationId
      ? messageState.messages
      : EMPTY_MESSAGES;
  const [timelineElement, setTimelineElement] = useScrollElement();
  useSmartScroll(timelineElement, [messages], false);

  useEffect(() => {
    let cancelled = false;
    listMessages(conversation.conversationId)
      .then((nextMessages) => {
        if (!cancelled) {
          setMessageState({
            conversationId: conversation.conversationId,
            messages: nextMessages,
          });
        }
      })
      .catch((err) => {
        console.error(err);
        toastApiError(err, "加载对话消息失败");
        if (!cancelled) {
          setMessageState({
            conversationId: conversation.conversationId,
            messages: [],
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [conversation.conversationId]);

  const isPipeline =
    conversation.category === "task" ||
    conversation.category === "pipeline" ||
    (conversation.agentType != null &&
      (PIPELINE_AGENT_TYPES as readonly string[]).includes(conversation.agentType));
  const displayMessages = useMemo(
    () =>
      isPipeline
        ? messages.filter((message) => message.role !== "user")
        : messages,
    [isPipeline, messages]
  );
  const firstAssistant = displayMessages.find(
    (message) =>
      message.role === "assistant" &&
      !message.parentToolCallId &&
      message.reasoningContent
  );
  const timeline = useMemo(
    () => messagesToTimeline(displayMessages),
    [displayMessages]
  );
  const historyError = useMemo(
    () => resolveHistoryErrorMessage(conversation, displayMessages),
    [conversation, displayMessages]
  );
  const stateExpired = conversation.agentStateStatus === "EXPIRED";

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/20 shrink-0">
        <h4 className="text-sm font-semibold truncate">{conversation.title}</h4>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          {conversation.agentType && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-medium">
              {getAgentTypeName(conversation.agentType)}
            </span>
          )}
          {conversation.createTime && conversation.lastMessageTime && (() => {
            const duration =
              new Date(conversation.lastMessageTime).getTime() -
              new Date(conversation.createTime).getTime();
            return duration > 0 ? (
              <span className="text-xs text-muted-foreground">
                耗时 {formatElapsed(duration)}
              </span>
            ) : null;
          })()}
          <span className="text-xs text-muted-foreground">
            {formatDatetime(conversation.createTime)}
          </span>
        </div>
      </div>

      <div ref={setTimelineElement} className="flex-1 min-h-0 overflow-y-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : displayMessages.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-8">暂无消息记录</p>
        ) : (
          <MessageTimeline
            reasoningText={firstAssistant?.reasoningContent || undefined}
            reasoningDurationMs={firstAssistant?.reasoningDurationMs ?? undefined}
            timeline={timeline}
            scrollElement={timelineElement}
            initialScrollToEnd
            streaming={false}
            error={historyError}
          />
        )}
        {!loading &&
          isPipeline &&
          stateExpired && (
          <div className="mt-3">
            <PipelineStateExpiredNotice />
          </div>
        )}
        {!loading &&
          isPipeline &&
          !stateExpired &&
          (historyError || conversation.status === "cancelled") && (
          <div className="mt-3">
            <PipelineContinueAction
              onContinue={() => onContinue(conversation, timeline)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function TaskListItem({
  label,
  subtitle,
  icon,
  selected,
  onClick,
}: {
  label: string;
  subtitle: ReactNode;
  icon: ReactNode;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-colors mb-0.5",
        selected ? "bg-foreground/10" : "hover:bg-foreground/5"
      )}
    >
      {icon}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate">{label}</p>
        <p className="text-[10px] text-muted-foreground truncate">{subtitle}</p>
      </div>
    </button>
  );
}

export function PipelineTaskCard({ task }: { task: PipelineTask }) {
  const setPanelExpanded = usePipelineStore((state) => state.setPanelExpanded);
  const setExpandedTaskId = usePipelineStore((state) => state.setExpandedTaskId);
  const isRunning = task.status === "running";
  const cancelling = !!task.state.cancelRequested;

  const statusIcon = {
    running: <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />,
    done: <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />,
    error: <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />,
    cancelled: <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />,
  };

  const statusText = {
    running: "运行中",
    done: "已完成",
    error: "出错",
    cancelled: "已取消",
  };

  const getLatestActivity = () => {
    if (cancelling) return "取消中…";
    const timeline = task.state.timeline;
    const isTaskStream = task.cancellable === false;
    for (let index = timeline.length - 1; index >= 0; index--) {
      const item = timeline[index];
      if (item.type === "reasoning") {
        return "AI 正在思考…";
      }
      if (item.type === "tool") {
        return item.status === "calling"
          ? `正在${getToolDisplayName(item.name)}…`
          : `${getToolDisplayName(item.name)} 已完成`;
      }
      if (item.type === "content") {
        if (isTaskStream) {
          return task.status === "running" ? "任务执行中…" : "已记录任务结果";
        }
        return task.status === "running" ? "AI 正在输出…" : "已生成回复";
      }
    }

    if (task.state.reasoningText) return "AI 正在思考…";
    return isTaskStream ? "任务准备中…" : "准备中…";
  };

  const handleOpenDetail = () => {
    setExpandedTaskId(task.id);
    setPanelExpanded(true);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8, height: 0 }}
      className={cn(
        "rounded-xl border overflow-hidden transition-colors",
        isRunning
          ? "border-blue-500/20 bg-blue-500/5"
          : task.status === "done"
            ? "border-green-500/20 bg-green-500/5"
            : task.status === "error"
              ? "border-destructive/20 bg-destructive/5"
              : "border-border/20 bg-muted/20"
      )}
    >
      <div
        className="flex items-center gap-2 px-3 py-2.5 cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
        onClick={handleOpenDetail}
      >
        {statusIcon[task.status]}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium truncate">{task.label}</p>
          <p className="text-[10px] text-muted-foreground truncate">
            {isRunning ? getLatestActivity() : statusText[task.status]}
            {" · "}
            <ElapsedText task={task} />
          </p>
        </div>
        <div className="flex items-center shrink-0">
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        </div>
      </div>
    </motion.div>
  );
}

type SelectedItem =
  | { type: "pipeline"; taskId: string }
  | { type: "history"; conversation: AgentConversation };

export function ExpandedPanel({ onClose }: { onClose: () => void }) {
  // Timeline chunks change task objects very frequently. The shell only needs
  // task metadata, so keep it stable until list-visible fields actually change.
  const taskListVersion = usePipelineStore(selectTaskListVersion);
  const taskListSnapshotRef = useRef<{
    version: string;
    tasks: PipelineTask[];
  } | null>(null);
  if (taskListSnapshotRef.current?.version !== taskListVersion) {
    taskListSnapshotRef.current = {
      version: taskListVersion,
      tasks: usePipelineStore.getState().tasks,
    };
  }
  const tasks = taskListSnapshotRef.current.tasks;
  const clearCompleted = usePipelineStore((state) => state.clearCompleted);
  const removePipeline = usePipelineStore((state) => state.removePipeline);
  const expandedTaskId = usePipelineStore((state) => state.expandedTaskId);
  const listEndRef = useRef<HTMLDivElement>(null);
  // mobile: track whether we're showing detail view
  const [mobileShowDetail, setMobileShowDetail] = useState(false);
  // closing animation state
  const [isClosing, setIsClosing] = useState(false);

  const handleClose = useCallback(() => {
    setIsClosing(true);
  }, []);

  const [selected, setSelected] = useState<SelectedItem | null>(() => {
    if (expandedTaskId) {
      const target = tasks.find((task) => task.id === expandedTaskId);
      if (target) return { type: "pipeline", taskId: target.id };
    }
    const running = tasks.find((task) => task.status === "running");
    return running ? { type: "pipeline", taskId: running.id } : null;
  });
  const [conversations, setConversations] = useState<AgentConversation[]>([]);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const pageSize = 20;

  const hasMore = conversations.length < historyTotal;

  const loadHistory = useCallback(
    async (page: number, append = false) => {
      setHistoryLoading(true);
      try {
        const result = await listConversations({ pageNo: page, pageSize });
        const currentConversationIds = new Set(
          usePipelineStore
            .getState()
            .tasks.filter((task) => task.state.conversationId)
            .map((task) => task.state.conversationId!)
        );
        const filtered = result.list.filter(
          (conversation) =>
            conversation.category !== "assistant" &&
            conversation.status !== "running" &&
            !currentConversationIds.has(conversation.conversationId)
        );

        if (append) {
          setConversations((current) => mergeConversationsById(current, filtered));
        } else {
          const uniqueConversations = mergeConversationsById([], filtered);
          setConversations(uniqueConversations);
          setSelected((current) => {
            if (current?.type === "pipeline") {
              const stillPresent = usePipelineStore
                .getState()
                .tasks.some((task) => task.id === current.taskId);
              if (stillPresent) return current;
            }
            if (current?.type === "history") {
              const refreshed = uniqueConversations.find(
                (conversation) => conversation.id === current.conversation.id
              );
              return refreshed
                ? { type: "history", conversation: refreshed }
                : current;
            }
            return uniqueConversations[0]
              ? { type: "history", conversation: uniqueConversations[0] }
              : null;
          });
        }

        setHistoryTotal(result.total - (result.list.length - filtered.length));
        setHistoryPage(page);
      } catch (err) {
        console.error("加载历史对话失败:", err);
        toastApiError(err, "加载历史对话失败");
      } finally {
        setHistoryLoading(false);
      }
    },
    [pageSize]
  );

  useEffect(() => {
    loadHistory(1);
  }, [loadHistory]);

  const prevTasksRef = useRef(tasks);
  useEffect(() => {
    const prevTasks = prevTasksRef.current;
    prevTasksRef.current = tasks;

    const justFinished = tasks.some((task) => {
      if (task.status === "running") return false;
      const prev = prevTasks.find((previousTask) => previousTask.id === task.id);
      return prev && prev.status === "running";
    });

    if (justFinished) {
      const timer = setTimeout(() => {
        loadHistory(1);
        usePipelineStore
          .getState()
          .tasks.filter(
            (task) => task.status === "done"
          )
          .forEach((task) => removePipeline(task.id));
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [tasks, loadHistory, removePipeline]);

  useEffect(() => {
    if (!listEndRef.current || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !historyLoading && hasMore) {
          loadHistory(historyPage + 1, true);
        }
      },
      { threshold: 0.1 }
    );
    observer.observe(listEndRef.current);
    return () => observer.disconnect();
  }, [hasMore, historyLoading, historyPage, loadHistory]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") handleClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleClose]);

  const selectedPipelineTaskId =
    selected?.type === "pipeline" &&
    tasks.some((task) => task.id === selected.taskId)
      ? selected.taskId
      : null;
  const selectedConversation =
    selected?.type === "history" ? selected.conversation : null;
  const runningTasks = tasks.filter((task) => task.status === "running");
  const completedTasks = tasks.filter((task) => task.status !== "running");
  const currentPipelineConversationIds = new Set(
    tasks
      .filter((task) => task.continuable && task.state.conversationId)
      .map((task) => task.state.conversationId)
  );
  const visibleConversations = conversations.filter(
    (conversation) =>
      !currentPipelineConversationIds.has(conversation.conversationId)
  );

  // helper: select an item and switch to detail view on mobile
  const handleSelect = (item: SelectedItem) => {
    setSelected(item);
    setMobileShowDetail(true);
  };

  // helper: go back to list on mobile
  const handleMobileBack = () => {
    setMobileShowDetail(false);
  };

  const handleContinueHistory = (
    conversation: AgentConversation,
    timeline: PipelineTask["state"]["timeline"]
  ) => {
    try {
      const taskId = usePipelineStore.getState().addPipelineContinuation({
        label: conversation.title,
        projectId: conversation.projectId,
        conversationId: conversation.conversationId,
        initialTimeline: timeline,
      });
      usePipelineStore.getState().setExpandedTaskId(taskId);
      setSelected({ type: "pipeline", taskId });
      setMobileShowDetail(true);
    } catch (error) {
      console.error("[Pipeline] 历史任务继续执行失败:", error);
      toastApiError(error, "继续执行失败");
    }
  };

  /* ---- shared list content (used in both desktop sidebar and mobile full view) ---- */
  const listContent = (
    <>
      {runningTasks.length > 0 && (
        <div className="px-3 pt-3 pb-1">
          <p className="text-[10px] font-medium text-muted-foreground px-1 pb-1.5 uppercase tracking-wider">
            运行中 ({runningTasks.length})
          </p>
          {runningTasks.map((task) => (
            <TaskListItem
              key={task.id}
              label={task.label}
              subtitle={task.state.cancelRequested
                ? <>取消中… · <ElapsedText task={task} /></>
                : <ElapsedText task={task} />}
              icon={<Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />}
              selected={selected?.type === "pipeline" && selected.taskId === task.id}
              onClick={() => handleSelect({ type: "pipeline", taskId: task.id })}
            />
          ))}
        </div>
      )}

      {completedTasks.length > 0 && (
        <div className="px-3 pt-2 pb-1">
          <p className="text-[10px] font-medium text-muted-foreground px-1 pb-1.5 uppercase tracking-wider">
            当前会话 ({completedTasks.length})
          </p>
          {completedTasks.map((task) => (
            <TaskListItem
              key={task.id}
              label={task.label}
              subtitle={`${
                task.status === "done"
                  ? "已完成"
                  : task.status === "error"
                    ? "出错"
                    : "已取消"
              } · ${getElapsedStr(task)}`}
              icon={
                task.status === "done" ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
                ) : task.status === "error" ? (
                  <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                ) : (
                  <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                )
              }
              selected={selected?.type === "pipeline" && selected.taskId === task.id}
              onClick={() => handleSelect({ type: "pipeline", taskId: task.id })}
            />
          ))}
        </div>
      )}

      <div className="px-3 pt-2 pb-3">
        <p className="text-[10px] font-medium text-muted-foreground px-1 pb-1.5 uppercase tracking-wider flex items-center gap-1">
          <Clock className="h-3 w-3" />
          历史记录
        </p>
        {visibleConversations.length === 0 && !historyLoading && (
          <p className="text-xs text-muted-foreground/60 px-1 py-3 text-center">
            暂无历史记录
          </p>
        )}
        {visibleConversations.map((conversation) => {
          const isError =
            conversation.status === "error" ||
            conversation.status === "failed";
          const isDone =
            conversation.status === "completed" ||
            conversation.status === "done";
          const isCancelled = conversation.status === "cancelled";

          return (
            <TaskListItem
              key={conversation.id}
              label={conversation.title}
              subtitle={
                <>
                  {conversation.agentType
                    ? getAgentTypeName(conversation.agentType)
                    : "对话"}
                  {isError
                    ? " · 出错"
                    : isCancelled
                      ? " · 已取消"
                      : isDone
                        ? " · 已完成"
                        : ""}
                  {conversation.createTime && conversation.lastMessageTime && (() => {
                    const duration =
                      new Date(conversation.lastMessageTime).getTime() -
                      new Date(conversation.createTime).getTime();
                    return duration > 0 ? ` · ${formatElapsed(duration)}` : "";
                  })()}
                </>
              }
              icon={
                isError ? (
                  <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                ) : isDone ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
                ) : isCancelled ? (
                  <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                ) : (
                  <MessageSquare className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0" />
                )
              }
              selected={
                selected?.type === "history" &&
                selected.conversation.id === conversation.id
              }
              onClick={() =>
                handleSelect({ type: "history", conversation })
              }
            />
          );
        })}

        <div ref={listEndRef} className="h-1" />
        {historyLoading && (
          <div className="flex items-center justify-center py-3">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        )}
      </div>
    </>
  );

  /* ---- detail content ---- */
  const detailContent = selectedPipelineTaskId ? (
    <PipelineDetailPanel
      key={selectedPipelineTaskId}
      taskId={selectedPipelineTaskId}
    />
  ) : selectedConversation ? (
    <HistoryDetailPanel
      key={selectedConversation.conversationId}
      conversation={selectedConversation}
      onContinue={handleContinueHistory}
    />
  ) : (
    <div className="flex items-center justify-center h-full text-muted-foreground">
      <div className="text-center">
        <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-30" />
        <p className="text-sm">选择一个任务查看详情</p>
      </div>
    </div>
  );

  return createPortal(
    <AnimatePresence onExitComplete={onClose}>
      {!isClosing && (
        <>
          <motion.div
            key="expanded-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="modal-overlay fixed inset-0 z-60"
            onClick={handleClose}
          />
          <motion.div
            key="expanded-panel"
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className={cn(
              "fixed z-61 flex flex-col overflow-hidden",
              // mobile: full-screen with safe padding
              "inset-3 rounded-xl",
              // desktop: centered dialog
              "md:inset-auto md:left-1/2 md:top-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-[1400px] md:max-w-[92vw] md:h-[75vh] md:rounded-2xl",
              "modal-surface border border-border/40"
            )}
          >
            {/* ---- Header ---- */}
            <div className="flex items-center justify-between px-4 md:px-5 py-3 md:py-3.5 border-b border-border/20 shrink-0">
              <div className="flex items-center gap-2">
                {/* Mobile back button when in detail view */}
                {mobileShowDetail && (
                  <button
                    onClick={handleMobileBack}
                    className="md:hidden p-1 -ml-1 rounded-lg hover:bg-muted transition-colors"
                  >
                    <ArrowLeft className="h-4 w-4 text-muted-foreground" />
                  </button>
                )}
                <h3 className="text-base font-semibold">AI 任务中心</h3>
              </div>
              <div className="flex items-center gap-2">
                {completedTasks.length > 0 && (
                  <button
                    onClick={clearCompleted}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-lg hover:bg-muted hidden md:block"
                  >
                    清除已完成
                  </button>
                )}
                <button
                  onClick={handleClose}
                  className="p-1.5 rounded-lg hover:bg-muted transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              </div>
            </div>

            {/* One responsive tree avoids mounting the full timeline twice. */}
            <div className="flex flex-1 min-h-0">
              <div
                className={cn(
                  "w-full min-h-0 flex-col overflow-hidden md:flex md:w-72 md:shrink-0 md:border-r md:border-border/20",
                  mobileShowDetail ? "hidden" : "flex"
                )}
              >
                <div className="flex-1 min-h-0 overflow-y-auto">
                  {listContent}
                </div>
              </div>
              <div
                className={cn(
                  "flex-1 min-w-0 min-h-0 bg-muted/10 md:block",
                  mobileShowDetail ? "block" : "hidden"
                )}
              >
                {detailContent}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
}
