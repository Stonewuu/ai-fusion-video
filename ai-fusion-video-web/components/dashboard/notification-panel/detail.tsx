"use client";

import {
  useEffect,
  useMemo,
  useState,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Ban,
  Loader2,
  RefreshCw,
} from "lucide-react";
import {
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

export function PipelineDetailPanel({ taskId }: { taskId: string }) {
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
    if (!isRunning) {
      return;
    }
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
            正在加载任务内容…
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
              <span className="text-xs text-primary/70 font-medium">正在处理…</span>
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

export function HistoryDetailPanel({
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
    conversation.category !== "task" &&
    (conversation.category === "pipeline" ||
    (conversation.agentType != null &&
      (PIPELINE_AGENT_TYPES as readonly string[]).includes(conversation.agentType)));
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
