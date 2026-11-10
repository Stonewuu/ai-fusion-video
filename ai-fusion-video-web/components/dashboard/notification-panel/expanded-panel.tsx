"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, MessageSquare, X } from "lucide-react";
import { listConversations, type AgentConversation } from "@/lib/api/ai-assistant";
import { toastApiError } from "@/lib/api/toast-api-error";
import type { ExecutionRecordRef } from "@/lib/api/execution-records";
import { selectedExecutionRecord } from "@/lib/execution-record-navigation";
import { usePipelineStore, type PipelineTask } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { PipelineDetailPanel, HistoryDetailPanel } from "./detail";
import { ExecutionRecordDetailPanel } from "./execution-record-detail";
import { NotificationViews, type NotificationView } from "./notification-views";
import { TaskHistoryList, type TaskSelection } from "./task-history-list";

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

export function ExpandedPanel({ onClose }: { onClose: () => void }) {
  // 时间线频繁更新时，只在列表可见字段变化后重绘外层面板。
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
  // 移动端在列表与详情之间切换。
  const [mobileShowDetail, setMobileShowDetail] = useState(() => !!expandedTaskId);
  // 完成退出动画后再关闭浮层。
  const [open, setOpen] = useState(true);

  const handleClose = useCallback(() => {
    setOpen(false);
  }, []);

  const [view, setView] = useState<NotificationView>(() => selectedExecutionRecord(expandedTaskId) ? "exceptions" : "tasks");
  const [selectedRecord, setSelectedRecord] = useState<ExecutionRecordRef | null>(() => selectedExecutionRecord(expandedTaskId));
  const [selected, setSelected] = useState<TaskSelection | null>(() => {
    if (expandedTaskId) {
      const target = tasks.find((task) => task.id === expandedTaskId);
      if (target) {
        return { type: "pipeline", taskId: target.id };
      }
    }
    const running = tasks.find((task) => task.status === "running");
    return running ? { type: "pipeline", taskId: running.id } : null;
  });
  const [observedExpandedId, setObservedExpandedId] = useState(expandedTaskId);
  if (observedExpandedId !== expandedTaskId) {
    setObservedExpandedId(expandedTaskId);
    const record = selectedExecutionRecord(expandedTaskId);
    if (record) {
      setSelectedRecord(record);
      setView("exceptions");
      setMobileShowDetail(true);
    } else if (expandedTaskId && tasks.some(task => task.id === expandedTaskId)) {
      setSelected({ type: "pipeline", taskId: expandedTaskId });
      setView("tasks");
      setMobileShowDetail(true);
    }
  }
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
              if (stillPresent) {
                return current;
              }
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
      if (task.status === "running") {
        return false;
      }
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
    if (!listEndRef.current || !hasMore) {
      return;
    }

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

  // 选择任务后在移动端显示详情。
  const handleSelect = (item: TaskSelection) => {
    setView("tasks");
    setSelected(item);
    setMobileShowDetail(true);
  };

  const handleSelectRecord = (record: ExecutionRecordRef) => {
    setSelectedRecord(record);
    setView("exceptions");
    setMobileShowDetail(true);
  };

  // 移动端返回当前分类的列表。
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

  // 桌面与移动端共用列表，保留切换前的阅读位置。
  const listContent = <TaskHistoryList runningTasks={runningTasks} completedTasks={completedTasks}
    visibleConversations={visibleConversations} selected={selected} onSelect={handleSelect}
    historyLoading={historyLoading} listEndRef={listEndRef} />;

  // 普通任务和异常详情分别保留，切换分类时不重建时间线。
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

  return (
    <Dialog open={open} onOpenChange={setOpen} onOpenChangeComplete={nextOpen => {
      if (!nextOpen) {
        onClose();
      }
    }}>
      <DialogContent showCloseButton={false} aria-describedby={undefined}
        className="flex h-full max-h-[calc(100vh-2rem)] w-full flex-col gap-0 overflow-hidden rounded-2xl border border-border/40 p-0 sm:max-w-[92vw] md:h-[75vh] md:w-[1400px] motion-reduce:animate-none">
        {/* 面板标题与操作 */}
        <div className="flex items-center justify-between px-4 md:px-5 py-3 md:py-3.5 border-b border-border/20 shrink-0">
          <div className="flex items-center gap-2">
            {/* 移动端详情返回入口 */}
            {mobileShowDetail && (
              <Button variant="ghost" size="icon" aria-label={view === "exceptions" ? "返回异常记录" : "返回任务列表"} onClick={handleMobileBack} className="md:hidden">
                <ArrowLeft />
              </Button>
            )}
            <DialogTitle className="text-base font-semibold">运行与通知</DialogTitle>
          </div>
          <div className="flex items-center gap-2">
            {view === "tasks" && completedTasks.length > 0 && (
              <Button variant="ghost" onClick={clearCompleted} className="hidden md:block">
                清除已完成
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={handleClose} aria-label="关闭任务详情"><X /></Button>
          </div>
        </div>

        {/* 两种屏幕尺寸共用时间线实例。 */}
        <div className="flex flex-1 min-h-0">
          <div
            data-sidebar-navigation="runs"
            className={cn(
              "w-full min-h-0 flex-col overflow-hidden md:flex md:w-72 md:shrink-0 md:border-r md:border-border/20",
              mobileShowDetail ? "hidden" : "flex"
            )}
          >
            <NotificationViews view={view} onViewChange={setView} selectedRecord={selectedRecord} onSelectRecord={handleSelectRecord}>
              {listContent}
            </NotificationViews>
          </div>
          <div
            className={cn(
              "relative flex-1 min-w-0 min-h-0 bg-muted/10 md:block",
              mobileShowDetail ? "block" : "hidden"
            )}
          >
            {/* 保留时间线的测量尺寸，避免切换分类后虚拟列表重置阅读位置。 */}
            <div aria-hidden={view !== "tasks"} inert={view !== "tasks"}
              className={cn("absolute inset-0", view !== "tasks" && "invisible")} aria-label="任务详情">
              {detailContent}
            </div>
            <div aria-hidden={view !== "exceptions"} inert={view !== "exceptions"}
              className={cn("absolute inset-0", view !== "exceptions" && "invisible")} aria-label="异常详情">
              {selectedRecord ? <ExecutionRecordDetailPanel key={`${selectedRecord.kind}:${selectedRecord.id}`} target={selectedRecord} />
                : <div className="flex h-full items-center justify-center text-sm text-muted-foreground">选择一条异常查看详情</div>}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
