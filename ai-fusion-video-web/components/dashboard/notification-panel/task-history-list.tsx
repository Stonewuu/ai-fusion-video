"use client";

import type { RefObject } from "react";
import { Ban, CheckCircle2, Clock, Loader2, MessageSquare, XCircle } from "lucide-react";
import type { AgentConversation } from "@/lib/api/ai-assistant";
import type { PipelineTask } from "@/lib/store/pipeline-store";
import { TaskListItem } from "./task-list-items";
import { getAgentTypeName } from "./constants";
import { ElapsedText } from "./hooks";
import { formatElapsed, getElapsedStr } from "./utils";

export type TaskSelection = { type: "pipeline"; taskId: string } | { type: "history"; conversation: AgentConversation };

export function TaskHistoryList({ runningTasks, completedTasks, visibleConversations, selected, onSelect: handleSelect, historyLoading, listEndRef }: {
  runningTasks: PipelineTask[];
  completedTasks: PipelineTask[];
  visibleConversations: AgentConversation[];
  selected: TaskSelection | null;
  onSelect: (item: TaskSelection) => void;
  historyLoading: boolean;
  listEndRef: RefObject<HTMLDivElement | null>;
}) {
  return (
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
}
