"use client";
import { userFacingError } from "@/lib/user-facing-error";

import { useState } from "react";
import { Bot, CheckCircle2, CircleAlert, Clock3, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ToolCallCard } from "@/components/dashboard/shared/tool-call-card";
import { canvasAgentApi, type CanvasAgentTask } from "@/lib/api/canvas-agents";
import { getApiErrorMessage } from "@/lib/api/api-error";

const labels: Record<string, string> = {
  RUNNING: "执行中",
  WAITING_CONFIRMATION: "等待确认",
  WAITING_EXTERNAL: "等待任务结果",
  CANCEL_REQUESTED: "正在停止",
  COMPLETED: "已完成",
  FAILED: "未完成",
  CANCELLED: "已停止",
};

export function CanvasAgentTaskProgress({
  task,
  runId,
  nodeId,
  canControl,
  onChanged,
}: {
  task: CanvasAgentTask;
  runId: string;
  nodeId: string;
  canControl: boolean;
  onChanged: () => void;
}) {
  const [selection, setSelection] = useState<{
    replyId: string;
    decisions: Record<string, boolean>;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const confirmation = task.confirmation;
  const decisions =
    confirmation && selection && confirmation.replyId === selection.replyId
      ? selection.decisions
      : {};
  const decide = async (next: Record<string, boolean>) => {
    if (!confirmation || busy || !canControl) return;
    setSelection({ replyId: confirmation.replyId, decisions: next });
    if (
      !confirmation.tools.every((tool) => Object.hasOwn(next, tool.toolCallId))
    )
      return;
    setBusy(true);
    setError(null);
    try {
      await canvasAgentApi.confirm(runId, nodeId, {
        runId: task.id,
        replyId: confirmation.replyId,
        decisions: confirmation.tools.map((tool) => ({
          toolCallId: tool.toolCallId,
          approved: next[tool.toolCallId],
        })),
      });
      onChanged();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      onChanged();
    } finally {
      setBusy(false);
    }
  };
  const Icon =
    task.status === "COMPLETED"
      ? CheckCircle2
      : task.status === "FAILED"
        ? CircleAlert
        : task.status === "WAITING_CONFIRMATION" ||
            task.status === "WAITING_EXTERNAL"
          ? Clock3
          : task.status === "RUNNING"
            ? Loader2
            : Bot;
  return (
    <section className="flex flex-col gap-3 border-t border-border/20 py-4">
      <div className="flex items-start gap-2">
        <Icon
          className={`mt-0.5 size-4 shrink-0 ${task.status === "RUNNING" ? "animate-spin text-primary motion-reduce:animate-none" : task.status === "FAILED" ? "text-destructive" : "text-primary"}`}
        />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium">{task.name}</h3>
          <p className="text-xs text-muted-foreground">
            {task.parentId ? "子任务" : "主任务"} ·{" "}
            {new Date(task.startedAt).toLocaleTimeString("zh-CN", {
              hour12: false,
            })}
          </p>
        </div>
        <Badge variant="outline">{labels[task.status] ?? task.status}</Badge>
      </div>
      {task.error ? (
        <p className="whitespace-pre-wrap break-words text-sm text-destructive">
          {userFacingError(task.error)}
        </p>
      ) : null}
      {task.summary ? (
        <details>
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
            查看任务结果{task.summaryTruncated ? "（前 8000 字）" : ""}
          </summary>
          <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-relaxed">
            {task.summary}
          </p>
        </details>
      ) : null}
      {confirmation ? (
        <div className="flex flex-col gap-3">
          <p className="text-xs text-muted-foreground">
            该任务仍保留此前待确认的操作。处理后按画布自动批准策略继续执行。
          </p>
          {confirmation.tools.map((tool) => (
            <ToolCallCard
              key={`${confirmation.replyId}:${tool.toolCallId}`}
              items={[
                {
                  toolCallId: tool.toolCallId,
                  toolName: tool.toolName,
                  argumentsText: tool.arguments,
                  status: "awaiting_approval",
                },
              ]}
              approval={{
                review: tool.review,
                decision: Object.hasOwn(decisions, tool.toolCallId)
                  ? decisions[tool.toolCallId]
                    ? "approve"
                    : "deny"
                  : undefined,
                submitting: busy,
                showActions: canControl,
                expiresAt: confirmation.expiresAt,
                onApprove: () =>
                  void decide({ ...decisions, [tool.toolCallId]: true }),
                onReject: () =>
                  void decide({ ...decisions, [tool.toolCallId]: false }),
              }}
            />
          ))}
          {confirmation.tools.length > 1 && canControl ? (
            <div className="flex flex-wrap justify-end gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={
                  busy || Date.parse(confirmation.expiresAt) <= Date.now()
                }
                onClick={() =>
                  void decide(
                    Object.fromEntries(
                      confirmation.tools.map((tool) => [
                        tool.toolCallId,
                        false,
                      ]),
                    ),
                  )
                }
              >
                全部拒绝
              </Button>
              <Button
                size="sm"
                disabled={
                  busy || Date.parse(confirmation.expiresAt) <= Date.now()
                }
                onClick={() =>
                  void decide(
                    Object.fromEntries(
                      confirmation.tools.map((tool) => [tool.toolCallId, true]),
                    ),
                  )
                }
              >
                全部允许
              </Button>
            </div>
          ) : null}
          {error ? (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          ) : null}
        </div>
      ) : task.status === "WAITING_CONFIRMATION" ? (
        <p className="text-xs text-muted-foreground">等待发起人确认此任务。</p>
      ) : null}
    </section>
  );
}
