"use client";

import { useEffect, useState } from "react";
import { Bot, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  canvasAgentApi,
  type CanvasAgentExecution,
} from "@/lib/api/canvas-agents";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useCanvasRuns } from "../runs/canvas-runs-provider";
import { CanvasAgentTaskProgress } from "./task-progress";
import { CanvasAgentModelRecovery } from "./model-recovery";
import { CanvasAgentGenerationProgress } from "./generation-progress";
import { canvasAgentExecutionFinished } from "@/lib/canvas/agent-execution";

export function CanvasAgentExecutionButton({
  runId,
  nodeId,
  attention,
  interrupted,
}: {
  runId: string;
  nodeId: string;
  attention: boolean;
  interrupted: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        size="xs"
        variant={attention || interrupted ? "default" : "outline"}
        onClick={() => setOpen(true)}
      >
        <Bot />
        {attention
          ? "确认智能体操作"
          : interrupted
            ? "查看中断详情"
            : "查看智能体任务"}
      </Button>
      {open ? (
        <CanvasAgentExecutionDialog
          runId={runId}
          nodeId={nodeId}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </>
  );
}

function CanvasAgentExecutionDialog({
  runId,
  nodeId,
  onClose,
}: {
  runId: string;
  nodeId: string;
  onClose: () => void;
}) {
  const runtime = useCanvasRuns();
  const [data, setData] = useState<CanvasAgentExecution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const load = async () => {
      let stopped = false;
      try {
        const result = await canvasAgentApi.get(
          runId,
          nodeId,
          controller.signal,
        );
        if (!active) return;
        setData(result);
        setError(null);
        stopped = canvasAgentExecutionFinished(result);
      } catch (failure) {
        if (!active) return;
        setError(getApiErrorMessage(failure));
        if ([401, 403, 404].includes(getApiErrorStatus(failure) ?? 0)) {
          setData(null);
          stopped = true;
        }
      } finally {
        if (active && !stopped) timer = setTimeout(() => void load(), 3000);
      }
    };
    void load();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [runId, nodeId, refresh]);
  const changed = () => {
    setRefresh((value) => value + 1);
    runtime.reload();
  };
  return (
    <Dialog
      open
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>智能体任务</DialogTitle>
          <DialogDescription>
            {data
              ? `已启动 ${data.tasks.length} / ${data.maxAgentRuns} 个任务。${
                  data.unresolvedModelCalls > 0
                    ? "请在下方处理未完成的调用。"
                    : canvasAgentExecutionFinished(data)
                      ? "本次任务已结束，已有结果保留。"
                      : "画布计划内的操作自动批准，关闭窗口后任务继续执行。"
                }`
              : "读取主任务与子任务状态"}
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="flex flex-col gap-3">
          {error ? (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          ) : null}
          {!data && !error ? (
            <p role="status" className="text-sm text-muted-foreground">
              正在读取任务…
            </p>
          ) : null}
          {data && data.unresolvedModelCalls > 0 ? (
            <CanvasAgentModelRecovery
              requestIds={data.unresolvedModelRequestIds}
              onChanged={changed}
            />
          ) : null}
          {data && (
            <CanvasAgentGenerationProgress
              generations={data.generations}
              onChanged={changed}
            />
          )}
          {data?.tasks.map((task) => (
            <CanvasAgentTaskProgress
              key={task.id}
              task={task}
              runId={runId}
              nodeId={nodeId}
              canControl={data.canControl}
              onChanged={changed}
            />
          ))}
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="ghost" onClick={changed}>
            <RefreshCw />
            刷新
          </Button>
          <Button variant="outline" onClick={onClose}>
            关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
