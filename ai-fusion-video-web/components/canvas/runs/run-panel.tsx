"use client";
import { useState } from "react";
import {
  CheckCircle2,
  CircleAlert,
  Loader2,
  Play,
  RefreshCw,
  Square,
  ChartColumn,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { runFinished, runLabels } from "@/lib/canvas/run-state";
import { useCanvasStoreApi } from "../canvas-provider";
import { CanvasSelect } from "../forms/canvas-select";
import { useCanvasRuns } from "./canvas-runs-provider";
import type { CanvasRunDraft } from "./run-plan-dialog";

import { CanvasRunNode } from "./run-node";
import { CanvasBatchResults } from "./batch-results";
import { CanvasRunStatistics } from "./run-statistics";
import { CanvasActivityDialog } from "./activity-dialog";
import { CanvasGuidance } from "../shared/canvas-guidance";
import { runHistoryOptions, runUsageSummary } from "@/lib/canvas/run-presentation";
import { userFacingError } from "@/lib/user-facing-error";
import { canvasRecoveryPlan } from "@/lib/canvas/run-recovery";

export function CanvasRunPanel({
  onPreview,
}: {
  onPreview: (draft?: CanvasRunDraft) => void;
}) {
  const runtime = useCanvasRuns();
  const store = useCanvasStoreApi();
  const [confirm, setConfirm] = useState<"cancel" | "clear" | null>(null);
  const [activityOpen, setActivityOpen] = useState(false);
  const run = runtime.run;
  const firstFailure = run?.nodes.find((node) => node.status === "FAILED" && node.error);
  const recovery = run ? canvasRecoveryPlan(run) : null;
  const continueFrom = (nodeId: string, nodeRunId: string) => {
    const graph = store.getState().snapshot?.graph;
    if (!graph) return;
    const selected = new Set<string>([nodeId]);
    const queue = [nodeId];
    while (queue.length) {
      const current = queue.shift();
      for (const edge of graph.edges)
        if (
          edge.kind === "execution" &&
          edge.sourceNodeId === current &&
          !selected.has(edge.targetNodeId)
        ) {
          selected.add(edge.targetNodeId);
          queue.push(edge.targetNodeId);
        }
    }
    onPreview({
      mode: "DEPENDENCIES",
      nodeIds: [...selected],
      reuse: { [nodeId]: nodeRunId },
    });
  };
  return (
    <div className="space-y-5 p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">运行与结果</h2>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="default"
            onClick={() => setActivityOpen(true)}
          >
            <ChartColumn />
            累计统计
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="刷新运行状态"
            disabled={runtime.busy}
            onClick={runtime.reload}
          >
            <RefreshCw />
          </Button>
        </div>
      </div>
      {activityOpen && (
        <CanvasActivityDialog
          canvasId={store.getState().snapshot!.document.id}
          onClose={() => setActivityOpen(false)}
        />
      )}
      {runtime.pending && (
        <CanvasGuidance title="上次运行的启动状态待确认" tone="warning">
          <p className="text-xs text-muted-foreground">
            {runUsageSummary(runtime.pending.cost)}
            。继续会找回原任务；若尚未启动，将沿用已确认的设置启动。
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              disabled={runtime.busy}
              onClick={() => void runtime.confirm()}
            >
              继续这次运行
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={runtime.busy}
              onClick={() => void runtime.withdraw()}
            >
              撤回启动
            </Button>
          </div>
        </CanvasGuidance>
      )}
      {runtime.corrupt && (
        <div className="space-y-2">
          <p className="text-sm text-destructive">
            本地启动记录无法读取，请先核对下方运行历史。
          </p>
          <Button
            variant="outline"
            size="sm"
            disabled={runtime.busy}
            onClick={() => setConfirm("clear")}
          >
            清除本机记录
          </Button>
        </div>
      )}
      {runtime.error && (
        <p role="alert" className="text-sm text-destructive">
          {runtime.error}
        </p>
      )}
      {runtime.loading ? (
        <p role="status" className="text-sm text-muted-foreground">
          正在读取运行记录…
        </p>
      ) : (
        <>
          {(!!run || !!runtime.history.length) && (
            <div className="space-y-2">
              <CanvasSelect
                label="查看运行记录"
                placeholder="选择一次运行"
                value={run?.id ?? null}
                items={runHistoryOptions(runtime.history, run)}
                onChange={runtime.chooseRun}
              />
              <div className="flex justify-between gap-2">
                <Button
                  size="xs"
                  variant="ghost"
                  disabled={runtime.historyPage === 0}
                  onClick={() => runtime.page(runtime.historyPage - 1)}
                >
                  较新记录
                </Button>
                <Button
                  size="xs"
                  variant="ghost"
                  disabled={runtime.history.length < 20}
                  onClick={() => runtime.page(runtime.historyPage + 1)}
                >
                  更早记录
                </Button>
              </div>
            </div>
          )}
          {!run ? (
            <div className="space-y-3 py-4">
              <p className="text-sm text-muted-foreground">
                运行节点，查看结果并继续创作。
              </p>
              <Button
                onClick={() => onPreview()}
                disabled={runtime.busy || !!runtime.pending}
              >
                <Play />
                预览运行计划
              </Button>
            </div>
          ) : (
            <>
              <div className="space-y-3">
                <div className="flex items-center gap-2" aria-live="polite">
                  {run.status === "SUCCEEDED" ? (
                    <CheckCircle2 className="size-4 text-primary" />
                  ) : ["FAILED", "PARTIAL_FAILED", "NEEDS_REVIEW", "UNKNOWN_SUBMISSION"].includes(
                      run.status,
                    ) ? (
                    <CircleAlert className="size-4 text-destructive" />
                  ) : !runFinished(run.status) ? (
                    <Loader2 className="size-4 animate-spin motion-reduce:animate-none" />
                  ) : null}
                  <strong className="text-sm">{runLabels[run.status]}</strong>
                </div>
                <p className="text-xs text-muted-foreground">
                  计划范围：{runUsageSummary(run.cost)}
                </p>
                {run.error && (
                  <CanvasGuidance title="这次运行需要处理" tone="error">
                    {userFacingError(run.error)}
                  </CanvasGuidance>
                )}
                {!run.error && firstFailure && (
                  <CanvasGuidance title={`${firstFailure.title} · 执行失败`} tone="error">
                    <p className="break-words text-xs leading-relaxed">
                      {userFacingError(firstFailure.error)}
                    </p>
                  </CanvasGuidance>
                )}
                {run.canControl && !runFinished(run.status) && (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={runtime.busy || run.cancelRequested}
                    onClick={() => setConfirm("cancel")}
                  >
                    <Square />
                    {run.cancelRequested
                      ? "正在等待已提交任务结束"
                      : "停止此运行"}
                  </Button>
                )}
                {runFinished(run.status) && (
                  <div className="flex flex-col gap-2">
                    <div className="flex flex-wrap gap-2">
                      {recovery && (
                        <Button
                          size="sm"
                          disabled={runtime.busy || !!runtime.pending}
                          onClick={() => onPreview(recovery)}
                        >
                          <Play />
                          继续未完成步骤
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={runtime.busy || !!runtime.pending}
                        onClick={() => onPreview()}
                      >
                        {recovery ? "选择其他运行范围" : "再次运行"}
                      </Button>
                    </div>
                    {recovery && (
                      <p className="text-xs text-muted-foreground">
                        先预览新计划；未变化的成功结果会复用，其余步骤与依赖按当前内容计算。
                        {run.nodes.some(
                          (node) =>
                            node.parentNodeRunId === null &&
                            node.status === "PARTIAL_FAILED",
                        ) &&
                          "部分完成的步骤会整体重新执行，请在计划中核对数量。"}
                      </p>
                    )}
                  </div>
                )}
              </div>
              <CanvasRunStatistics value={run.statistics} />
              <div className="space-y-4">
                {run.nodes
                  .filter((node) => node.parentNodeRunId === null)
                  .map((node) => (
                    <CanvasRunNode
                      key={node.id}
                      run={run}
                      node={node}
                      onContinue={continueFrom}
                    >
                      {node.kind === "foreach" && (
                        <CanvasBatchResults run={run} node={node} />
                      )}
                    </CanvasRunNode>
                  ))}
              </div>
            </>
          )}
        </>
      )}
      <Dialog
        open={!!confirm}
        onOpenChange={(open) => {
          if (!open && !runtime.busy) setConfirm(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {confirm === "cancel"
                ? "停止这次运行？"
                : "清理无法读取的本地记录？"}
            </DialogTitle>
            <DialogDescription>
              {confirm === "cancel"
                ? "停止后续步骤，并尝试取消正在处理的任务。已完成的结果会保留；服务方已产生的费用不会撤回。"
                : "请先核对运行历史；清除本机记录不会停止已经开始的运行。"}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={runtime.busy}
              onClick={() => setConfirm(null)}
            >
              返回
            </Button>
            <Button
              variant="destructive"
              disabled={runtime.busy}
              onClick={() =>
                void (
                  confirm === "cancel"
                    ? runtime.cancel()
                    : runtime.clearCorrupt()
                ).then((ok) => {
                  if (ok) setConfirm(null);
                })
              }
            >
              {confirm === "cancel" ? "停止运行" : "清理记录"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
