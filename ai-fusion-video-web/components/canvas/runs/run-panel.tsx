"use client";

import { useState } from "react";
import {
  ChartColumn,
  CheckCircle2,
  CircleAlert,
  Loader2,
  Play,
  RefreshCw,
  Square,
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
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasPicker } from "../ui/canvas-picker";
import { runFinished, runLabels } from "@/lib/canvas/run-state";
import { useCanvasStoreApi } from "../canvas-provider";
import { useCanvasRuns } from "./canvas-runs-provider";
import type { CanvasRunDraft } from "./run-plan-dialog";
import { CanvasRunNode } from "./run-node";
import { CanvasBatchResults } from "./batch-results";
import { CanvasRunStatistics } from "./run-statistics";
import { CanvasActivityDialog } from "./activity-dialog";
import { CanvasGuidance } from "../shared/canvas-guidance";
import {
  runHistoryOptions,
  runUsageSummary,
} from "@/lib/canvas/run-presentation";
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
  const firstFailure = run?.nodes.find(
    (node) => node.status === "FAILED" && node.error,
  );
  const recovery = run ? canvasRecoveryPlan(run) : null;
  const continueFrom = (nodeId: string, nodeRunId: string) => {
    const graph = store.getState().snapshot?.graph;
    if (!graph) {
      return;
    }
    const selected = new Set<string>([nodeId]);
    const queue = [nodeId];
    while (queue.length) {
      const current = queue.shift();
      for (const edge of graph.edges) {
        if (
          edge.kind === "execution" &&
          edge.sourceNodeId === current &&
          !selected.has(edge.targetNodeId)
        ) {
          selected.add(edge.targetNodeId);
          queue.push(edge.targetNodeId);
        }
      }
    }
    onPreview({
      mode: "DEPENDENCIES",
      nodeIds: [...selected],
      reuse: { [nodeId]: nodeRunId },
    });
  };
  return (
    <div className="flex flex-col gap-3 p-3.5">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-[13px] font-medium">运行与结果</h2>
        <div className="flex items-center gap-1">
          <CanvasButton
            variant="ghost"
            size="sm"
            onClick={() => setActivityOpen(true)}
          >
            <ChartColumn />
            累计统计
          </CanvasButton>
          <CanvasIconButton
            size="icon-sm"
            label="刷新运行状态"
            disabled={runtime.busy}
            onClick={runtime.reload}
          >
            <RefreshCw />
          </CanvasIconButton>
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
          <div className="flex flex-wrap gap-1.5">
            <CanvasButton
              variant="generate"
              size="sm"
              disabled={runtime.busy}
              onClick={() => void runtime.confirm()}
            >
              继续这次运行
            </CanvasButton>
            <CanvasButton
              variant="secondary"
              size="sm"
              disabled={runtime.busy}
              onClick={() => void runtime.withdraw()}
            >
              撤回启动
            </CanvasButton>
          </div>
        </CanvasGuidance>
      )}
      {runtime.corrupt && (
        <div className="flex flex-col items-start gap-2 rounded-[12px] border border-destructive/20 bg-destructive/5 p-3">
          <p className="text-xs text-destructive">
            本地启动记录无法读取，请先核对下方运行历史。
          </p>
          <CanvasButton
            variant="secondary"
            size="sm"
            disabled={runtime.busy}
            onClick={() => setConfirm("clear")}
          >
            清除本机记录
          </CanvasButton>
        </div>
      )}
      {runtime.error && (
        <p
          role="alert"
          className="rounded-[10px] border border-destructive/20 bg-destructive/5 px-2.5 py-2 text-xs text-destructive"
        >
          {runtime.error}
        </p>
      )}
      {runtime.loading ? (
        <p role="status" className="text-xs text-muted-foreground">
          正在读取运行记录…
        </p>
      ) : (
        <>
          {(!!run || !!runtime.history.length) && (
            <div className="flex flex-col gap-1.5 p-0.5">
              <CanvasPicker
                label="查看运行记录"
                placeholder="选择一次运行"
                searchPlaceholder="搜索运行记录…"
                side="bottom"
                value={run?.id ?? null}
                items={runHistoryOptions(runtime.history, run)}
                onChange={runtime.chooseRun}
              />
              {(runtime.historyPage > 0 || runtime.history.length >= 20) && (
                <div className="flex justify-between gap-2">
                  <CanvasButton
                    size="sm"
                    variant="ghost"
                    disabled={runtime.historyPage === 0}
                    onClick={() => runtime.page(runtime.historyPage - 1)}
                  >
                    较新记录
                  </CanvasButton>
                  <CanvasButton
                    size="sm"
                    variant="ghost"
                    disabled={runtime.history.length < 20}
                    onClick={() => runtime.page(runtime.historyPage + 1)}
                  >
                    更早记录
                  </CanvasButton>
                </div>
              )}
            </div>
          )}
          {!run ? (
            <div className="flex flex-col items-center gap-2.5 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-8 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
              <span className="grid size-9 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
                <Play className="size-4" aria-hidden />
              </span>
              <p className="text-xs leading-relaxed text-muted-foreground">
                运行节点，实时跟踪进度并查看生成产物。
              </p>
              <CanvasButton
                variant="generate"
                size="sm"
                onClick={() => onPreview()}
                disabled={runtime.busy || !!runtime.pending}
              >
                <Play className="size-3 fill-current" />
                预览运行计划
              </CanvasButton>
            </div>
          ) : (
            <>
              <div className="flex flex-col gap-2.5 rounded-[12px] border border-foreground/[0.08] bg-background/75 p-3 dark:border-white/[0.09] dark:bg-white/[0.035]">
                <div className="flex items-center gap-2" aria-live="polite">
                  {run.status === "SUCCEEDED" ? (
                    <CheckCircle2 className="size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
                  ) : [
                    "FAILED",
                    "PARTIAL_FAILED",
                    "NEEDS_REVIEW",
                    "UNKNOWN_SUBMISSION",
                  ].includes(run.status) ? (
                    <CircleAlert className="size-4 shrink-0 text-destructive" />
                  ) : !runFinished(run.status) ? (
                    <Loader2 className="size-4 shrink-0 animate-spin text-primary motion-reduce:animate-none" />
                  ) : null}
                  <strong className="text-[13px] font-medium">
                    {runLabels[run.status]}
                  </strong>
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
                  <CanvasGuidance
                    title={`${firstFailure.title} · 执行失败`}
                    tone="error"
                  >
                    <p className="break-words text-xs leading-relaxed">
                      {userFacingError(firstFailure.error)}
                    </p>
                  </CanvasGuidance>
                )}
                {run.canControl && !runFinished(run.status) && (
                  <div>
                    <CanvasButton
                      variant="secondary"
                      size="sm"
                      disabled={runtime.busy || run.cancelRequested}
                      onClick={() => setConfirm("cancel")}
                    >
                      <Square className="size-3" />
                      {run.cancelRequested
                        ? "正在等待已提交任务结束"
                        : "停止此运行"}
                    </CanvasButton>
                  </div>
                )}
                {runFinished(run.status) && (
                  <div className="flex flex-col gap-2">
                    <div className="flex flex-wrap gap-1.5">
                      {recovery && (
                        <CanvasButton
                          variant="generate"
                          size="sm"
                          disabled={runtime.busy || !!runtime.pending}
                          onClick={() => onPreview(recovery)}
                        >
                          <Play className="size-3 fill-current" />
                          继续未完成步骤
                        </CanvasButton>
                      )}
                      <CanvasButton
                        variant={recovery ? "secondary" : "generate"}
                        size="sm"
                        disabled={runtime.busy || !!runtime.pending}
                        onClick={() => onPreview()}
                      >
                        {!recovery && <Play className="size-3 fill-current" />}
                        {recovery ? "选择其他运行范围" : "再次运行"}
                      </CanvasButton>
                    </div>
                    {recovery && (
                      <p className="text-[11px] leading-relaxed text-muted-foreground">
                        先预览新计划；未变化的成功结果会复用，其余步骤按当前内容计算。
                        {run.nodes.some(
                          (node) =>
                            node.parentNodeRunId === null &&
                            node.status === "PARTIAL_FAILED",
                        ) && "部分完成的步骤会整体重新执行，请在计划中核对数量。"}
                      </p>
                    )}
                  </div>
                )}
              </div>
              <CanvasRunStatistics value={run.statistics} />
              <div className="flex flex-col gap-2.5">
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
          if (!open && !runtime.busy) {
            setConfirm(null);
          }
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
                  if (ok) {
                    setConfirm(null);
                  }
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
