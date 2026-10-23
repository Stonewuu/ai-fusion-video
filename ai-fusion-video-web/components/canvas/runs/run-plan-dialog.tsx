"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useEffect, useState } from "react";
import { Loader2, Play, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import type { CanvasRunBudget, CanvasRunMode } from "@/lib/api/canvas-runs";
import { defaultRunBudget, runModes } from "@/lib/canvas/run-state";
import { useCanvasRuns } from "./canvas-runs-provider";
import { CanvasSelect } from "../forms/canvas-select";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasGuidance } from "../shared/canvas-guidance";
import { userFacingError } from "@/lib/user-facing-error";
import {
  runUsageSummary,
  usesExternalCreation,
} from "@/lib/canvas/run-presentation";
import { CanvasPlanInputs } from "./plan-inputs";

export interface CanvasRunDraft {
  nodeIds: string[];
  mode: CanvasRunMode;
  reuse: Record<string, string>;
}
const limits: {
  key: keyof CanvasRunBudget;
  label: string;
  min: number;
  max: number;
}[] = [
  { key: "maxNodes", label: "最多执行次数（含子项与重试）", min: 1, max: 1000 },
  { key: "maxImages", label: "最多生成图片", min: 0, max: 1000 },
  { key: "maxVideoSeconds", label: "最多视频秒数", min: 0, max: 36000 },
  { key: "maxParallel", label: "同时执行任务", min: 1, max: 8 },
  { key: "maxBatchItems", label: "最多批量对象", min: 1, max: 1000 },
];
export function CanvasRunPlanDialog({
  draft,
  onClose,
  onStarted,
}: {
  draft: CanvasRunDraft;
  onClose: () => void;
  onStarted: () => void;
}) {
  const runtime = useCanvasRuns();
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const [mode, setMode] = useState(draft.mode);
  const [reuse, setReuse] = useState(draft.reuse);
  const [budget, setBudget] = useState(defaultRunBudget);
  const [dirty, setDirty] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const plan = runtime.plan;
  const externalCreation =
    plan && usesExternalCreation(plan, snapshot.graph.nodes);
  const changed = plan && plan.documentRevision !== snapshot.document.revision;
  const ready =
    plan &&
    !dirty &&
    !changed &&
    !plan.issues.length &&
    Date.parse(plan.expiresAt) > now;
  const preview = async () => {
    if (await runtime.preview(mode, draft.nodeIds, reuse, budget))
      setDirty(false);
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !runtime.busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>确认运行范围</DialogTitle>
          <DialogDescription>
            使用已保存的内容执行。生成结果保留在运行记录中，可预览后放入画布。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-4">
          <div className="flex flex-col gap-2">
            <Label>运行哪些节点</Label>
            <CanvasSelect
              label="运行范围"
              value={mode}
              disabled={runtime.busy}
              items={runModes.filter((item) =>
                draft.nodeIds.length
                  ? item.value !== "SINGLE" || draft.nodeIds.length === 1
                  : item.value === "ALL" || item.value === "STALE",
              )}
              onChange={(value) => {
                setMode(value as CanvasRunMode);
                setDirty(true);
              }}
              placeholder="选择运行范围"
            />
          </div>
          {plan && dirty && (
            <CanvasGuidance title="设置已改变，请重新检查" tone="warning">
              下方是上次检查的结果。点击“检查运行计划”后，范围与用量才会更新。
            </CanvasGuidance>
          )}
          {plan && (
            <div className="space-y-3">
              {!plan.issues.length && (
                <CanvasGuidance
                  title={runUsageSummary(plan.cost)}
                  tone={externalCreation ? "action" : "info"}
                >
                  {externalCreation ? (
                    <>
                      本次会调用所选服务。
                      {plan.cost.amount === "未配置"
                        ? "费用以服务商计费为准。"
                        : `预计费用：${plan.cost.amount}。`}
                      需要保留已有结果时，可在下方选择“使用历史结果”。
                    </>
                  ) : (
                    "本次使用已有内容，不会重新生成素材。"
                  )}
                </CanvasGuidance>
              )}
              {plan.cost.batchItems > 0 && (
                <p className="text-xs text-muted-foreground">
                  包含最多 {plan.cost.batchItems}{" "}
                  个批量子项，以上数量按处理上限估算；仅实际收到的项目会执行。
                </p>
              )}
              {plan.cost.agentRuns > 0 ? (
                <p className="text-xs text-muted-foreground">
                  包含最多 {plan.cost.agentRuns}{" "}
                  个智能体任务，主任务与子任务共用上面的生成预算。
                </p>
              ) : null}
              {plan.cost.mcpCalls > 0 ? (
                <p className="text-sm font-medium">
                  包含最多 {plan.cost.mcpCalls} 次外部 MCP
                  调用，请核对下方操作影响。
                </p>
              ) : null}
              {plan.cost.agentRuns > 0 && (
                <p className="text-sm text-muted-foreground">
                  智能体的读取、编辑和生成自动批准，主任务与子任务沿用同一策略。
                </p>
              )}
              <ol className="divide-y divide-border/20 rounded-lg border border-border/20 bg-background/70">
                {plan.nodes.map((node) => {
                  const previous = runtime.run?.nodes.find(
                    (item) =>
                      item.nodeId === node.nodeId &&
                      item.parentNodeRunId === null &&
                      item.status === "SUCCEEDED",
                  );
                  return (
                    <li
                      key={node.nodeId}
                      className="flex items-center justify-between gap-3 p-3"
                    >
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">
                          {node.title}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {reuse[node.nodeId]
                            ? "使用已选的历史结果"
                            : node.imageCount
                              ? `${node.kind === "foreach" ? "最多" : "新"}生成 ${node.imageCount} 张图片`
                              : node.videoSeconds
                                ? `${node.kind === "foreach" ? "最多" : "新"}生成 ${node.videoSeconds} 秒视频`
                                : node.kind === "selection"
                                  ? "等待你选择候选"
                                  : node.kind === "foreach"
                                    ? "逐项执行并按顺序汇总"
                                    : node.kind === "writeback"
                                      ? "预览差异并等待确认写入"
                                      : "执行此节点"}
                        </p>
                        {node.externalOperation ? (
                          <p className="mt-1 text-xs text-muted-foreground">
                            {node.externalOperation.name} ·{" "}
                            {node.externalOperation.serverName} ·{" "}
                            {node.externalOperation.effect === "READ_ONLY"
                              ? "只读取外部内容"
                              : node.externalOperation.effect === "DESTRUCTIVE"
                                ? "可能删除或覆盖外部内容"
                                : "将修改外部内容"}{" "}
                            · 最多 {node.externalOperation.maximumCalls} 次
                          </p>
                        ) : null}
                        {node.action !== "REUSE" && !reuse[node.nodeId] && (
                          <CanvasPlanInputs
                            nodeId={node.nodeId}
                            dependencies={node.dependencies}
                            graph={snapshot.graph}
                          />
                        )}
                      </div>
                      {reuse[node.nodeId] ? (
                        <Button
                          size="xs"
                          variant="ghost"
                          disabled={runtime.busy}
                          onClick={() => {
                            setReuse((current) => {
                              const next = { ...current };
                              delete next[node.nodeId];
                              return next;
                            });
                            setDirty(true);
                          }}
                        >
                          改为重新执行
                        </Button>
                      ) : previous ? (
                        <Button
                          size="xs"
                          variant="outline"
                          disabled={runtime.busy}
                          onClick={() => {
                            setReuse((current) => ({
                              ...current,
                              [node.nodeId]: previous.id,
                            }));
                            setDirty(true);
                          }}
                        >
                          <RotateCcw />
                          使用历史结果
                        </Button>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
              {!!plan.issues.length && (
                <CanvasGuidance
                  title={`还需完成 ${plan.issues.length} 项设置`}
                  tone="warning"
                >
                  <ul className="space-y-2" aria-label="运行前需要处理的问题">
                    {plan.issues.map((issue, index) => (
                      <li key={index} className="text-sm text-destructive">
                        {issue.nodeId && (
                          <Button
                            variant="link"
                            size="xs"
                            onClick={() => {
                              store.getState().focus([issue.nodeId!]);
                              onClose();
                            }}
                          >
                            {snapshot.graph.nodes.find(
                              (node) => node.id === issue.nodeId,
                            )?.title ?? "定位节点"}
                          </Button>
                        )}{" "}
                        {userFacingError(issue.message)}
                      </li>
                    ))}
                  </ul>
                </CanvasGuidance>
              )}
            </div>
          )}
          <details className="space-y-3">
            <summary className="cursor-pointer text-sm text-muted-foreground">
              设置运行上限
            </summary>
            <div className="grid grid-cols-2 gap-3">
              {limits.map((item) => (
                <div key={item.key} className="flex flex-col gap-2">
                  <Label htmlFor={`run-${item.key}`}>{item.label}</Label>
                  <Input
                    id={`run-${item.key}`}
                    type="number"
                    min={item.min}
                    max={item.max}
                    disabled={runtime.busy}
                    value={budget[item.key]}
                    onChange={(event) => {
                      setBudget((current) => ({
                        ...current,
                        [item.key]: Number(event.target.value),
                      }));
                      setDirty(true);
                    }}
                  />
                </div>
              ))}
            </div>
          </details>
          {changed && (
            <p role="status" className="text-sm text-destructive">
              画布已更新，请重新检查运行计划。
            </p>
          )}
          {runtime.pending && (
            <p role="status" className="text-sm text-muted-foreground">
              上次运行的启动状态尚不明确，请到“运行”面板确认后再继续。
            </p>
          )}
          {runtime.error && (
            <p role="alert" className="text-sm text-destructive">
              {runtime.error}
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={runtime.busy} onClick={onClose}>
            返回画布
          </Button>
          {!ready ? (
            <Button
              disabled={
                runtime.busy ||
                runtime.loading ||
                !!runtime.pending ||
                runtime.corrupt
              }
              onClick={() => void preview()}
            >
              {runtime.busy && (
                <Loader2 className="animate-spin motion-reduce:animate-none" />
              )}
              检查运行计划
            </Button>
          ) : (
            <Button
              disabled={runtime.busy || !!runtime.pending || runtime.corrupt}
              onClick={() =>
                void runtime.start().then((ok) => {
                  if (ok) onStarted();
                })
              }
            >
              {runtime.busy ? (
                <Loader2 className="animate-spin motion-reduce:animate-none" />
              ) : (
                <Play />
              )}
              开始运行
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
