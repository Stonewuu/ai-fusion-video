"use client";

import { useState } from "react";
import { CircleAlert, CircleCheck, Loader2, Play, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { userFacingError } from "@/lib/user-facing-error";
import type { CanvasNodeActivity, CanvasNodeRun } from "@/lib/api/canvas-runs";
import type { CanvasCapability, CanvasNode } from "@/lib/canvas/types";
import { batchNodeProgress, canRunCanvasNode } from "@/lib/canvas/node-execution";
import { runFinished, runLabels } from "@/lib/canvas/run-state";
import { useCanvas } from "../canvas-provider";
import { useNodeGenerationProgress } from "../runs/use-node-generation-progress";
import { useCanvasNodeRunReady } from "../runs/canvas-runs-provider";

export function CanvasNodeRunButton({ node, capability, activity, starting, onRun }: {
  node: CanvasNode; capability: CanvasCapability; activity?: CanvasNodeActivity; starting: boolean; onRun: () => void;
}) {
  const allowed = useCanvas((state) => !!state.snapshot?.permissions.canExecute && !state.snapshot.document.archived);
  const syncing = useCanvas((state) => !!state.pending || state.saving || state.sync !== "connected");
  const ready = useCanvasNodeRunReady();
  if (!canRunCanvasNode(capability)) return null;
  const running = activity && !runFinished(activity.node.status);
  const waiting = activity && ["WAITING_SELECTION", "WAITING_CONFIRMATION", "UNKNOWN_SUBMISSION", "NEEDS_REVIEW"].includes(activity.node.status);
  const failed = activity && ["FAILED", "PARTIAL_FAILED", "BLOCKED", "CANCELLED"].includes(activity.node.status);
  const label = starting ? "提交中" : running ? runLabels[activity.node.status] : failed ? "重试" : "运行";
  const variant = capability.executor === "image" ? "image" : capability.executor === "video" ? "video" : "ai";
  return <Button variant={variant} size="sm" className="nodrag nopan shrink-0" aria-label={`${label}${node.title}`}
    title={!allowed ? "没有运行权限" : syncing ? "等待画布保存与同步" : running ? "此节点正在运行" : !ready ? "正在准备运行服务或处理上次提交" : "运行此节点及所需前置步骤，复用未变化的结果"}
    disabled={!allowed || syncing || !ready || starting || !!running} onPointerDown={(event) => event.stopPropagation()}
    onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}
    onClick={(event) => { event.stopPropagation(); onRun(); }}>
    {waiting ? <CircleAlert /> : starting || running ? <Loader2 className="animate-spin motion-reduce:animate-none" /> : failed ? <RotateCcw /> : <Play />}
    {starting || running ? null : label}
  </Button>;
}

export function CanvasNodeRunStatus({ activity, childrenRuns, starting, startError, onShow, onDismiss }: {
  activity?: CanvasNodeActivity; childrenRuns: readonly CanvasNodeRun[]; starting: boolean; startError: string | null;
  onShow: () => void; onDismiss: () => void;
}) {
  const generation = useNodeGenerationProgress(activity);
  const [expanded, setExpanded] = useState(false);
  const node = activity?.node;
  if (!node && !starting && !startError) return null;
  const active = !!node && !runFinished(node.status);
  const waiting = !!node && ["WAITING_SELECTION", "WAITING_CONFIRMATION", "UNKNOWN_SUBMISSION", "NEEDS_REVIEW"].includes(node.status);
  const error = startError ?? (node?.contentUnavailable ? node.error : node?.error ?? generation?.value?.error);
  const failure = !!error || !!node && ["FAILED", "PARTIAL_FAILED", "BLOCKED"].includes(node.status);
  const completed = node?.status === "SUCCEEDED" && !starting && !startError;
  const progress = node?.externalKind === "BATCH" ? batchNodeProgress(node, childrenRuns) : generation?.value;
  const label = starting ? "准备运行" : startError ? "未能启动" : node ? runLabels[node.status] : "";
  const message = error ? userFacingError(error) : null;
  return <section className={cn("canvas-node-run flex flex-col gap-2 border-b border-border/20 px-3 py-2 text-xs",
    failure ? "bg-destructive/5 text-destructive" : completed ? "bg-emerald-500/5 text-emerald-700 dark:text-emerald-300" : "text-muted-foreground")}
    aria-label="节点运行状态" data-run-state={startError ? "START_FAILED" : node?.status}>
    <div className="flex min-w-0 items-center justify-between gap-2">
      <span className="flex min-w-0 items-center gap-1.5" role="status">
        {starting || active && !waiting ? <Loader2 className="size-3.5 shrink-0 animate-spin motion-reduce:animate-none" />
          : failure || waiting ? <CircleAlert className="size-3.5 shrink-0" /> : completed ? <CircleCheck className="size-3.5 shrink-0" /> : null}
        {label}
      </span>
      <div className="nodrag nopan flex shrink-0 gap-2" onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
        {activity && <Button variant="ghost" size="xs" onClick={onShow}>{waiting ? "查看并处理" : "详情"}</Button>}
        {startError && <Button variant="ghost" size="icon-xs" aria-label="关闭本次启动提示" onClick={onDismiss}><X /></Button>}
      </div>
    </div>
    {!starting && active && !waiting && generation?.value && <p>{generation.value.label}</p>}
    {!starting && active && progress && progress.total > 1 && <>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted" role="progressbar" aria-label="已结束的子任务数量"
        aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={progress.completed}>
        <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(100, progress.completed / progress.total * 100)}%` }} />
      </div>
      <span>已结束 {progress.completed} / {progress.total} 项</span>
    </>}
    {generation?.unavailable && active && <p className="text-muted-foreground">进度暂未更新，正在重新连接…</p>}
    {message && <>
      <p role="alert" className={cn("whitespace-pre-wrap break-words leading-relaxed", !expanded && "line-clamp-3")}>{message}</p>
      {message.length > 70 && <Button variant="ghost" size="xs" className="nodrag nopan self-start"
        onClick={(event) => { event.stopPropagation(); setExpanded(!expanded); }}>{expanded ? "收起原因" : "展开原因"}</Button>}
    </>}
    {node && node.derivedState !== "CURRENT" && !starting && <p className="text-muted-foreground">{node.derivedState === "SOURCE_UNAVAILABLE" ? "部分内容不可访问" : "输入已变化，显示上次运行结果"}</p>}
  </section>;
}
