"use client";

import { useId, useState } from "react";
import { Check, RotateCcw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import {
  canvasRunApi,
  type CanvasNodeRun,
  type CanvasRun,
} from "@/lib/api/canvas-runs";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { runFinished } from "@/lib/canvas/run-state";
import { useCanvasRuns } from "./canvas-runs-provider";
import { CanvasRunNode } from "./run-node";

type Decision = "RETRY_FAILED" | "USE_SUCCEEDED" | "FINISH";
const decisions = {
  RETRY_FAILED: {
    label: "重试失败项",
    icon: RotateCcw,
    description:
      "只重新执行未完整成功的项目，使用原输入和配置。可能产生新的费用，仍受本次运行上限约束。",
  },
  USE_SUCCEEDED: {
    label: "采用已有结果",
    icon: Check,
    description:
      "跳过失败项，按原顺序汇总已有结果并继续后续步骤；原失败记录仍会保留。",
  },
  FINISH: {
    label: "结束批次",
    icon: Square,
    description: "保留已有结果，结束本批次并停止依赖它的后续步骤节点。",
  },
};

export function CanvasBatchResults({
  run,
  node,
}: {
  run: CanvasRun;
  node: CanvasNodeRun;
}) {
  const runtime = useCanvasRuns();
  const noteId = useId();
  const [page, setPage] = useState(0);
  const [decision, setDecision] = useState<{
    action: Decision;
    revision: number;
  } | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const attempts = run.nodes.filter((item) => item.parentNodeRunId === node.id);
  const indexed = new Map<number, CanvasNodeRun>();
  for (const item of attempts)
    if (
      !indexed.has(item.itemIndex) ||
      indexed.get(item.itemIndex)!.attempt < item.attempt
    )
      indexed.set(item.itemIndex, item);
  const items = [...indexed.values()].sort((a, b) => a.itemIndex - b.itemIndex);
  const waiting = items.filter((item) => item.status === "WAITING_CONFIRMATION");
  const completed = items.filter((item) => runFinished(item.status)).length;
  const succeeded = items.filter((item) => item.status === "SUCCEEDED").length;
  const withResults = items.some((item) =>
    Object.values(item.outputs.values).some((values) => values.length),
  );
  const pages = Math.max(1, Math.ceil(items.length / 6));
  const current = Math.min(page, pages - 1);
  const review = async () => {
    if (!decision || !runtime.scope || busy) return;
    setBusy(true);
    setError(null);
    try {
      await canvasRunApi.reviewBatch(run.id, node.id, {
        action: decision.action,
        expectedRevision: decision.revision,
        note: note.trim(),
        scope: runtime.scope,
      });
      runtime.reload();
      setDecision(null);
      setNote("");
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  if (!items.length) return null;
  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="font-medium">
            已处理 {completed} / {items.length} 项
          </span>
          <span className="text-muted-foreground">
            成功 {succeeded} · 未成功 {completed - succeeded}
          </span>
        </div>
        <div
          role="progressbar"
          aria-label="批量处理进度"
          aria-valuemin={0}
          aria-valuemax={items.length}
          aria-valuenow={completed}
          className="h-1.5 overflow-hidden rounded-full bg-muted"
        >
          <div
            className="h-full bg-primary"
            style={{ width: `${(completed / items.length) * 100}%` }}
          />
        </div>
      </div>
      {waiting.length > 0 && run.canControl && !run.cancelRequested ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">{waiting.length} 项需要确认</span>
          {waiting.slice(0, 5).map((item) => (
            <Button key={item.id} variant="outline" size="xs" onClick={() => setPage(Math.floor(items.indexOf(item) / 6))}>
              查看第 {item.itemIndex + 1} 项
            </Button>
          ))}
        </div>
      ) : null}
      {node.status === "SUCCEEDED" && succeeded < items.length && (
        <p className="text-xs text-muted-foreground">
          已确认采用已有结果；{items.length - succeeded}{" "}
          项未完整成功，原记录仍保留。
        </p>
      )}
      {node.status === "WAITING_CONFIRMATION" &&
        run.canControl &&
        !run.cancelRequested && (
          <div className="flex flex-wrap gap-2">
            {(Object.keys(decisions) as Decision[]).map((action) => {
              const Icon = decisions[action].icon;
              return (
                <Button
                  key={action}
                  size="xs"
                  variant={
                    action === "RETRY_FAILED"
                      ? "default"
                      : action === "FINISH"
                        ? "destructive-ghost"
                        : "outline"
                  }
                  disabled={
                    busy ||
                    runtime.busy ||
                    (action === "USE_SUCCEEDED" && !withResults)
                  }
                  onClick={() => {
                    setError(null);
                    setDecision({ action, revision: node.revision });
                  }}
                >
                  <Icon />
                  {decisions[action].label}
                </Button>
              );
            })}
          </div>
        )}
      <details
        className="space-y-2"
        open={
          node.status === "WAITING_CONFIRMATION" ||
          waiting.length > 0 ||
          run.status === "UNKNOWN_SUBMISSION" ||
          run.status === "NEEDS_REVIEW"
        }
      >
        <summary className="cursor-pointer text-sm text-muted-foreground">
          查看每项进度与结果
        </summary>
        {items.slice(current * 6, current * 6 + 6).map((item) => (
          <div key={item.id}>
            <CanvasRunNode run={run} node={item} />
            {item.attempt > 1 && (
              <details className="mt-3">
                <summary className="cursor-pointer text-xs text-muted-foreground">
                  之前的执行记录（{item.attempt - 1} 次）
                </summary>
                {attempts
                  .filter(
                    (previous) =>
                      previous.itemIndex === item.itemIndex &&
                      previous.attempt < item.attempt,
                  )
                  .map((previous) => (
                    <CanvasRunNode
                      key={previous.id}
                      run={run}
                      node={previous}
                    />
                  ))}
              </details>
            )}
          </div>
        ))}
        {pages > 1 && (
          <div className="flex items-center justify-between gap-2">
            <Button
              variant="ghost"
              size="xs"
              disabled={current === 0}
              onClick={() => setPage(current - 1)}
            >
              上一页
            </Button>
            <span className="text-xs text-muted-foreground">
              {current + 1} / {pages}
            </span>
            <Button
              variant="ghost"
              size="xs"
              disabled={current + 1 === pages}
              onClick={() => setPage(current + 1)}
            >
              下一页
            </Button>
          </div>
        )}
      </details>
      <Dialog
        open={!!decision}
        onOpenChange={(open) => {
          if (!open && !busy) setDecision(null);
        }}
      >
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
          <DialogHeader className="shrink-0">
            <DialogTitle>
              {decision && decisions[decision.action].label}
            </DialogTitle>
            <DialogDescription>
              {decision && decisions[decision.action].description}
            </DialogDescription>
          </DialogHeader>
          <DialogScrollBody className="flex flex-col gap-2">
            <Label htmlFor={noteId}>处理说明</Label>
            <Textarea
              id={noteId}
              value={note}
              maxLength={1000}
              rows={3}
              disabled={busy}
              placeholder="填写本次决定的原因"
              onChange={(event) => setNote(event.target.value)}
            />
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
          </DialogScrollBody>
          <DialogFooter className="shrink-0">
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setDecision(null)}
            >
              返回
            </Button>
            <Button
              variant={
                decision?.action === "FINISH" ? "destructive" : "default"
              }
              disabled={busy || !note.trim()}
              onClick={() => void review()}
            >
              {busy ? "处理中…" : decision && decisions[decision.action].label}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
