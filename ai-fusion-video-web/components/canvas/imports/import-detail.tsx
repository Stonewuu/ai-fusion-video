"use client";
import { useState } from "react";
import {
  CheckCircle2,
  Clock3,
  Eye,
  Loader2,
  LocateFixed,
  RotateCcw,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { importFinished, importLabels } from "@/lib/canvas/import-state";
import type { MediaImport } from "@/lib/api/media-imports";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { worldPosition } from "@/lib/canvas/graph";
import { freeMediaPosition } from "@/lib/canvas/placement";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";
import { useCanvasMediaImports } from "./media-imports-provider";

export function CanvasImportDetail() {
  const imports = useCanvasMediaImports();
  return (
    <Dialog
      open={!!imports.job}
      onOpenChange={(open) => {
        if (!open) imports.choose(null);
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>素材导入</DialogTitle>
          <DialogDescription>
            关闭此窗口后会继续导入，进度保存在「素材」中的导入记录。
          </DialogDescription>
        </DialogHeader>
        {imports.job && (
          <ImportDetailBody key={imports.job.id} job={imports.job} />
        )}
      </DialogContent>
    </Dialog>
  );
}

function ImportDetailBody({ job }: { job: MediaImport }) {
  const imports = useCanvasMediaImports();
  const store = useCanvasStoreApi();
  const nodes = useCanvas((s) => s.snapshot?.graph.nodes ?? []);
  const [selected, setSelected] = useState<number[]>([]);
  const [preview, setPreview] = useState<CanvasMediaPreview | null>(null);
  const failed = job.items.filter(
    (i) => i.status === "FAILED" || i.status === "CANCELLED",
  );
  const indices = selected.filter((i) =>
    failed.some((item) => item.index === i),
  );
  const succeeded = job.items.filter((i) => i.status === "SUCCEEDED");
  const completed = job.items.filter(
    (i) => !["PENDING", "RUNNING"].includes(i.status),
  ).length;
  const failedCount = job.items.filter((i) => i.status === "FAILED").length;
  const cancelledCount = job.items.filter(
    (i) => i.status === "CANCELLED",
  ).length;
  const ready = succeeded.filter((i) => !i.commandId && i.resource);
  const placed = succeeded.filter(
    (i) => i.commandId && nodes.some((n) => n.id === i.nodeId),
  );
  const stopped = importFinished(job.status);
  const place = () => {
    const points = nodes.map((n) => worldPosition(n, nodes));
    const position =
      job.automaticPlacement || job.items.some((i) => i.commandId)
        ? null
        : freeMediaPosition(
            nodes,
            {
              x: points.length ? Math.max(...points.map((p) => p.x)) + 380 : 64,
              y: points.length ? Math.min(...points.map((p) => p.y)) : 64,
            },
            job.items.length,
          );
    void imports.place(job, position);
  };
  return (
    <>
      <div className="min-h-0 space-y-4 overflow-y-auto p-1">
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-3 text-sm">
            <strong>{importLabels[job.status]}</strong>
            <span className="text-muted-foreground">
              已导入 {succeeded.length} / {job.items.length}
            </span>
          </div>
          <div
            role="progressbar"
            aria-label="素材导入进度"
            aria-valuemin={0}
            aria-valuemax={job.items.length}
            aria-valuenow={completed}
            aria-valuetext={`已处理 ${completed} / ${job.items.length} 项，成功 ${succeeded.length} 项，失败 ${failedCount} 项`}
            className="flex h-2 overflow-hidden rounded-full bg-muted"
          >
            <span
              className="h-full bg-primary"
              style={{
                width: `${(succeeded.length / job.items.length) * 100}%`,
              }}
            />
            <span
              className="h-full bg-destructive/60"
              style={{ width: `${(failedCount / job.items.length) * 100}%` }}
            />
            <span
              className="h-full bg-muted-foreground/30"
              style={{ width: `${(cancelledCount / job.items.length) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            {job.cancelRequested && !stopped
              ? "正在停止，已导入的素材会保留"
              : job.phase}
          </p>
        </div>
        <ul className="divide-y divide-border/20">
          {job.items.map((item) => (
            <li key={item.index} className="flex items-start gap-3 py-3">
              {failed.some((f) => f.index === item.index) ? (
                <Checkbox
                  aria-label={`选择重试${item.title}`}
                  checked={indices.includes(item.index)}
                  disabled={!stopped || !job.canControl || imports.busy}
                  onCheckedChange={(value) =>
                    setSelected((old) =>
                      value
                        ? [...old, item.index]
                        : old.filter((i) => i !== item.index),
                    )
                  }
                />
              ) : item.status === "SUCCEEDED" ? (
                <CheckCircle2 className="mt-1 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
              ) : item.status === "RUNNING" ? (
                <Loader2 className="mt-1 size-4 shrink-0 animate-spin text-primary motion-reduce:animate-none" />
              ) : (
                <Clock3 className="mt-1 size-4 shrink-0 text-muted-foreground" />
              )}
              <div className="min-w-0 flex-1 space-y-1">
                <p className="break-words text-sm font-medium">{item.title}</p>
                <p className="text-xs text-muted-foreground">
                  {importLabels[item.status]} · 第 {item.attempt} 次
                  {item.commandId &&
                    (nodes.some((n) => n.id === item.nodeId)
                      ? " · 已放入画布"
                      : " · 曾放入画布，节点已移除")}
                </p>
                {item.message && (
                  <p className="break-words text-xs text-destructive">
                    {item.message}
                  </p>
                )}
              </div>
              {item.resource && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label={`预览${item.title}`}
                  onClick={() =>
                    setPreview({
                      title: item.title,
                      source: `media://${item.resource!.id}`,
                      kind: (
                        {
                          IMAGE: "image",
                          VIDEO: "video",
                          AUDIO: "audio",
                          TEXT: "textFile",
                          DOCUMENT: "document",
                        } as const
                      )[item.resource!.content.kind],
                    })
                  }
                >
                  <Eye />
                </Button>
              )}
            </li>
          ))}
        </ul>
        {failed.length > 0 && stopped && (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={imports.busy || !job.canControl}
              onClick={() => setSelected(failed.map((i) => i.index))}
            >
              选择未完成项
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!indices.length || imports.busy || !job.canControl}
              onClick={() => void imports.retry(job, indices)}
            >
              <RotateCcw />
              重试所选（{indices.length}）
            </Button>
          </div>
        )}
        {(imports.error || imports.placementError) && (
          <p role="alert" className="text-sm text-destructive">
            {imports.error ?? imports.placementError}
          </p>
        )}
      </div>
      <DialogFooter className="shrink-0">
        {!stopped && (
          <Button
            variant="outline"
            disabled={imports.busy || job.cancelRequested}
            onClick={() => void imports.cancel(job)}
          >
            <Square />
            停止导入
          </Button>
        )}
        {!!placed.length && (
          <Button
            variant="outline"
            onClick={() => {
              store.getState().focus(placed.map((i) => i.nodeId));
              imports.choose(null);
            }}
          >
            <LocateFixed />
            定位结果
          </Button>
        )}
        {!!ready.length && (
          <Button disabled={imports.busy || !job.canControl} onClick={place}>
            放入画布（{ready.length}）
          </Button>
        )}
        {!ready.length && (
          <Button variant="outline" onClick={() => imports.choose(null)}>
            关闭
          </Button>
        )}
      </DialogFooter>
      {preview && (
        <CanvasMediaPreviewDialog
          media={preview}
          onClose={() => setPreview(null)}
        />
      )}
    </>
  );
}
