"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, Download, RefreshCw } from "lucide-react";
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
import { importLabels } from "@/lib/canvas/import-state";
import { useCanvasMediaImports } from "./media-imports-provider";
import { CanvasImportSources } from "./import-sources";

export function CanvasImportPanel() {
  const imports = useCanvasMediaImports();
  const [selecting, setSelecting] = useState(false);
  const [clear, setClear] = useState(false);
  return (
    <section
      className="flex flex-col gap-2.5 rounded-[12px] border border-foreground/[0.07] bg-foreground/[0.02] p-3 dark:border-white/[0.08] dark:bg-white/[0.025]"
      aria-label="素材导入记录"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <h3 className="text-xs font-medium text-foreground/85">导入记录</h3>
          {imports.active.length > 0 && (
            <span className="rounded-[6px] bg-primary/12 px-1.5 py-0.5 text-[11px] font-medium text-primary">
              {imports.active.length} 批处理中
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <CanvasIconButton
            size="icon-xs"
            label="刷新导入记录"
            disabled={imports.busy}
            onClick={imports.reload}
          >
            <RefreshCw />
          </CanvasIconButton>
          <CanvasButton
            variant="secondary"
            size="sm"
            disabled={!imports.ready}
            onClick={() => setSelecting(true)}
          >
            <Download />
            导入节点素材
          </CanvasButton>
        </div>
      </div>
      {imports.pending && (
        <div className="flex flex-col gap-2 rounded-[10px] border border-primary/25 bg-primary/[0.06] p-2.5">
          <p className="text-xs font-medium">
            有 {imports.pending.request.sources.length} 项导入尚未确认
          </p>
          <div className="flex flex-wrap gap-1.5">
            <CanvasButton
              variant="primary"
              size="sm"
              disabled={imports.busy}
              onClick={() => void imports.resume()}
            >
              确认启动状态
            </CanvasButton>
            <CanvasButton
              variant="secondary"
              size="sm"
              disabled={imports.busy}
              onClick={() => void imports.withdraw()}
            >
              撤回
            </CanvasButton>
          </div>
        </div>
      )}
      {imports.corrupt && (
        <CanvasButton
          variant="secondary"
          size="sm"
          onClick={() => setClear(true)}
        >
          清理损坏的本地请求
        </CanvasButton>
      )}
      {(imports.error || imports.placementError) && (
        <p role="alert" className="text-xs text-destructive">
          {imports.error ?? imports.placementError}
        </p>
      )}
      {!imports.history.length && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          选择画布上的内容节点，可批量导入首尾帧、视频、配乐等素材。
        </p>
      )}
      {imports.history.length > 0 && (
        <ul className="flex flex-col gap-0.5">
          {imports.history.map((job) => (
            <li key={job.id}>
              <CanvasButton
                variant="ghost"
                size="sm"
                className="w-full justify-between"
                onClick={() => imports.choose(job.id)}
              >
                <span className="truncate text-left">
                  {importLabels[job.status]} · {job.succeeded}/{job.total} 项
                </span>
                <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                  {new Date(job.createdAt).toLocaleTimeString("zh-CN", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </CanvasButton>
            </li>
          ))}
        </ul>
      )}
      {(imports.page > 0 || imports.hasMore) && (
        <div className="flex items-center justify-between gap-2 pt-0.5">
          <CanvasIconButton
            size="icon-xs"
            label="上一页导入记录"
            disabled={imports.page === 0}
            onClick={() => imports.setPage(imports.page - 1)}
          >
            <ChevronLeft />
          </CanvasIconButton>
          <span className="text-[11px] tabular-nums text-muted-foreground">
            第 {imports.page + 1} 页
          </span>
          <CanvasIconButton
            size="icon-xs"
            label="下一页导入记录"
            disabled={!imports.hasMore}
            onClick={() => imports.setPage(imports.page + 1)}
          >
            <ChevronRight />
          </CanvasIconButton>
        </div>
      )}
      {selecting && <CanvasImportSources onClose={() => setSelecting(false)} />}
      <Dialog open={clear} onOpenChange={setClear}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>清理本地请求记录</DialogTitle>
            <DialogDescription>
              此记录无法还原提交内容。请先核对导入历史；清理本地记录不会取消已经受理的任务。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setClear(false)}>
              返回
            </Button>
            <Button
              disabled={imports.busy}
              onClick={() =>
                void imports.clearCorrupt().then((ok) => {
                  if (ok) {
                    setClear(false);
                  }
                })
              }
            >
              已核对，清理记录
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
