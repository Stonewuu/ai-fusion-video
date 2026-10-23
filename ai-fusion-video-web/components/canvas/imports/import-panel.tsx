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
import { importLabels } from "@/lib/canvas/import-state";
import { useCanvasMediaImports } from "./media-imports-provider";
import { CanvasImportSources } from "./import-sources";

export function CanvasImportPanel() {
  const imports = useCanvasMediaImports();
  const [selecting, setSelecting] = useState(false);
  const [clear, setClear] = useState(false);
  return (
    <section
      className="space-y-3 border-b border-border/20 pb-4"
      aria-label="素材导入记录"
    >
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">
          导入记录
          {imports.active.length > 0 && (
            <span className="ml-2 text-xs text-primary">
              {imports.active.length} 批处理中
            </span>
          )}
        </h3>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="刷新导入记录"
          disabled={imports.busy}
          onClick={imports.reload}
        >
          <RefreshCw />
        </Button>
      </div>
      <Button
        variant="outline"
        className="w-full"
        disabled={!imports.ready}
        onClick={() => setSelecting(true)}
      >
        <Download />
        导入所选节点素材
      </Button>
      {imports.pending && (
        <div className="space-y-2 rounded-lg border border-primary/30 bg-primary/5 p-3">
          <p className="text-sm font-medium">
            有 {imports.pending.request.sources.length} 项导入尚未确认
          </p>
          <p className="text-xs text-muted-foreground">
            确认会继续处理原请求，不会重复创建任务。
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              disabled={imports.busy}
              onClick={() => void imports.resume()}
            >
              确认启动状态
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={imports.busy}
              onClick={() => void imports.withdraw()}
            >
              撤回
            </Button>
          </div>
        </div>
      )}
      {imports.corrupt && (
        <Button variant="outline" onClick={() => setClear(true)}>
          清理损坏的本地请求
        </Button>
      )}
      {(imports.error || imports.placementError) && (
        <p role="alert" className="text-xs text-destructive">
          {imports.error ?? imports.placementError}
        </p>
      )}
      {!imports.history.length && (
        <p className="text-xs text-muted-foreground">
          选择画布上的内容节点，可批量导入首尾帧、视频、配乐等素材。
        </p>
      )}
      <ul className="space-y-1">
        {imports.history.map((job) => (
          <li key={job.id}>
            <Button
              variant="ghost"
              className="w-full justify-between"
              onClick={() => imports.choose(job.id)}
            >
              <span className="truncate text-left">
                {importLabels[job.status]} · {job.succeeded}/{job.total} 项
              </span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {new Date(job.createdAt).toLocaleTimeString("zh-CN", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </Button>
          </li>
        ))}
      </ul>
      {(imports.page > 0 || imports.hasMore) && (
        <div className="flex items-center justify-between gap-2">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="上一页导入记录"
            disabled={imports.page === 0}
            onClick={() => imports.setPage(imports.page - 1)}
          >
            <ChevronLeft />
          </Button>
          <span className="text-xs text-muted-foreground">
            第 {imports.page + 1} 页
          </span>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="下一页导入记录"
            disabled={!imports.hasMore}
            onClick={() => imports.setPage(imports.page + 1)}
          >
            <ChevronRight />
          </Button>
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
                  if (ok) setClear(false);
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
