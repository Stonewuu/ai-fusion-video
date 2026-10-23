"use client";
import { userFacingError } from "@/lib/user-facing-error";
import { useEffect, useState } from "react";
import { Download, Loader2, PackageCheck, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  canvasExportsApi,
  type CanvasExportSummary,
} from "@/lib/api/canvas-exports";
import { getApiErrorMessage } from "@/lib/api/api-error";

const STATUS: Record<CanvasExportSummary["status"], string> = {
  QUEUED: "等待生成",
  RUNNING: "正在生成",
  CANCEL_REQUESTED: "正在取消",
  SUCCEEDED: "可以下载",
  FAILED: "生成失败",
  CANCELLED: "已取消",
};
export function exportSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
export function CanvasExportHistory({ canvasId }: { canvasId: string }) {
  const [page, setPage] = useState(0);
  const [items, setItems] = useState<CanvasExportSummary[]>([]);
  const [loadedPage, setLoadedPage] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const next = await canvasExportsApi.list(
          canvasId,
          page,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setItems(next);
          setLoadedPage(page);
          setError(null);
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 3000);
      }
    };
    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [canvasId, page]);
  const act = async (item: CanvasExportSummary, cancel: boolean) => {
    setBusy(item.id);
    setError(null);
    try {
      if (cancel) await canvasExportsApi.cancel(canvasId, item.id);
      else await canvasExportsApi.download(canvasId, item.id);
      setItems(await canvasExportsApi.list(canvasId, page));
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(null);
    }
  };
  return (
    <div className="flex flex-col gap-4">
      {loadedPage !== page ? (
        <p className="text-sm text-muted-foreground">正在读取导出记录…</p>
      ) : items.length === 0 ? (
        <div className="flex flex-col items-center gap-3 py-8 text-muted-foreground">
          <PackageCheck className="size-8" />
          <p className="text-sm">此页没有导出记录</p>
        </div>
      ) : (
        <ul className="divide-y divide-border/20">
          {items.map((item) => (
            <li key={item.id} className="flex items-start gap-3 py-4">
              <div className="min-w-0 flex-1 space-y-2">
                <p className="break-all text-sm font-medium">{item.filename}</p>
                <p className="text-xs text-muted-foreground">
                  {item.actorName} · {new Date(item.createdAt).toLocaleString()}
                </p>
                <p
                  className={`text-xs ${item.status === "SUCCEEDED" ? "text-primary" : item.status === "FAILED" ? "text-destructive" : "text-muted-foreground"}`}
                >
                  {STATUS[item.status]}
                  {item.byteSize !== null
                    ? ` · ${exportSize(item.byteSize)}`
                    : ""}
                  {!item.contentUnavailable
                    ? ` · ${item.businessObjects} 个项目内容 · ${item.mediaFiles} 个媒体`
                    : ""}
                </p>
                {item.error && (
                  <p className="text-xs text-destructive">{userFacingError(item.error)}</p>
                )}
                {item.contentHash && (
                  <details className="text-xs text-muted-foreground">
                    <summary className="cursor-pointer">查看文件校验值</summary>
                    <p className="mt-2 break-all font-mono">
                      SHA-256: {item.contentHash}
                    </p>
                  </details>
                )}
              </div>
              {item.status === "SUCCEEDED" && !item.contentUnavailable && (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={busy === item.id}
                  onClick={() => void act(item, false)}
                >
                  <Download />
                  下载
                </Button>
              )}
              {item.canCancel && (
                <Button
                  size="icon-sm"
                  variant="ghost"
                  title="取消此次交付"
                  disabled={
                    busy === item.id || item.status === "CANCEL_REQUESTED"
                  }
                  onClick={() => void act(item, true)}
                >
                  {busy === item.id ? (
                    <Loader2 className="animate-spin motion-reduce:animate-none" />
                  ) : (
                    <X />
                  )}
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="flex items-center justify-between gap-2">
        <Button
          variant="ghost"
          size="sm"
          disabled={page === 0}
          onClick={() => setPage((value) => value - 1)}
        >
          上一页
        </Button>
        <span className="text-xs text-muted-foreground">第 {page + 1} 页</span>
        <Button
          variant="ghost"
          size="sm"
          disabled={loadedPage !== page || items.length < 40}
          onClick={() => setPage((value) => value + 1)}
        >
          下一页
        </Button>
      </div>
    </div>
  );
}

export function CanvasExportDownloadButton({
  canvasId,
  exportId,
}: {
  canvasId: string;
  exportId: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="flex flex-col gap-2">
      <Button
        variant="outline"
        size="sm"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await canvasExportsApi.download(canvasId, exportId);
          } catch (failure) {
            setError(getApiErrorMessage(failure));
          } finally {
            setBusy(false);
          }
        }}
      >
        <Download />
        下载交付文件
      </Button>
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
