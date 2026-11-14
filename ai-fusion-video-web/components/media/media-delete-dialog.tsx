"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { deleteManagedMedia, previewMediaDeletion, type ManagedMedia, type MediaDeletionMode, type MediaDeletionPreview, type MediaDeletionResult } from "@/lib/api/media";
import { markMediaAccessDeleted } from "@/lib/media-access-cache";
import { markMediaPreviewDeleted } from "@/lib/media-preview-cache";
import { MEDIA_LIBRARY_CHANGED } from "@/lib/media-library-events";

export function MediaDeleteDialog({ media, onClose, onDeleted }: {
  media: Pick<ManagedMedia, "id" | "name">;
  onClose: () => void;
  onDeleted: (result: MediaDeletionResult) => void;
}) {
  const group = useId();
  const [preview, setPreview] = useState<MediaDeletionPreview | null>(null);
  const [mode, setMode] = useState<MediaDeletionMode>("LIBRARY");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(crypto.randomUUID());
  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const result = await previewMediaDeletion(media.id, signal);
      if (signal?.aborted) {
        return;
      }
      setPreview(result);
      setMode(result.libraryRemoved && !result.permanentBlockedReason ? "PERMANENT"
        : result.usages.length || result.permanentBlockedReason ? "LIBRARY" : "PERMANENT");
      requestId.current = crypto.randomUUID();
    } catch (failure) {
      if (!signal?.aborted) {
        setError(getApiErrorMessage(failure));
      }
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
      }
    }
  }, [media.id]);
  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);
  const submit = async () => {
    if (!preview || busy) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await deleteManagedMedia(media.id, {
        requestId: requestId.current, expectedVersion: preview.version,
        referenceToken: preview.referenceToken, mode,
      });
      if (mode === "PERMANENT") {
        markMediaAccessDeleted(media.id);
        markMediaPreviewDeleted(media.id);
      }
      window.dispatchEvent(new CustomEvent(MEDIA_LIBRARY_CHANGED, { detail: result }));
      toast.success(mode === "LIBRARY" ? "已从素材库移除，已有引用仍可使用"
        : result.fileStatus === "PENDING" ? "引用已解除，文件正在清理"
          : result.fileStatus === "SHARED" ? "素材已删除，独立副本仍保留" : "素材及文件已删除");
      onDeleted(result);
      onClose();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      if (getApiErrorStatus(failure) === 409) {
        setPreview(null);
      }
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open onOpenChange={(open) => {
      if (!open && !busy) {
        onClose();
      }
    }}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>删除素材</DialogTitle>
          <DialogDescription className="break-all">{media.name}</DialogDescription>
        </DialogHeader>
        <DialogScrollBody>
          <div className="flex min-w-0 flex-col gap-4">
            {loading && <p role="status" className="text-sm text-muted-foreground">正在检查素材引用…</p>}
            {preview && !loading && (
              <>
                <p className="text-sm">
                  {preview.usages.length
                    ? "此素材已被引用：" + preview.usages.map((usage) => usage.label + " " + usage.count + " 处").join("、") + "。"
                    : "此素材尚未被项目或画布引用。"}
                </p>
                <fieldset disabled={busy} className="flex min-w-0 flex-col gap-4 p-1">
                  <legend className="sr-only">删除方式</legend>
                  <label className="flex min-w-0 items-start gap-3">
                    <input type="radio" name={group} checked={mode === "LIBRARY"} disabled={preview.libraryRemoved}
                      className="mt-1 size-4 shrink-0 accent-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                      onChange={() => {
                        setMode("LIBRARY");
                        requestId.current = crypto.randomUUID();
                      }} />
                    <span className="flex min-w-0 flex-col gap-1">
                      <span className="text-sm font-medium">{preview.libraryRemoved ? "已从素材库移除" : "仅从素材库移除"}</span>
                      <span className="text-xs text-muted-foreground">保留文件，已有画布、项目和生成记录仍可使用。</span>
                    </span>
                  </label>
                  <label className="flex min-w-0 items-start gap-3">
                    <input type="radio" name={group} checked={mode === "PERMANENT"} disabled={!!preview.permanentBlockedReason}
                      className="mt-1 size-4 shrink-0 accent-destructive focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                      onChange={() => {
                        setMode("PERMANENT");
                        requestId.current = crypto.randomUUID();
                      }} />
                    <span className="flex min-w-0 flex-col gap-1">
                      <span className="text-sm font-medium">解除引用并永久删除</span>
                      <span className="text-xs text-muted-foreground">清除当前画布和项目中的素材引用。历史记录及其他位置的引用将无法读取此文件，此操作不可撤销。</span>
                      {preview.permanentBlockedReason && <span className="text-xs text-destructive">{preview.permanentBlockedReason}</span>}
                    </span>
                  </label>
                </fieldset>
                {preview.sharedCopies > 0 && <p className="text-xs text-muted-foreground">还有 {preview.sharedCopies} 份独立副本共用此文件。删除当前素材后，副本及其需要的文件仍会保留。</p>}
              </>
            )}
            {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
            {!preview && !loading && <Button variant="outline" onClick={() => void load()}>重新检查引用</Button>}
          </div>
        </DialogScrollBody>
        <DialogFooter className="shrink-0 gap-2">
          <Button variant="outline" disabled={busy} onClick={onClose}>取消</Button>
          <Button variant={mode === "PERMANENT" ? "destructive" : "default"}
            disabled={busy || loading || !preview || mode === "PERMANENT" && !!preview.permanentBlockedReason
              || mode === "LIBRARY" && !!preview?.libraryRemoved}
            onClick={() => void submit()}>
            {busy && <Loader2 className="animate-spin motion-reduce:animate-none" />}
            {mode === "PERMANENT" ? "永久删除" : "移除素材"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
