"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { ArrowLeft, Play } from "lucide-react";
import { CanvasShareDialog } from "../collaboration/share-dialog";
import { CanvasPresenceBar } from "../collaboration/presence-bar";
import { CanvasExportDialog } from "../delivery/export-dialog";
import { CanvasProjectForkDialog } from "../collaboration/project-fork-dialog";
import { CanvasDocumentActions } from "./canvas-document-actions";
import { requestAssistantOpen } from "@/components/dashboard/assistant/open-assistant";
import { AssistantBrandIcon } from "@/components/dashboard/assistant/assistant-brand-icon";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { useCanvasRuns } from "../runs/canvas-runs-provider";

export function CanvasWorkspaceHeader({
  onRun,
  onOpenRuns,
}: {
  onRun: () => void;
  onOpenRuns: () => void;
}) {
  const router = useRouter();
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => state.pending);
  const saving = useCanvas((state) => state.saving);
  const saveError = useCanvas((state) => state.saveError);
  const sync = useCanvas((state) => state.sync);
  const syncError = useCanvas((state) => state.syncError);
  const selection = useCanvas((state) => state.selection);
  const runtime = useCanvasRuns();
  const [rename, setRename] = useState<{
    title: string;
    revision: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sharing, setSharing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [forking, setForking] = useState(false);
  const requests = useCommandRequestKey();
  const document = snapshot.document;
  const hasRunnableNodes = snapshot.graph.nodes.some((node) => node.kind !== "group" && node.kind !== "note");
  const editable =
    snapshot.permissions.canEdit && !document.archived && !pending;
  const update = async (
    change: { title?: string; archived?: boolean },
    expectedRevision: number,
  ) => {
    setBusy(true);
    setError(null);
    const body = { ...change, expectedRevision };
    try {
      await canvasApi.metadata(document.id, {
        ...body,
        requestId: requests.forPayload(body),
      });
      await store.getState().load();
      setRename(null);
      requests.clear();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      if (!rename) toast.error(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <header className="flex shrink-0 items-center gap-2 border-b border-border/20 px-4 py-3">
        <Button
          variant="ghost"
          size="icon"
          title="返回画布列表"
          onClick={() =>
            router.push(
              document.projectId
                ? `/projects/${document.projectId}/canvases`
                : "/canvases",
            )
          }
        >
          <ArrowLeft />
        </Button>
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-base font-semibold">{document.title}</h1>
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs text-muted-foreground" aria-live="polite" title={pending && saveError ? "正在自动确认保存，恢复连接后会继续同步。" : syncError ?? undefined}>
              {saving
                ? "正在保存…"
                : pending && saveError
                  ? "正在恢复保存…"
                  : sync === "connected"
                    ? "已保存并同步"
                    : sync === "retrying"
                      ? "正在恢复同步"
                      : "正在连接"}
              {document.archived
                ? " · 已归档"
                : !snapshot.permissions.canEdit
                  ? " · 只读"
                  : ""}
            </p>
            <CanvasPresenceBar />
          </div>
        </div>
        <CanvasDocumentActions
          editable={editable}
          canManage={snapshot.permissions.canManage}
          archived={document.archived}
          busy={!!pending || busy}
          onRename={() =>
            setRename({ title: document.title, revision: document.revision })
          }
          onArchive={() =>
            void update({ archived: !document.archived }, document.revision)
          }
          onShare={() => setSharing(true)}
          onExport={() => setExporting(true)}
          onFork={() => setForking(true)}
        />
        <Button
          variant="outline"
          onClick={requestAssistantOpen}
          aria-label="助手"
          title="让助手理解选区、修改内容并运行画布"
        >
          <AssistantBrandIcon className="size-6" />
          <span className="hidden sm:inline">助手</span>
        </Button>
        <Button
          variant="ai"
          aria-label={runtime.pending ? "处理待确认运行" : selection.length ? "运行所选" : "运行画布"}
          disabled={
            !snapshot.permissions.canExecute ||
            document.archived ||
            !!pending ||
            saving ||
            sync !== "connected" ||
            runtime.busy ||
            runtime.loading ||
            (!hasRunnableNodes && !runtime.pending && !runtime.corrupt)
          }
          onClick={runtime.pending || runtime.corrupt ? onOpenRuns : onRun}
          title={hasRunnableNodes ? "先检查范围和数量，再开始运行" : "先添加文字、素材或创作步骤，再运行画布"}
        >
          <Play />
          <span className="hidden sm:inline">
            {runtime.pending
              ? "处理待确认运行"
              : selection.length
                ? "运行所选"
                : "运行画布"}
          </span>
        </Button>
      </header>
      {sharing && <CanvasShareDialog onClose={() => setSharing(false)} />}
      {exporting && <CanvasExportDialog onClose={() => setExporting(false)} />}
      {forking && <CanvasProjectForkDialog onClose={() => setForking(false)} />}
      <Dialog
        open={!!rename}
        onOpenChange={(open) => {
          if (!open && !busy) setRename(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>重命名画布</DialogTitle>
            <DialogDescription>
              为当前创作空间起一个容易识别的名称。
            </DialogDescription>
          </DialogHeader>
          <Input
            aria-label="画布名称"
            value={rename?.title ?? ""}
            disabled={busy}
            maxLength={200}
            onChange={(event) =>
              setRename((current) =>
                current ? { ...current, title: event.target.value } : null,
              )
            }
          />
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setRename(null)}
            >
              取消
            </Button>
            <Button
              disabled={busy || !rename?.title.trim()}
              onClick={() => {
                if (rename)
                  void update({ title: rename.title.trim() }, rename.revision);
              }}
            >
              保存名称
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
