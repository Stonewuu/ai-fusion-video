"use client";
import { CanvasContentValue } from "../shared/canvas-content-value";

import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import {
  canvasHistoryApi,
  type BusinessHistoryDetail as Detail,
  type BusinessRestorePlan,
} from "@/lib/api/canvas-history";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import {
  businessOperationNames,
  historyFieldLabel,
} from "@/lib/canvas/history-presentation";
import { canvasDate, canvasTypeNames } from "@/lib/canvas/labels";
import {
  useCanvas,
  useCanvasResourceCache,
  useCanvasStoreApi,
} from "../canvas-provider";

export function BusinessHistoryDetail({
  commandId,
  onClose,
  onRestored,
}: {
  commandId: string;
  onClose: () => void;
  onRestored: () => void;
}) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const store = useCanvasStoreApi();
  const resources = useCanvasResourceCache();
  const requests = useCommandRequestKey();
  const [detail, setDetail] = useState<Detail | null>(null);
  const [plan, setPlan] = useState<BusinessRestorePlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void canvasHistoryApi
      .detail(snapshot.document.id, commandId, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setDetail(result);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      });
    return () => controller.abort();
  }, [snapshot.document.id, commandId]);
  const preview = async () => {
    setBusy(true);
    setError(null);
    try {
      setPlan(await canvasHistoryApi.preview(snapshot.document.id, commandId));
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const restore = async () => {
    if (!plan) return;
    setBusy(true);
    setError(null);
    try {
      await canvasHistoryApi.restore(
        snapshot.document.id,
        plan,
        requests.forPayload({ commandId, digest: plan.digest }),
      );
      resources.invalidate();
      await store.getState().load();
      requests.clear();
      onRestored();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const changes = plan
    ? plan.changes.map((entry) => ({
        ...entry,
        before: entry.current,
        after: entry.proposed,
      }))
    : (detail?.revisions ?? []);
  const editable = snapshot.permissions.canEdit && !snapshot.document.archived;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-4xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {plan
              ? "确认恢复的内容"
              : detail
                ? (businessOperationNames[detail.summary.operation] ??
                  "内容变更")
                : "读取内容变更"}
          </DialogTitle>
          <DialogDescription>
            {plan
              ? "只恢复原操作影响的内容，保留其他字段的后续修改。已发生的生成费用不会撤销。"
              : detail
                ? `${detail.summary.actorName} · ${canvasDate(detail.summary.committedAt)}`
                : "正在读取已提交的版本记录。"}
          </DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-6 overflow-y-auto p-1">
          {plan?.notices.map((notice) => (
            <p
              key={notice}
              className="rounded-lg border border-border/30 bg-muted/30 p-3 text-sm"
            >
              {notice}
            </p>
          ))}
          {plan?.conflicts.map((conflict, index) => (
            <p key={index} role="alert" className="text-sm text-destructive">
              {canvasTypeNames[conflict.resourceType]} ·{" "}
              {historyFieldLabel(conflict.resourceType, conflict.field)}：
              {conflict.message}
            </p>
          ))}
          {changes.map((entry) => (
            <section
              key={`${entry.resourceType}:${entry.resourceId}`}
              className="space-y-3"
            >
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold">
                  {String(
                    entry.after.name ??
                      entry.after.title ??
                      entry.after.sceneHeading ??
                      entry.after.shotNumber ??
                      canvasTypeNames[entry.resourceType],
                  )}
                </h3>
                <span className="text-xs text-muted-foreground">
                  {canvasTypeNames[entry.resourceType]}
                </span>
              </div>
              {!entry.before && (
                <p className="text-xs text-sky-700 dark:text-sky-300">此次操作新建了这项内容。</p>
              )}
              {entry.fields.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  相关内容的版本也已同步更新。
                </p>
              )}
              {entry.fields.map((field) => (
                <div
                  key={field}
                  className="space-y-2 border-b border-border/20 pb-4"
                >
                  <h4 className="text-sm font-medium">
                    {historyFieldLabel(entry.resourceType, field)}
                  </h4>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-2 rounded-lg border border-border/20 bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">
                        {plan ? "当前内容" : "修改前"}
                      </p>
                      <CanvasContentValue
                        type={entry.resourceType}
                        field={field}
                        value={entry.before?.[field]}
                      />
                    </div>
                    <div className="space-y-2 rounded-lg border border-primary/30 bg-primary/5 p-3">
                      <p className="text-xs text-sky-700 dark:text-sky-300">
                        {plan ? "恢复后" : "修改后"}
                      </p>
                      <CanvasContentValue
                        type={entry.resourceType}
                        field={field}
                        value={entry.after[field]}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </section>
          ))}
          {detail?.summary.snapshotSchemaVersion === 0 && (
            <p className="text-sm text-muted-foreground">
              这条记录只支持查看。需要调整内容时，请回到编辑面板修改。
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter className="shrink-0">
          <Button
            variant="outline"
            disabled={busy}
            onClick={() => (plan ? setPlan(null) : onClose())}
          >
            {plan ? "返回变更记录" : "关闭"}
          </Button>
          {detail?.summary.snapshotSchemaVersion === 1 &&
            editable &&
            (!plan || !plan.canRestore || error) && (
              <Button
                variant={plan ? "outline" : "default"}
                disabled={busy}
                onClick={() => void preview()}
              >
                <RotateCcw />
                {plan ? "重新检查" : "预览恢复"}
              </Button>
            )}
          {plan?.canRestore && editable && (
            <Button
              variant={
                plan.changes.some((entry) => entry.proposed.deleted === true)
                  ? "destructive"
                  : "default"
              }
              disabled={busy || !!error}
              onClick={() => void restore()}
            >
              确认恢复
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
