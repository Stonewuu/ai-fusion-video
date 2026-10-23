"use client";
import { userFacingError } from "@/lib/user-facing-error";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
  type CanvasMediaExecution,
  type CanvasMediaReviewAction,
} from "@/lib/api/canvas-runs";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useCanvasRuns } from "./canvas-runs-provider";

const labels: Record<CanvasMediaReviewAction, string> = {
  RECHECK: "查询原任务",
  RETRY_INGEST: "继续保存结果",
  RECORD_RECEIPT: "补录原任务编号",
  CONFIRM_FINISHED: "确认任务已结束",
};
export function CanvasMediaExecutionDialog({
  runId,
  nodeId,
  onClose,
}: {
  runId: string;
  nodeId: string;
  onClose: () => void;
}) {
  const runtime = useCanvasRuns();
  const [data, setData] = useState<CanvasMediaExecution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [decision, setDecision] = useState<{
    action: CanvasMediaReviewAction;
    revision: number;
  } | null>(null);
  const [note, setNote] = useState("");
  const [providerId, setProviderId] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      let stopped = false;
      try {
        const result = await canvasRunApi.media(runId, nodeId);
        if (active) {
          setData(result);
          setError(null);
        }
        stopped = ["SUCCEEDED", "FAILED", "CANCELLED"].includes(result.status);
      } catch (failure) {
        if (active) {
          setError(getApiErrorMessage(failure));
          if ([401, 403, 404].includes(getApiErrorStatus(failure) ?? 0)) {
            setData(null);
            stopped = true;
          }
        }
      } finally {
        if (active && !stopped) timer = setTimeout(() => void load(), 3000);
      }
    };
    void load();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [runId, nodeId, refresh]);
  const review = async (
    action: CanvasMediaReviewAction,
    revision: number,
    reason: string,
  ) => {
    if (!runtime.scope || busy) return;
    setBusy(true);
    setActionError(null);
    try {
      const result = await canvasRunApi.reviewMedia(runId, nodeId, {
        expectedRevision: revision,
        action,
        note: reason,
        providerTaskId:
          action === "RECORD_RECEIPT" ? providerId.trim() : undefined,
        scope: runtime.scope,
      });
      setData(result);
      setDecision(null);
      setNote("");
      setRefresh((value) => value + 1);
      runtime.reload();
    } catch (failure) {
      setActionError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>媒体任务详情</DialogTitle>
          <DialogDescription>
            查看处理进度、已生成的结果，以及需要你确认的操作。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-4">
          {data && (
            <>
              <p className="text-sm font-medium">{data.phase}</p>
              {data.error && (
                <p className="text-sm text-destructive">{userFacingError(data.error)}</p>
              )}
              {data.providerTaskId && (
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">
                    供应商任务编号
                  </p>
                  <p className="break-all font-mono text-xs">
                    {data.providerTaskId}
                  </p>
                </div>
              )}
              {data.remoteCapacityHeld && (
                <p className="text-xs text-muted-foreground">
                  服务仍在处理中，同类任务可能需要等待；请勿重复提交。
                </p>
              )}
              {!decision && (
                <div className="flex flex-wrap gap-2">
                  {data.actions.map((action) => (
                    <Button
                      key={action}
                      size="sm"
                      variant={
                        action === "CONFIRM_FINISHED"
                          ? "destructive-ghost"
                          : "outline"
                      }
                      disabled={busy || !!error}
                      onClick={() =>
                        action === "RECHECK" || action === "RETRY_INGEST"
                          ? void review(action, data.revision, labels[action])
                          : setDecision({ action, revision: data.revision })
                      }
                    >
                      {labels[action]}
                    </Button>
                  ))}
                </div>
              )}
            </>
          )}
          {decision && (
            <div className="space-y-3 rounded-lg border border-border/20 bg-background/70 p-3">
              <p className="text-sm font-medium">{labels[decision.action]}</p>
              {decision.action === "RECORD_RECEIPT" ? (
                <div className="flex flex-col gap-2">
                  <Label htmlFor="canvas-audio-receipt">
                    已确认的供应商原任务编号
                  </Label>
                  <Input
                    id="canvas-audio-receipt"
                    value={providerId}
                    maxLength={200}
                    disabled={busy}
                    onChange={(event) => setProviderId(event.target.value)}
                  />
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  核实供应商原任务已结束后，结束本地等待并释放占用。此操作不会向供应商发送取消请求。
                </p>
              )}
              <div className="flex flex-col gap-2">
                <Label htmlFor="canvas-media-review-note">核实依据</Label>
                <Textarea
                  id="canvas-media-review-note"
                  value={note}
                  maxLength={2000}
                  disabled={busy}
                  onChange={(event) => setNote(event.target.value)}
                />
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={busy}
                  onClick={() => setDecision(null)}
                >
                  返回
                </Button>
                <Button
                  size="sm"
                  variant={
                    decision.action === "CONFIRM_FINISHED"
                      ? "destructive"
                      : "default"
                  }
                  disabled={
                    busy ||
                    !note.trim() ||
                    (decision.action === "RECORD_RECEIPT" && !providerId.trim())
                  }
                  onClick={() =>
                    void review(decision.action, decision.revision, note.trim())
                  }
                >
                  确认处理
                </Button>
              </div>
            </div>
          )}
          {(error || actionError) && (
            <p role="alert" className="text-sm text-destructive">
              {actionError ?? error}
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={busy} onClick={onClose}>
            关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
