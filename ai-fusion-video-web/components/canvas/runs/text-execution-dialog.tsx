"use client";
import { userFacingError } from "@/lib/user-facing-error";

import { useEffect, useId, useState } from "react";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { Button } from "@/components/ui/button";
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
  type CanvasTextExecution,
  type CanvasTextReviewAction,
} from "@/lib/api/canvas-runs";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useCanvasRuns } from "./canvas-runs-provider";

const labels: Record<CanvasTextReviewAction, string> = {
  USE_RECEIVED: "采用已返回文本",
  CONFIRM_FINISHED: "确认原调用已结束",
};

export function CanvasTextExecutionButton({
  runId,
  nodeId,
}: {
  runId: string;
  nodeId: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button size="xs" variant="outline" onClick={() => setOpen(true)}>
        查看文本生成
      </Button>
      {open && (
        <CanvasTextExecutionDialog
          runId={runId}
          nodeId={nodeId}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function CanvasTextExecutionDialog({
  runId,
  nodeId,
  onClose,
}: {
  runId: string;
  nodeId: string;
  onClose: () => void;
}) {
  const runtime = useCanvasRuns();
  const noteId = useId();
  const [data, setData] = useState<CanvasTextExecution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [decision, setDecision] = useState<{
    action: CanvasTextReviewAction;
    revision: number;
  } | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      let stopped = false;
      try {
        const result = await canvasRunApi.text(runId, nodeId);
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

  const review = async () => {
    if (!runtime.scope || !decision || busy) return;
    setBusy(true);
    setActionError(null);
    try {
      const result = await canvasRunApi.reviewText(runId, nodeId, {
        expectedRevision: decision.revision,
        action: decision.action,
        note: note.trim(),
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
          <DialogTitle>文本生成详情</DialogTitle>
          <DialogDescription>
            查看原调用与已保存的响应，核实后继续处理。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-4">
          {data && (
            <>
              <p className="text-sm font-medium">{data.phase}</p>
              {data.error && (
                <p className="text-sm text-destructive">{userFacingError(data.error)}</p>
              )}
              {data.remoteCapacityHeld && (
                <p className="text-xs text-muted-foreground">
                  模型仍未返回明确结果，请先核对这次任务；系统不会重复生成。
                </p>
              )}
              {data.receivedText !== null && (
                <div className="flex flex-col gap-2">
                  <Label htmlFor={`${noteId}-result`}>已返回的文本</Label>
                  <Textarea
                    id={`${noteId}-result`}
                    readOnly
                    rows={8}
                    value={data.receivedText}
                  />
                  {!data.completeResponse && (
                    <p className="text-xs text-muted-foreground">
                      模型没有正常完成这段内容，请检查后决定是否采用。
                    </p>
                  )}
                </div>
              )}
              {!decision && (
                <div className="flex flex-wrap gap-2">
                  {data.actions.map((action) => (
                    <Button
                      key={action}
                      size="sm"
                      variant={
                        action === "CONFIRM_FINISHED"
                          ? "destructive"
                          : "outline"
                      }
                      disabled={busy || !!error}
                      onClick={() =>
                        setDecision({ action, revision: data.revision })
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
            <div className="space-y-4 rounded-lg border border-border/20 bg-background/70 p-3">
              <p className="text-sm font-medium">{labels[decision.action]}</p>
              <p className="text-xs text-muted-foreground">
                {decision.action === "USE_RECEIVED"
                  ? "采用当前显示的文本并继续后续步骤，不会再次调用模型。"
                  : "请先在供应商处确认原调用已经结束。确认后结束本地等待并释放额度，此操作不会重发或取消远端请求。"}
              </p>
              <div className="flex flex-col gap-2">
                <Label htmlFor={noteId}>核实依据</Label>
                <Textarea
                  id={noteId}
                  value={note}
                  maxLength={2000}
                  disabled={busy}
                  onChange={(event) => setNote(event.target.value)}
                />
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
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
                  disabled={busy || !note.trim() || !!error}
                  onClick={() => void review()}
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
