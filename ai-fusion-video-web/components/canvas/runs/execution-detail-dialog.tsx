"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CanvasDialogContent } from "@/components/canvas/shared/canvas-dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { ExecutionRecordButton } from "@/components/dashboard/notification-panel/execution-record-button";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import type { ExecutionRecordKind } from "@/lib/api/execution-records";
import { userFacingError } from "@/lib/user-facing-error";

interface ExecutionDetail {
  id: string;
  status: string;
  error: string | null;
  remoteCapacityHeld: boolean;
}

export function ExecutionDetailDialog<T extends ExecutionDetail>({ title, kind, load, onClose, children }: {
  title: string;
  kind: ExecutionRecordKind;
  load: () => Promise<T>;
  onClose: () => void;
  children: (data: T) => ReactNode;
}) {
  const [loaded, setLoaded] = useState<{ load: typeof load; value: T } | null>(null);
  const [failure, setFailure] = useState<{ load: typeof load; message: string } | null>(null);
  const [revision, setRevision] = useState(0);
  const data = loaded?.load === load ? loaded.value : null;
  const error = failure?.load === load ? failure.message : null;
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const read = async () => {
      let complete = false;
      try {
        const value = await load();
        if (active) {
          setLoaded({ load, value });
          setFailure(null);
        }
        complete = ["SUCCEEDED", "FAILED", "CANCELLED", "CONFLICT"].includes(value.status)
          || value.status === "UNKNOWN_SUBMISSION" && !value.remoteCapacityHeld;
      } catch (problem) {
        if (active) {
          setFailure({ load, message: getApiErrorMessage(problem) });
        }
        complete = [401, 403, 404].includes(getApiErrorStatus(problem) ?? 0);
      } finally {
        if (active && !complete) {
          timer = setTimeout(() => void read(), 3000);
        }
      }
    };
    void read();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [load, revision]);
  return <Dialog open onOpenChange={(open) => {
    if (!open) {
      onClose();
    }
  }}>
    <CanvasDialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
      <DialogHeader className="shrink-0"><DialogTitle>{title}</DialogTitle><DialogDescription>查看原请求的状态与已保存结果。</DialogDescription></DialogHeader>
      <DialogScrollBody className="flex flex-col gap-4">
        {!data && !error && <div role="status" aria-label="加载任务详情" className="flex justify-center p-4"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>}
        {data && <>
          {data.status === "UNKNOWN_SUBMISSION" && <div className="rounded-lg border border-border/30 bg-muted/40 p-3 text-sm">
            <p className="font-medium">远端结果未知</p>
            <p className="mt-2 text-muted-foreground">{data.remoteCapacityHeld ? "系统正在自动处理原请求。" : "已结束本地等待，原请求不会自动重发。"}</p>
          </div>}
          {data.error && <p className="text-sm text-destructive">{userFacingError(data.error)}</p>}
          {children(data)}
          {["UNKNOWN_SUBMISSION", "NEEDS_REVIEW", "FAILED", "CONFLICT"].includes(data.status) && <div><ExecutionRecordButton target={{ kind, id: data.id }} onOpen={onClose} /></div>}
        </>}
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      </DialogScrollBody>
      <DialogFooter className="shrink-0"><Button variant="ghost" onClick={() => setRevision((value) => value + 1)}>刷新</Button><Button variant="outline" onClick={onClose}>关闭</Button></DialogFooter>
    </CanvasDialogContent>
  </Dialog>;
}
