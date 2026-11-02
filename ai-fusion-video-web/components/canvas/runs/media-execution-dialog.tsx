"use client";

import { useCallback } from "react";
import { canvasRunApi } from "@/lib/api/canvas-runs";
import { ExecutionDetailDialog } from "./execution-detail-dialog";

export function CanvasMediaExecutionDialog({ runId, nodeId, onClose }: {
  runId: string;
  nodeId: string;
  onClose: () => void;
}) {
  const load = useCallback(() => canvasRunApi.media(runId, nodeId), [runId, nodeId]);
  return <ExecutionDetailDialog title="媒体任务详情" kind="CANVAS_MEDIA" load={load} onClose={onClose}>
    {(data) => <>
      <p className="text-sm">{data.status === "UNKNOWN_SUBMISSION" ? "已停止本地等待" : data.phase}</p>
      {data.providerTaskId && <div className="flex flex-col gap-2 text-xs"><p className="text-muted-foreground">供应商任务编号</p><p className="break-all font-mono">{data.providerTaskId}</p></div>}
    </>}
  </ExecutionDetailDialog>;
}
