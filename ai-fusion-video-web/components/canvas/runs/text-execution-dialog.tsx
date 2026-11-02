"use client";

import { useCallback, useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { canvasRunApi } from "@/lib/api/canvas-runs";
import { ExecutionDetailDialog } from "./execution-detail-dialog";

export function CanvasTextExecutionButton({ runId, nodeId }: { runId: string; nodeId: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const load = useCallback(() => canvasRunApi.text(runId, nodeId), [runId, nodeId]);
  return <>
    <Button size="xs" variant="outline" onClick={() => setOpen(true)}>查看文本生成</Button>
    {open && <ExecutionDetailDialog title="文本生成详情" kind="TEXT" load={load} onClose={() => setOpen(false)}>
      {(data) => <>
        <p className="text-sm">{data.status === "UNKNOWN_SUBMISSION" ? "本次文本生成已停止" : data.phase}</p>
        {data.receivedText !== null && <div className="flex min-w-0 flex-col gap-2">
          <Label htmlFor={id}>已返回文本</Label>
          <Textarea id={id} readOnly rows={8} value={data.receivedText} />
          {!data.completeResponse && <p className="text-xs text-muted-foreground">返回内容不完整，未用于后续任务。</p>}
        </div>}
      </>}
    </ExecutionDetailDialog>}
  </>;
}
