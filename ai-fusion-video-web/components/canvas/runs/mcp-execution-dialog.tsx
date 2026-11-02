"use client";

import { useCallback, useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { canvasMcpApi, mcpEffectNames } from "@/lib/api/canvas-mcp";
import { ExecutionDetailDialog } from "./execution-detail-dialog";

export function CanvasMcpExecutionButton({ runId, nodeId }: { runId: string; nodeId: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const load = useCallback(() => canvasMcpApi.execution(runId, nodeId), [runId, nodeId]);
  return <>
    <Button size="xs" variant="outline" onClick={() => setOpen(true)}>查看 MCP 调用</Button>
    {open && <ExecutionDetailDialog title="MCP 调用详情" kind="MCP" load={load} onClose={() => setOpen(false)}>
      {(data) => <>
        <p className="text-sm">{data.serverName} · {data.toolName} · {mcpEffectNames[data.effect]}</p>
        {data.received?.output !== undefined && <div className="flex min-w-0 flex-col gap-2">
          <Label htmlFor={id}>已返回结果</Label>
          <Textarea id={id} readOnly rows={8} value={typeof data.received.output === "string" ? data.received.output : JSON.stringify(data.received.output, null, 2)} />
        </div>}
        {data.history.length > 0 && <section className="flex flex-col gap-2"><h3 className="text-sm font-medium">历史处理记录</h3>
          {data.history.map((item) => <p key={item.id} className="text-xs text-muted-foreground">{item.note}</p>)}
        </section>}
      </>}
    </ExecutionDetailDialog>}
  </>;
}
