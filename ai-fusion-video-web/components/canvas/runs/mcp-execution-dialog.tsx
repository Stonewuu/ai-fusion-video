"use client";

import { useEffect, useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { canvasMcpApi, mcpEffectNames, type McpExecution, type McpReviewAction } from "@/lib/api/canvas-mcp";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { runLabels } from "@/lib/canvas/run-state";
import { useCanvasRuns } from "./canvas-runs-provider";
import { userFacingError } from "@/lib/user-facing-error";

const actionLabels: Record<McpReviewAction, string> = { USE_RECEIVED: "采用原结果", CONFIRM_FINISHED: "确认原调用已结束" };
export function CanvasMcpExecutionButton({ runId, nodeId }: { runId: string; nodeId: string }) {
  const [open, setOpen] = useState(false);
  return <><Button size="xs" variant="outline" onClick={() => setOpen(true)}>查看 MCP 调用</Button>
    {open ? <McpExecutionDialog runId={runId} nodeId={nodeId} onClose={() => setOpen(false)} /> : null}</>;
}
function McpExecutionDialog({ runId, nodeId, onClose }: { runId: string; nodeId: string; onClose: () => void }) {
  const runtime = useCanvasRuns();
  const id = useId();
  const [data, setData] = useState<McpExecution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [decision, setDecision] = useState<{ requestId: string; action: McpReviewAction; revision: number } | null>(null);
  const [note, setNote] = useState("");
  const [finished, setFinished] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      let stopped = false;
      try {
        const result = await canvasMcpApi.execution(runId, nodeId);
        if (active) { setData(result); setError(null); }
        stopped = ["SUCCEEDED", "FAILED", "CANCELLED"].includes(result.status);
      } catch (failure) {
        if (active) {
          setError(getApiErrorMessage(failure));
          if ([401, 403, 404].includes(getApiErrorStatus(failure) ?? 0)) { setData(null); stopped = true; }
        }
      } finally { if (active && !stopped) timer = setTimeout(() => void load(), 3000); }
    };
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [nodeId, runId, refresh]);
  const submit = async () => {
    if (!decision || !runtime.scope || busy) return;
    setBusy(true); setActionError(null);
    try {
      setData(await canvasMcpApi.review(runId, nodeId, { requestId: decision.requestId, expectedRevision: decision.revision,
        action: decision.action, note: note.trim(), providerFinished: finished, scope: runtime.scope }));
      setDecision(null); setNote(""); setFinished(false); setRefresh((value) => value + 1); runtime.reload();
    } catch (failure) { setActionError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  const output = data?.received?.output;
  return (
    <Dialog open onOpenChange={(open) => { if (!open && !busy) onClose(); }}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0"><DialogTitle>{data?.name ?? "MCP 调用"}</DialogTitle><DialogDescription>查看原调用、结果和核实记录。</DialogDescription></DialogHeader>
        <DialogScrollBody className="space-y-4">
          {data ? <>
            <div className="space-y-2"><p className="text-sm font-medium">{runLabels[data.status]}</p>
              <p className="text-xs text-muted-foreground">{data.serverName} · {data.toolName} · {mcpEffectNames[data.effect]}</p>
              {data.error ? <p className="text-sm text-destructive">{userFacingError(data.error)}</p> : null}
              {data.remoteCapacityHeld ? <p className="text-sm text-muted-foreground">原调用尚未核实结束，后续同类操作正在等待；系统不会自动重发。</p> : null}</div>
            {output !== undefined ? <div className="flex flex-col gap-2"><label htmlFor={`${id}-output`} className="text-sm font-medium">已返回结果</label>
              <Textarea id={`${id}-output`} readOnly rows={8} value={typeof output === "string" ? output : JSON.stringify(output, null, 2)} /></div> : null}
            {data.actions.length && !decision ? <div className="flex flex-wrap gap-2">{data.actions.map((action) => <Button key={action} size="sm" variant="outline"
              onClick={() => { setDecision({ requestId: crypto.randomUUID(), action, revision: data.revision }); setNote(""); setFinished(false); setActionError(null); }}>{actionLabels[action]}</Button>)}</div> : null}
            {decision ? <section className="flex flex-col gap-3 rounded-lg border border-border/30 bg-background/70 p-3" aria-label="核实 MCP 原调用">
              <p className="text-sm font-medium">{actionLabels[decision.action]}</p>
              {decision.revision !== data.revision ? <p role="alert" className="text-sm text-destructive">任务状态已更新，请先取消此次确认，再查看最新结果。</p> : null}
              <div className="flex flex-col gap-2"><label htmlFor={`${id}-note`} className="text-sm font-medium">核实依据</label>
                <Textarea id={`${id}-note`} rows={3} value={note} maxLength={2000} disabled={busy} onChange={(event) => setNote(event.target.value)} /></div>
              {decision.action === "CONFIRM_FINISHED" ? <label className="flex items-start gap-2 text-sm"><Checkbox checked={finished} disabled={busy} onCheckedChange={(value) => setFinished(value === true)} />已核实远端操作结束；此决定不会重发请求或撤回外部修改。</label> : null}
              {actionError ? <p role="alert" className="text-sm text-destructive">{actionError}</p> : null}
              <div className="flex justify-end gap-2"><Button variant="outline" size="sm" disabled={busy} onClick={() => setDecision(null)}>取消</Button>
                <Button size="sm" disabled={busy || !note.trim() || decision.revision !== data.revision || decision.action === "CONFIRM_FINISHED" && !finished}
                  onClick={() => void submit()}>{busy ? "提交中…" : "确认"}</Button></div>
            </section> : null}
            {data.history.length ? <section className="space-y-2"><h3 className="text-sm font-medium">核实记录</h3>{data.history.map((item) =>
              <p key={item.id} className="text-xs text-muted-foreground">{actionLabels[item.action]} · {item.note}</p>)}</section> : null}
          </> : <p className="text-sm text-muted-foreground">正在读取原调用…</p>}
          {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
        </DialogScrollBody>
        <DialogFooter className="shrink-0"><Button variant="outline" disabled={busy} onClick={() => setRefresh((value) => value + 1)}>刷新</Button><Button variant="outline" disabled={busy} onClick={onClose}>关闭</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
