"use client";

import { useEffect, useState } from "react";
import { Plus, Pencil, Trash2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useAuthStore } from "@/lib/store/auth-store";
import { canvasMcpApi, mcpEffectNames, type McpContract, type McpWorkflowTool } from "@/lib/api/canvas-mcp";
import { McpContractEditor } from "./contract-editor";

export function McpContractsDialog({ onClose, onChanged, serverName }: { onClose: () => void; onChanged?: () => void; serverName?: string }) {
  const userId = useAuthStore((state) => state.user?.id);
  const { confirm } = useConfirm();
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState<{ key: string; contracts: McpContract[]; tools: McpWorkflowTool[]; error: string | null } | null>(null);
  const [editing, setEditing] = useState<{ userId: number; contract: McpContract | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const key = `${userId}:${attempt}`;
  const data = loaded?.key === key ? loaded : null;
  useEffect(() => {
    if (!userId) return;
    let active = true;
    void Promise.allSettled([canvasMcpApi.list(), canvasMcpApi.tools()]).then(([contracts, tools]) => {
      if (!active) return;
      const failure = contracts.status === "rejected" ? contracts.reason : tools.status === "rejected" ? tools.reason : null;
      setLoaded({ key, contracts: contracts.status === "fulfilled" ? contracts.value : [], tools: tools.status === "fulfilled" ? tools.value : [],
        error: failure ? failure instanceof Error ? failure.message : "MCP 目录加载失败" : null });
    });
    return () => { active = false; };
  }, [key, userId]);
  const remove = async (contract: McpContract) => {
    if (!await confirm({ title: "移除画布操作", description: `移除「${contract.name}」后不能再发起新调用，已有执行记录仍保留。`, variant: "destructive", confirmText: "移除" })) return;
    setBusy(true); setActionError(null);
    try { await canvasMcpApi.remove(contract.id, contract.version); setAttempt((value) => value + 1); onChanged?.(); }
    catch (failure) { setActionError(failure instanceof Error ? failure.message : "移除失败"); }
    finally { setBusy(false); }
  };
  return (
    <Dialog open onOpenChange={(open) => { if (!open && !busy) onClose(); }}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0"><DialogTitle>MCP 画布操作</DialogTitle><DialogDescription>登记你允许画布使用的工具。操作只使用当前账号的连接授权。</DialogDescription></DialogHeader>
        <DialogScrollBody className="space-y-4">
          {editing && editing.userId === userId && data ? <McpContractEditor key={editing.contract?.id ?? "new"} contract={editing.contract}
            tools={data.tools.filter((tool) => !serverName || tool.serverName === serverName)} onCancel={() => setEditing(null)}
            onSaved={() => { setEditing(null); setAttempt((value) => value + 1); onChanged?.(); }} /> : (
            <>
              <div className="flex justify-between gap-2">
                <Button variant="ghost" onClick={() => setAttempt((value) => value + 1)} disabled={!data || busy}><RefreshCw />刷新目录</Button>
                <Button disabled={!data?.tools.length || busy || !userId} onClick={() => setEditing({ userId: userId!, contract: null })}><Plus />登记操作</Button>
              </div>
              {!data ? <p className="text-sm text-muted-foreground">正在读取已连接的 MCP 工具…</p> : null}
              {data?.error ? <p role="alert" className="text-sm text-destructive">{data.error}</p> : null}
              {data && !data.error && !data.tools.length ? <p className="text-sm text-muted-foreground">先在系统设置的“智能体”页面连接 MCP 服务，再登记可用于画布的操作。</p> : null}
              {data?.contracts.filter((contract) => !serverName || contract.definition.tool.serverName === serverName).map((contract) => (
                <div key={contract.id} className="flex items-center justify-between gap-3 border-b border-border/20 py-3">
                  <div className="min-w-0"><p className="truncate text-sm font-medium">{contract.name}{!contract.enabled ? " · 已停用" : ""}</p>
                    <p className="text-xs text-muted-foreground">{contract.definition.tool.serverName} · {mcpEffectNames[contract.definition.effect]} · 版本 {contract.version + 1}</p></div>
                  <div className="flex shrink-0 gap-2">
                    <Button variant="ghost" size="icon-sm" aria-label={`编辑${contract.name}`} disabled={busy || !userId || !data.tools.some((tool) => tool.toolName === contract.definition.tool.toolName)} onClick={() => setEditing({ userId: userId!, contract })}><Pencil /></Button>
                    <Button variant="destructive-ghost" size="icon-sm" aria-label={`移除${contract.name}`} disabled={busy} onClick={() => void remove(contract)}><Trash2 /></Button>
                  </div>
                </div>
              ))}
              {actionError ? <p role="alert" className="text-sm text-destructive">{actionError}</p> : null}
            </>
          )}
        </DialogScrollBody>
        {!editing ? <DialogFooter className="shrink-0"><Button variant="outline" onClick={onClose} disabled={busy}>关闭</Button></DialogFooter> : null}
      </DialogContent>
    </Dialog>
  );
}
