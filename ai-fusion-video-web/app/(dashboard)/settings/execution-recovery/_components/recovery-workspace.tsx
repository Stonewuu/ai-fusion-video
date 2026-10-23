"use client";

import { useEffect, useState } from "react";
import { ClipboardCheck, History, RefreshCw, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CanvasSegmentedControl } from "@/components/canvas/workspace/segmented-control";
import { useAuthStore } from "@/lib/store/auth-store";
import { executionRecoveryApi, recoveryKindNames, type RecoveryExecution, type RecoveryHistory } from "@/lib/api/execution-recovery";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { settingsTypography } from "../../_shared";
import { ExecutionRecoveryDialog } from "@/components/dashboard/execution-recovery/recovery-dialog";

export function ExecutionRecoveryWorkspace() {
  const user = useAuthStore(state => state.user);
  const isAdmin = !!user?.roles.includes("admin");
  const [tab, setTab] = useState<"pending" | "history">("pending");
  const [page, setPage] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [loaded, setLoaded] = useState<{ key: string; items: RecoveryExecution[]; history: RecoveryHistory[]; more: boolean }>();
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState<RecoveryExecution | null>(null);
  const [opening, setOpening] = useState(false);
  const key = `${user?.id}:${tab}:${page}:${refresh}`;
  const current = loaded?.key === key ? loaded : null;
  const reload = () => setRefresh(value => value + 1);
  useEffect(() => {
    if (!isAdmin) return;
    const controller = new AbortController();
    const load = async () => {
      try {
        const result = tab === "pending" ? await executionRecoveryApi.list(page, controller.signal) : null;
        const history = tab === "history" ? await executionRecoveryApi.history(page, controller.signal) : [];
        if (!controller.signal.aborted) { setLoaded({ key, items: result?.items ?? [], history, more: result ? result.hasMore : history.length === 50 }); setError(null); }
      } catch (failure) { if (!controller.signal.aborted) setError(getApiErrorMessage(failure)); }
    };
    void load();
    return () => controller.abort();
  }, [isAdmin, key, page, tab]);
  const open = async (item: RecoveryExecution) => {
    if (opening) return;
    setOpening(true);
    try { setTarget(await executionRecoveryApi.get(item)); setError(null); }
    catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setOpening(false); }
  };
  return <div className="flex w-full flex-col gap-6">
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className={settingsTypography.pageTitle}>执行核实</h1>
        <p className={settingsTypography.pageDescription}>核实中断的智能体模型调用，以及账号或权限失效后留下的任务，释放已确认结束的额度。</p>
      </div>
      {isAdmin && <Button variant="outline" onClick={reload}><RefreshCw />刷新</Button>}
    </header>
    {!isAdmin ? <p className="text-sm text-muted-foreground">此入口仅限平台管理员。</p> : <>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <CanvasSegmentedControl label="核实记录范围" value={tab} options={[
          { value: "pending", label: "待核实", icon: ClipboardCheck }, { value: "history", label: "核实记录", icon: History },
        ]} onChange={value => { setTab(value); setPage(0); }} />
        <p className="text-xs text-muted-foreground">第 {page + 1} 页 · 每页最多 50 项</p>
      </div>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {!current && !error && <p aria-live="polite" className="text-sm text-muted-foreground">正在读取执行状态…</p>}
      {current && (tab === "pending" ? <section className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm">
        {!current.items.length ? <p className="p-6 text-sm text-muted-foreground">当前没有待核实的远端调用。</p> : <ul className="divide-y divide-border/20">
          {current.items.map(item => <li key={`${item.kind}:${item.id}`} className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 flex-col gap-2">
              <div className="flex flex-wrap items-center gap-2 text-sm font-medium"><ShieldCheck className="size-4 text-primary" />{recoveryKindNames[item.kind]}
                <span className="rounded-full bg-muted px-2 py-1 text-xs font-normal text-muted-foreground">{item.capacityHeld ? "额度保留中" : "额度已释放"}</span>
              </div>
              <p className="text-xs text-muted-foreground">用户 {item.actorUserId}{item.actorTeamId ? ` · 团队 ${item.actorTeamId}` : " · 个人"}{item.projectId ? ` · 项目 ${item.projectId}` : ""}</p>
              <p className="break-all font-mono text-xs text-muted-foreground">{item.id}</p>
              <p className="text-xs text-muted-foreground">{item.reason}</p>
            </div>
            <Button variant="outline" disabled={!item.canResolve || opening} onClick={() => void open(item)}>核实</Button>
          </li>)}
        </ul>}
      </section> : <section className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm">
        {!current.history.length ? <p className="p-6 text-sm text-muted-foreground">尚无管理员核实记录。</p> : <ul className="divide-y divide-border/20">
          {current.history.map(item => <li key={item.decisionId} className="flex flex-col gap-2 p-4">
            <p className="text-sm font-medium">{recoveryKindNames[item.kind]} · 已确认结束</p>
            <p className="text-xs text-muted-foreground">管理员 {item.adminUserId} · 原用户 {item.actorUserId} · {new Date(item.createdAt).toLocaleString("zh-CN")}</p>
            <p className="whitespace-pre-wrap break-words text-sm">{item.note}</p>
            <p className="break-all font-mono text-xs text-muted-foreground">任务：{item.jobId}</p>
          </li>)}
        </ul>}
      </section>)}
      <div className="flex justify-end gap-2">
        <Button variant="outline" disabled={page === 0} onClick={() => setPage(value => value - 1)}>上一页</Button>
        <Button variant="outline" disabled={!current?.more} onClick={() => setPage(value => value + 1)}>下一页</Button>
      </div>
      {target && user && <ExecutionRecoveryDialog key={`${user.id}:${target.kind}:${target.id}`} target={target} adminUserId={user.id}
        onClose={() => { setTarget(null); reload(); }} onCompleted={() => { setTarget(null); reload(); }} />}
    </>}
  </div>;
}
