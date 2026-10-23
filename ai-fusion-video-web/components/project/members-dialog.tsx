"use client";

import { useEffect, useState } from "react";
import { History, Search, ShieldCheck, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CanvasSegmentedControl } from "@/components/canvas/workspace/segmented-control";
import { projectMembersApi, projectRoleLabel, type ProjectMember, type ProjectMemberAudit, type ProjectMembers } from "@/lib/api/project-members";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { ProjectMemberEditor, type ProjectMemberEditorState } from "./member-editor";

export function ProjectMembersDialog({ projectId, onClose }: { projectId: number; onClose: () => void }) {
  const [data, setData] = useState<ProjectMembers | null>(null);
  const [history, setHistory] = useState<{ list: ProjectMemberAudit[]; total: number } | null>(null);
  const [tab, setTab] = useState("members");
  const [page, setPage] = useState(1);
  const [username, setUsername] = useState("");
  const [editor, setEditor] = useState<ProjectMemberEditorState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const requests = useCommandRequestKey();
  useEffect(() => {
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const result = await projectMembersApi.list(projectId, tab === "members" ? page : 1, controller.signal);
        if (controller.signal.aborted) return;
        setData(result);
        if (tab === "history" && result.canManage) {
          const audit = await projectMembersApi.history(projectId, page, controller.signal);
          if (!controller.signal.aborted) setHistory(audit);
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 10_000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [projectId, tab, page, reload]);
  const edit = (member: ProjectMember, revision = data!.revision) => {
    setEditor({ member, revision, role: String(member.role ?? 4), removing: false });
    setError(null);
  };
  const find = async () => {
    setBusy(true); setError(null);
    try {
      const member = await projectMembersApi.find(projectId, username.trim());
      if (member.protectedOwner) throw new Error("拥有者权限不需要另行设置");
      edit(member);
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  const reread = async () => {
    if (!editor) return;
    setBusy(true); setError(null);
    try {
      const current = await projectMembersApi.list(projectId, page);
      const member = await projectMembersApi.find(projectId, editor.member.person.username);
      setData(current); edit(member, current.revision);
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  const save = async () => {
    if (!editor) return;
    const payload = { expectedRevision: editor.revision, userId: editor.member.person.userId, role: editor.removing ? null : Number(editor.role) };
    setBusy(true); setError(null);
    try {
      await projectMembersApi.change(projectId, { ...payload, requestId: requests.forPayload(payload) });
      requests.clear(); setEditor(null); setUsername(""); setReload((value) => value + 1);
    } catch (failure) { setError(getApiErrorMessage(failure)); setReload((value) => value + 1); }
    finally { setBusy(false); }
  };
  const stale = !!editor && !!data && editor.revision !== data.revision;
  const total = tab === "history" ? history?.total ?? 0 : data?.total ?? 0;
  return (
    <Dialog open onOpenChange={(open) => { if (!open && !busy) onClose(); }}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>项目成员</DialogTitle>
          <DialogDescription>统一管理项目内容与画布的访问权限。</DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-5 overflow-y-auto px-1 py-1">
          {!editor && data?.canManage && <CanvasSegmentedControl label="成员信息" value={tab} options={[
            { value: "members", label: "成员", icon: Users }, { value: "history", label: "权限记录", icon: History },
          ]} onChange={(value) => { setTab(value); setPage(1); setHistory(null); setError(null); }} />}
          {!editor && tab === "members" && (
            <>
              {data?.canManage && (
                <form className="space-y-2" onSubmit={(event) => { event.preventDefault(); void find(); }}>
                  <Label htmlFor="project-member-username">添加成员</Label>
                  <div className="flex gap-2">
                    <Input id="project-member-username" value={username} placeholder="输入完整用户名" onChange={(event) => setUsername(event.target.value)} disabled={busy} />
                    <Button type="submit" variant="outline" disabled={busy || !username.trim()}><Search />查找</Button>
                  </div>
                </form>
              )}
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">单独授权 · {data?.total ?? "…"} 位</p>
                {!data && !error && <p className="text-sm text-muted-foreground">正在读取成员…</p>}
                {data?.list.map((member) => (
                  <div key={member.person.userId} className="flex items-center justify-between gap-3 border-b border-border/20 py-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{member.person.name}{!member.enabled && " · 账号已停用"}</p>
                      <p className="truncate text-xs text-muted-foreground">@{member.person.username} · {projectRoleLabel(member.role)}</p>
                    </div>
                    {member.protectedOwner ? <ShieldCheck className="size-4 text-primary" aria-label="拥有者" /> : data.canManage && (
                      <Button variant="outline" size="sm" onClick={() => edit(member)}>设置权限</Button>
                    )}
                  </div>
                ))}
                {data?.total === 0 && <p className="text-sm text-muted-foreground">还没有单独授权的项目成员。</p>}
              </div>
              <p className="text-xs text-muted-foreground">{data?.teamOwned ? "团队拥有者和管理员继承管理权限，其他团队成员继承查看权限。" : "同一团队成员可能继承查看权限；上方仅列出拥有者和单独授权的账号。"}团队权限在团队设置中管理。</p>
            </>
          )}
          {!editor && tab === "history" && data?.canManage && (
            <div className="space-y-3">
              {history?.list.map(({ change, actor, member }) => (
                <div key={change.commandId} className="space-y-1 border-b border-border/20 py-3">
                  <p className="text-sm"><span className="font-medium">{actor.name}</span> 调整了 {member.name} 的权限</p>
                  <p className="text-sm text-muted-foreground">{projectRoleLabel(change.previousRole)} → {projectRoleLabel(change.currentRole)}</p>
                  {change.removedCanvasGrants > 0 && <p className="text-xs text-muted-foreground">同时撤销 {change.removedCanvasGrants} 项画布授权</p>}
                  <p className="text-xs text-muted-foreground">{new Date(change.createdAt).toLocaleString()} · 版本 {change.revision}</p>
                </div>
              ))}
              {!history && <p className="text-sm text-muted-foreground">正在读取权限记录…</p>}
              {history?.total === 0 && <p className="text-sm text-muted-foreground">还没有权限变更记录。</p>}
            </div>
          )}
          {!editor && total > 40 && <div className="flex items-center justify-end gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>上一页</Button>
            <span className="text-xs text-muted-foreground">第 {page} 页</span>
            <Button variant="outline" size="sm" disabled={page * 40 >= total} onClick={() => setPage((value) => value + 1)}>下一页</Button>
          </div>}
          {editor && <ProjectMemberEditor editor={editor} onChange={setEditor} busy={busy} stale={stale} onReload={() => void reread()} />}
          {error && <div className="space-y-2" role="alert">
            <p className="text-sm text-destructive">{error}</p>
            {!editor && <Button variant="outline" size="sm" onClick={() => { setError(null); setReload((value) => value + 1); }}>重新读取</Button>}
          </div>}
        </div>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={busy} onClick={() => editor ? setEditor(null) : onClose()}>{editor ? "返回成员列表" : "关闭"}</Button>
          {editor && <Button variant={editor.removing ? "destructive" : "default"} disabled={busy || stale || !data?.canManage || (!editor.removing && Number(editor.role) === editor.member.role)} onClick={() => void save()}>
            {busy ? "保存中…" : editor.removing ? "确认移除" : "保存权限"}
          </Button>}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
