"use client";

import { useEffect, useState } from "react";
import { Copy, Search, ShieldCheck, Users, UserMinus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import {
  canvasCollaborationApi,
  type CanvasMember,
  type CanvasPerson,
} from "@/lib/api/canvas-collaboration";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { ProjectMembersDialog } from "@/components/project/members-dialog";

type Role = "VIEWER" | "EDITOR" | "MANAGER";
interface Editor {
  person: CanvasPerson;
  role: Role;
  execute: boolean;
  revision: number;
  existing: boolean;
}
const roles = [
  { value: "VIEWER", label: "查看" },
  { value: "EDITOR", label: "编辑" },
  { value: "MANAGER", label: "管理" },
] as const;
const roleOf = (member: CanvasMember): Role =>
  member.effective.canManage
    ? "MANAGER"
    : member.effective.canEdit
      ? "EDITOR"
      : "VIEWER";
const roleLabel = (member: CanvasMember) =>
  !member.effective.canView
    ? "访问已失效"
    : roleOf(member) === "MANAGER"
      ? "可管理"
      : roleOf(member) === "EDITOR"
        ? "可编辑"
        : "可查看";

export function CanvasShareDialog({ onClose }: { onClose: () => void }) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const store = useCanvasStoreApi();
  const requests = useCommandRequestKey();
  const [members, setMembers] = useState<CanvasMember[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [username, setUsername] = useState("");
  const [editor, setEditor] = useState<Editor | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [reload, setReload] = useState(0);
  const [loadedKey, setLoadedKey] = useState("");
  const [projectMembersOpen, setProjectMembersOpen] = useState(false);
  const document = snapshot.document;
  const canManage = snapshot.permissions.canManage;
  const membersReady = loadedKey === `${document.id}:${page}`;
  useEffect(() => {
    const controller = new AbortController();
    void canvasCollaborationApi
      .members(document.id, page, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setMembers(result.list);
          setTotal(result.total);
          setLoadedKey(`${document.id}:${page}`);
        }
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      });
    return () => controller.abort();
  }, [document.id, document.revision, page, reload]);

  const find = async () => {
    setBusy(true);
    setError(null);
    try {
      const member = await canvasCollaborationApi.find(
        document.id,
        username.trim(),
      );
      if (member.protectedOwner)
        throw new Error("此账号已拥有固定管理权限，无需重复授权。");
      setEditor({
        person: member.person,
        role: roleOf(member),
        execute: member.effective.canExecute,
        revision: store.getState().snapshot!.document.revision,
        existing: !!member.granted,
      });
      requests.clear();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const save = async (remove: boolean) => {
    if (!editor) return;
    const body = {
      expectedRevision: editor.revision,
      userId: editor.person.userId,
      canEdit: editor.role !== "VIEWER",
      canExecute: editor.execute,
      canManage: editor.role === "MANAGER",
      remove,
    };
    setBusy(true);
    setError(null);
    try {
      await canvasCollaborationApi.grant(document.id, {
        ...body,
        requestId: requests.forPayload(body),
      });
      setEditor(null);
      setRemoving(false);
      setUsername("");
      requests.clear();
      setReload((value) => value + 1);
      await store.getState().load();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const edit = (member: CanvasMember) => {
    setEditor({
      person: member.person,
      role: roleOf(member),
      execute: member.effective.canExecute,
      revision: document.revision,
      existing: !!member.granted,
    });
    setError(null);
    requests.clear();
  };
  if (projectMembersOpen && document.projectId) return (
    <ProjectMembersDialog key={document.projectId} projectId={document.projectId} onClose={() => {
      setProjectMembersOpen(false);
      setReload((value) => value + 1);
      void store.getState().load();
    }} />
  );
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
        <DialogHeader className="shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <Users className="size-5 text-primary" />
            分享画布
          </DialogTitle>
          <DialogDescription>
            {document.projectId
              ? "项目成员继承项目权限，也可单独设置画布权限。"
              : "仅受邀账号可以打开这张画布。"}
          </DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-6 overflow-y-auto p-1">
          {document.projectId && <Button variant="outline" onClick={() => setProjectMembersOpen(true)}><Users />项目成员</Button>}
          <div className="flex items-center gap-3 rounded-lg border border-border/20 bg-background/70 p-3">
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{document.title}</p>
              <p className="text-xs text-muted-foreground">
                链接只对有权限的账号开放
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                void navigator.clipboard
                  .writeText(window.location.href)
                  .then(() => setCopied(true))
                  .catch((failure) => setError(getApiErrorMessage(failure)))
              }
            >
              <Copy />
              {copied ? "已复制" : "复制链接"}
            </Button>
          </div>
          {!editor && (
            <>
              {canManage && (
                <div className="space-y-2">
                  <Label htmlFor="canvas-share-username">添加成员</Label>
                  <form
                    className="flex gap-2"
                    onSubmit={(event) => {
                      event.preventDefault();
                      if (!busy && username.trim()) void find();
                    }}
                  >
                    <Input
                      id="canvas-share-username"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      maxLength={64}
                      placeholder="输入完整用户名"
                      disabled={busy}
                    />
                    <Button
                      variant="outline"
                      disabled={busy || !username.trim()}
                      type="submit"
                    >
                      <Search />
                      查找
                    </Button>
                  </form>
                </div>
              )}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold">
                  成员{" "}
                  <span className="font-normal text-muted-foreground">
                    {membersReady ? total : "…"}
                  </span>
                </h3>
                {!membersReady && (
                  <p role="status" className="text-sm text-muted-foreground">
                    正在读取成员…
                  </p>
                )}
                {membersReady &&
                  members.map((member) => (
                    <div
                      key={member.person.userId}
                      className="flex items-center gap-3 border-b border-border/20 pb-3"
                    >
                      <span
                        aria-hidden
                        className="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-medium text-primary"
                      >
                        {member.person.name.slice(0, 1)}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">
                          {member.person.name}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {roleLabel(member)}
                          {member.effective.canExecute
                            ? " · 可运行"
                            : ""} ·{" "}
                          {member.source === "OWNER"
                            ? "拥有者或项目管理者"
                            : member.source === "PROJECT"
                              ? "继承项目"
                              : "单独授权"}
                        </p>
                      </div>
                      {canManage && !member.protectedOwner && (
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={busy}
                          onClick={() => edit(member)}
                        >
                          权限
                        </Button>
                      )}
                    </div>
                  ))}
                {total > 40 && (
                  <div className="flex items-center justify-between gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={page === 1}
                      onClick={() => setPage((value) => value - 1)}
                    >
                      上一页
                    </Button>
                    <span className="text-xs text-muted-foreground">
                      第 {page} 页
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={page * 40 >= total}
                      onClick={() => setPage((value) => value + 1)}
                    >
                      下一页
                    </Button>
                  </div>
                )}
              </div>
            </>
          )}
          {editor && (
            <div className="space-y-4">
              <div>
                <p className="text-sm font-semibold">
                  设置 {editor.person.name} 的权限
                </p>
                <p className="text-xs text-muted-foreground">
                  @{editor.person.username}
                </p>
              </div>
              <CanvasSegmentedControl
                label="画布权限"
                value={editor.role}
                options={roles.map((role) => ({ ...role, disabled: busy }))}
                onChange={(role) => setEditor({ ...editor, role })}
                className="w-full"
              />
              <p className="text-sm text-muted-foreground">
                {editor.role === "MANAGER"
                  ? "可以编辑、管理成员和归档画布。"
                  : editor.role === "EDITOR"
                    ? "可以编辑节点和评论；项目内容仍遵守原项目权限。"
                    : "可以查看节点、评论和已授权内容。"}
              </p>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">生成与运行</p>
                  <p className="text-xs text-muted-foreground">
                    运行前显示范围与用量
                  </p>
                </div>
                <CanvasSegmentedControl
                  label="允许运行"
                  value={editor.execute ? "yes" : "no"}
                  compact
                  options={[
                    { value: "no", label: "不允许", disabled: busy },
                    { value: "yes", label: "允许", disabled: busy },
                  ]}
                  onChange={(value) =>
                    setEditor({ ...editor, execute: value === "yes" })
                  }
                />
              </div>
              {editor.existing && (
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={busy}
                  onClick={() => setRemoving(true)}
                >
                  <UserMinus />
                  {document.projectId ? "恢复项目继承" : "移除成员"}
                </Button>
              )}
              {removing && (
                <div className="space-y-3 rounded-lg border border-destructive/30 p-3">
                  <p className="text-sm">
                    {document.projectId
                      ? "移除这项单独授权后，该账号将恢复项目赋予的权限。"
                      : "移除后，该账号将无法继续打开或操作这张画布。"}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={busy}
                      onClick={() => setRemoving(false)}
                    >
                      取消
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={busy}
                      onClick={() => void save(true)}
                    >
                      确认移除授权
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
          <p className="flex items-start gap-2 text-xs text-muted-foreground">
            <ShieldCheck className="size-4 shrink-0" />
            成员只能查看自己有权访问的项目内容和素材。分享画布不会自动公开你的私人素材。
          </p>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter className="shrink-0">
          <Button
            variant="outline"
            disabled={busy}
            onClick={() => {
              if (editor) {
                setEditor(null);
                setRemoving(false);
              } else onClose();
            }}
          >
            {editor ? "返回成员列表" : "关闭"}
          </Button>
          {editor && (
            <Button
              disabled={busy || !canManage || removing}
              onClick={() => void save(false)}
            >
              保存权限
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
