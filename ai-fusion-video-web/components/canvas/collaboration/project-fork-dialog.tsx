"use client";
import { useId, useState } from "react";
import { useRouter } from "next/navigation";
import { FolderGit2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { canvasProjectForksApi, type CanvasProjectForkPreview, type CanvasProjectForkRequest } from "@/lib/api/canvas-project-forks";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { useCanvas } from "../canvas-provider";
import { exportSize } from "../delivery/export-history";

function readRequest(key: string, canvas: string): CanvasProjectForkRequest | null {
  const stored = localStorage.getItem(key);
  if (!stored) return null;
  const data: unknown = JSON.parse(stored);
  if (!data || typeof data !== "object") throw new Error("本地副本请求记录不完整");
  const value = data as Record<string, unknown>;
  if (value.sourceCanvasId !== canvas || typeof value.title !== "string" || typeof value.expectedRevision !== "number"
    || !Number.isSafeInteger(value.expectedRevision) || value.expectedRevision < 1
    || ![value.canvasId, value.requestId].every(id => typeof id === "string" && /^[0-9a-f-]{36}$/.test(id))
    || typeof value.digest !== "string" || !/^[0-9a-f]{64}$/.test(value.digest)) throw new Error("本地副本请求记录不完整");
  return value as unknown as CanvasProjectForkRequest;
}

export function CanvasProjectForkDialog({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const snapshot = useCanvas(state => state.snapshot!);
  const saving = useCanvas(state => state.saving || !!state.pending);
  const userId = useAuthStore(state => state.user!.id);
  const canvas = snapshot.document;
  const key = `canvas-project-fork:${userId}:${canvas.id}`;
  const [initial] = useState(() => {
    try { return { request: readRequest(key, canvas.id), error: null }; }
    catch (failure) { return { request: null, error: getApiErrorMessage(failure) }; }
  });
  const [pending, setPending] = useState(initial.request);
  const [corrupt, setCorrupt] = useState(!!initial.error);
  const [title, setTitle] = useState(initial.request?.title ?? `${canvas.title} · 独立副本`.slice(0, 200));
  const [preview, setPreview] = useState<CanvasProjectForkPreview | null>(null);
  const [error, setError] = useState<string | null>(initial.error);
  const [busy, setBusy] = useState(false);
  const fieldId = useId();
  const fresh = preview?.revision === canvas.revision ? preview : null;
  const openCopy = (id: number, canvasId: string) => {
    localStorage.removeItem(key);
    setPending(null);
    router.push(`/projects/${id}/canvases/${canvasId}`);
    onClose();
  };
  const inspect = async () => {
    setBusy(true); setError(null);
    try { setPreview(await canvasProjectForksApi.preview(canvas.id, canvas.revision)); }
    catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  const submit = async () => {
    if (!pending && !fresh) return;
    setBusy(true); setError(null);
    try {
      await navigator.locks.request(key, { ifAvailable: true }, async lock => {
        if (!lock) throw new Error("另一窗口正在创建此画布的副本，请等待该请求完成。");
        const stored = readRequest(key, canvas.id);
        if (stored && stored.requestId !== pending?.requestId) {
          setPending(stored); setTitle(stored.title);
          throw new Error("已有待确认的副本请求，请先确认启动状态。");
        }
        const request = pending ?? {
          requestId: crypto.randomUUID(), canvasId: crypto.randomUUID(), sourceCanvasId: canvas.id,
          expectedRevision: fresh!.revision, digest: fresh!.digest, title: title.trim(),
        };
        localStorage.setItem(key, JSON.stringify(request));
        setPending(request);
        try {
          const created = await canvasProjectForksApi.copy(request);
          openCopy(created.result.id, created.result.canvasId);
        } catch (failure) {
          // 只有同一请求已明确被拒绝且目标不存在时才能重新确认；断网保留原请求。
          const status = getApiErrorStatus(failure);
          if (status === 400 || status === 409 || status === 422) {
            try {
              const existing = await canvasApi.snapshot(request.canvasId);
              if (existing.document.projectId) {
                openCopy(existing.document.projectId, existing.document.id);
                return;
              }
            } catch (lookupFailure) {
              if (getApiErrorStatus(lookupFailure) === 404) {
                localStorage.removeItem(key); setPending(null); setPreview(null);
              }
            }
          }
          throw failure;
        }
      });
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  return (
    <Dialog open onOpenChange={open => { if (!open && !busy) onClose(); }}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle className="flex items-center gap-2"><FolderGit2 className="size-5 text-primary" />创建独立项目</DialogTitle>
          <DialogDescription>
            {canvas.projectId ? "复制整个项目的当前剧本、分镜和资产，以及当前画布。" : "将画布和使用的独立资产复制到一个新项目。"}
            新项目可独立修改，原内容保留。
          </DialogDescription>
        </DialogHeader>
        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto px-1 py-2">
          <div className="flex flex-col gap-2">
            <Label htmlFor={fieldId}>新项目名称</Label>
            <Input id={fieldId} value={pending?.title ?? title} maxLength={200} disabled={busy || !!pending} onChange={event => setTitle(event.target.value)} />
          </div>
          <p className="text-sm text-muted-foreground">固定结果与媒体会一并保存。成员、评论和运行记录留在原画布；正在执行的任务不会随副本启动。</p>
          {pending ? (
            <p role="status" className="rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">原副本请求尚待确认。继续会查询或完成同一次操作，刷新或关闭后也可回来继续。</p>
          ) : fresh ? (
            <div className="flex flex-col gap-3">
              <dl className="grid grid-cols-2 gap-3 rounded-lg border border-border/30 bg-background/70 p-3 text-sm">
                <dt className="text-muted-foreground">画布节点</dt><dd className="text-right tabular-nums">{fresh.nodes}</dd>
                <dt className="text-muted-foreground">项目内容</dt><dd className="text-right tabular-nums">{fresh.businessObjects}</dd>
                <dt className="text-muted-foreground">媒体文件</dt><dd className="text-right tabular-nums">{fresh.mediaFiles} · {exportSize(fresh.mediaBytes)}</dd>
                <dt className="text-muted-foreground">保存到</dt><dd className="text-right">{fresh.ownership.name} · {fresh.ownership.type === 2 ? "团队" : "个人"}</dd>
              </dl>
              {fresh.ownership.type === 2 && <p className="text-sm text-muted-foreground">该团队成员可读取新项目及其复制的素材。</p>}
              {fresh.historicalObjects > 0 && <p className="text-sm text-muted-foreground">同时保留 {fresh.historicalObjects} 个固定结果依赖的历史对象。</p>}
              {fresh.warnings.map(warning => <p key={warning} className="text-sm text-amber-700 dark:text-amber-300">{warning}</p>)}
            </div>
          ) : <p className="text-sm text-muted-foreground">检查后可查看复制范围和缺失素材。</p>}
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
          {corrupt && <Button variant="outline" onClick={() => { localStorage.removeItem(key); setCorrupt(false); setError(null); }}>清除损坏的本地记录</Button>}
        </div>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={busy} onClick={onClose}>{pending ? "稍后继续" : "取消"}</Button>
          {!pending && <Button variant={fresh ? "outline" : "default"} disabled={busy || saving || corrupt} onClick={() => void inspect()}>检查范围</Button>}
          {(pending || fresh) && <Button disabled={busy || saving || !title.trim() || (!pending && !!fresh?.warnings.length)} onClick={() => void submit()}>
            {busy && <Loader2 className="animate-spin" />}{pending ? "确认启动状态" : "创建独立项目"}
          </Button>}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
