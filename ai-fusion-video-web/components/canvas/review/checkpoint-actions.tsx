"use client";
import { useId, useState } from "react";
import { useRouter } from "next/navigation";
import { Copy, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import {
  canvasCheckpointsApi,
  type CanvasCheckpointPlan,
} from "@/lib/api/canvas-checkpoints";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

const CHANGE = {
  RESTORED: "恢复节点",
  MODIFIED: "恢复修改",
  REMOVED: "移除节点",
};
export function CanvasCheckpointActions({
  id,
  title,
  onApplied,
}: {
  id: string;
  title: string;
  onApplied: () => void;
}) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => !!state.pending || state.saving);
  const store = useCanvasStoreApi();
  const router = useRouter();
  const requests = useCommandRequestKey();
  const [plan, setPlan] = useState<CanvasCheckpointPlan | null>(null);
  const [fork, setFork] = useState<{ id: string; title: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState(40);
  const fieldId = useId();
  const editable = snapshot.permissions.canEdit && !snapshot.document.archived;
  const inspect = async () => {
    setBusy(true);
    setError(null);
    try {
      setPlan(await canvasCheckpointsApi.preview(snapshot.document.id, id));
      setVisible(40);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const restore = async () => {
    if (!plan) return;
    setBusy(true);
    setError(null);
    const body = {
      expectedRevision: plan.expectedRevision,
      digest: plan.digest,
    };
    try {
      await canvasCheckpointsApi.restore(snapshot.document.id, id, {
        ...body,
        requestId: requests.forPayload(body),
      });
      await store.getState().load();
      requests.clear();
      setPlan(null);
      onApplied();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const copy = async () => {
    if (!fork) return;
    setBusy(true);
    setError(null);
    const body = { canvasId: fork.id, title: fork.title.trim() };
    try {
      const created = await canvasCheckpointsApi.fork(
        snapshot.document.id,
        id,
        { ...body, requestId: requests.forPayload(body) },
      );
      requests.clear();
      setFork(null);
      router.push(
        created.projectId
          ? `/projects/${created.projectId}/canvases/${created.id}`
          : `/canvases/${created.id}`,
      );
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={!editable || pending || busy}
          onClick={() => void inspect()}
        >
          <RotateCcw />
          比较与恢复
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={pending || busy}
          onClick={() => {
            setError(null);
            setFork({
              id: crypto.randomUUID(),
              title: `${title} · 副本`.slice(0, 200),
            });
          }}
        >
          <Copy />
          创建画布副本
        </Button>
        {error && !plan && !fork && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </div>
      <Dialog
        open={!!plan}
        onOpenChange={(open) => {
          if (!open && !busy) setPlan(null);
        }}
      >
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
          <DialogHeader className="shrink-0">
            <DialogTitle>恢复“{plan?.title}”</DialogTitle>
            <DialogDescription>
              恢复此版本的节点、布局和连线。剧本、分镜等项目内容使用当前版本。
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 overflow-y-auto px-1 py-2">
            {plan && (
              <div className="flex flex-col gap-4">
                <p className="text-sm">
                  {plan.nodes.length} 个节点变化 · {plan.upsertedEdges}{" "}
                  条连线恢复 · {plan.removedEdges} 条连线移除
                </p>
                {plan.resources.length > 0 && (
                  <p className="rounded-lg bg-muted/30 p-3 text-sm text-muted-foreground">
                    {plan.resources.length}{" "}
                    个项目内容在保存后有新版本，恢复后会继续引用这些当前内容。历史正文可以在此命名版本中查看。
                  </p>
                )}
                {!plan.hasChanges && (
                  <p className="text-sm text-muted-foreground">
                    当前画布已与此版本一致。
                  </p>
                )}
                <ul className="divide-y divide-border/20">
                  {plan.nodes.slice(0, visible).map((node) => (
                    <li
                      key={node.id}
                      className="flex items-start justify-between gap-3 py-3"
                    >
                      <span className="min-w-0 break-words text-sm">
                        {node.title}
                      </span>
                      <span
                        className={
                          node.change === "REMOVED"
                            ? "shrink-0 text-xs text-destructive"
                            : "shrink-0 text-xs text-muted-foreground"
                        }
                      >
                        {CHANGE[node.change]}
                        {node.change === "MODIFIED"
                          ? ` · ${[node.content && "内容配置", node.layout && "布局"].filter(Boolean).join("、")}`
                          : ""}
                      </span>
                    </li>
                  ))}
                </ul>
                {plan.nodes.length > visible && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setVisible((value) => value + 40)}
                  >
                    更多变化
                  </Button>
                )}
              </div>
            )}
          </div>
          {error && (
            <p role="alert" className="shrink-0 text-sm text-destructive">
              {error}
            </p>
          )}
          {plan && plan.expectedRevision !== snapshot.document.revision && (
            <p role="status" className="text-sm text-destructive">
              画布有了新修改，请重新比较。
            </p>
          )}
          <DialogFooter className="shrink-0">
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => setPlan(null)}
            >
              取消
            </Button>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => void inspect()}
            >
              重新比较
            </Button>
            <Button
              disabled={
                busy ||
                pending ||
                !plan?.hasChanges ||
                plan.expectedRevision !== snapshot.document.revision
              }
              onClick={() => void restore()}
            >
              恢复这个版本
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!fork}
        onOpenChange={(open) => {
          if (!open && !busy) setFork(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>创建画布副本</DialogTitle>
            <DialogDescription>
              新画布引用相同的项目内容，可以继续调整节点与连线。剧本、分镜和素材的内容修改仍会同步。
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3 px-1 py-2">
            <Label htmlFor={fieldId}>画布名称</Label>
            <Input
              id={fieldId}
              value={fork?.title ?? ""}
              maxLength={200}
              disabled={busy}
              onChange={(event) =>
                setFork(
                  (current) =>
                    current && { ...current, title: event.target.value },
                )
              }
            />
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => setFork(null)}
            >
              取消
            </Button>
            <Button
              disabled={busy || !fork?.title.trim()}
              onClick={() => void copy()}
            >
              创建副本
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
