"use client";

import { useEffect, useState } from "react";
import { LocateFixed, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import {
  canvasCollaborationApi,
  type CanvasPresenceSnapshot,
} from "@/lib/api/canvas-collaboration";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

export function CanvasPresenceBar() {
  const store = useCanvasStoreApi();
  const canvasId = useCanvas((state) => state.snapshot!.document.id);
  const scope = useResourceChangeStore((state) => state.scope);
  const [presence, setPresence] = useState<CanvasPresenceSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const sessionId = crypto.randomUUID();
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const state = store.getState();
        if (!state.snapshot || state.sync === "denied") return;
        const result = await canvasCollaborationApi.heartbeat(
          canvasId,
          sessionId,
          state.selection.slice(0, 20),
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setPresence(result);
        setError(null);
        const current = store.getState().snapshot?.permissions;
        if (
          current &&
          JSON.stringify(current) !== JSON.stringify(result.permissions)
        )
          await store.getState().load(controller.signal);
      } catch (failure) {
        if (!controller.signal.aborted) {
          setError(getApiErrorMessage(failure));
          setPresence(null);
        }
      } finally {
        if (!controller.signal.aborted)
          timer = setTimeout(() => void tick(), 10_000);
      }
    };
    void tick();
    return () => {
      controller.abort();
      clearTimeout(timer);
      // 存在信息有服务端 TTL；离开请求失败不影响已提交编辑。
      void canvasCollaborationApi
        .leave(canvasId, sessionId)
        .catch(() => undefined);
    };
  }, [canvasId, scope, store]);
  const people = [
    ...new Map(
      (presence?.participants ?? []).map((item) => [
        item.person.userId,
        item.person,
      ]),
    ).values(),
  ];
  return (
    <>
      <Button
        variant="ghost"
        size="xs"
        onClick={() => setOpen(true)}
        aria-label="查看在线协作者"
        title={
          error
            ? "暂时无法读取在线状态，画布编辑同步独立运行"
            : people.map((person) => person.name).join("、")
        }
      >
        <Users />
        {error
          ? "在线状态待恢复"
          : `在线 ${people.length}${presence?.hasMore ? "+" : ""}`}
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
          <DialogHeader className="shrink-0">
            <DialogTitle>正在此画布协作</DialogTitle>
            <DialogDescription>
              在线状态每 10 秒更新；定位只移动你的视野。
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 space-y-3 overflow-y-auto p-1">
            {people.map((person) => {
              const ids = [
                ...new Set(
                  presence!.participants
                    .filter((item) => item.person.userId === person.userId)
                    .flatMap((item) => item.nodeIds),
                ),
              ];
              return (
                <div
                  key={person.userId}
                  className="flex items-center gap-3 border-b border-border/20 pb-3"
                >
                  <span
                    className="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-medium text-primary"
                    aria-hidden
                  >
                    {person.name.slice(0, 1)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">
                      {person.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {ids.length
                        ? `正在查看 ${ids.length} 个节点`
                        : "正在浏览画布"}
                    </p>
                  </div>
                  {ids.length > 0 && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        const graph = store.getState().snapshot?.graph;
                        store
                          .getState()
                          .focus(
                            ids.filter((id) =>
                              graph?.nodes.some((node) => node.id === id),
                            ),
                          );
                        setOpen(false);
                      }}
                    >
                      <LocateFixed />
                      定位
                    </Button>
                  )}
                </div>
              );
            })}
            {error && (
              <p role="status" className="text-sm text-destructive">
                {error}
              </p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
