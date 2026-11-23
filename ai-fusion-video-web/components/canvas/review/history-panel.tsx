"use client";

import { useEffect, useState } from "react";
import { Bookmark, History, LocateFixed, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasSegmented } from "../ui/canvas-segmented";
import { useAuthStore } from "@/lib/store/auth-store";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { canvasDate, canvasEventNames } from "@/lib/canvas/labels";
import type { CanvasCheckpoint, CanvasHistory } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasCheckpointPreview } from "./checkpoint-preview";
import { CanvasBusinessHistory } from "./business-history";

const historyViewOptions = [
  { value: "graph", label: "画布操作" },
  { value: "business", label: "项目内容" },
  { value: "versions", label: "命名版本" },
] as const;

export function CanvasHistoryPanel() {
  const userId = useAuthStore((state) => state.user?.id);
  const snapshot = useCanvas((state) => state.snapshot!);
  const store = useCanvasStoreApi();
  const pending = useCanvas((state) => !!state.pending);
  const requests = useCommandRequestKey();
  const [history, setHistory] = useState<CanvasHistory[]>([]);
  const [checkpoints, setCheckpoints] = useState<CanvasCheckpoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [more, setMore] = useState(false);
  const [moreCheckpoints, setMoreCheckpoints] = useState(false);
  const [saving, setSaving] = useState(false);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [checkpointId, setCheckpointId] = useState(() => crypto.randomUUID());
  const [preview, setPreview] = useState<string | null>(null);
  const [view, setView] = useState<"graph" | "business" | "versions">("graph");
  const editable =
    snapshot.permissions.canEdit && !snapshot.document.archived && !pending;

  useEffect(() => {
    let active = true;
    void Promise.all([
      canvasApi.history(snapshot.document.id),
      canvasApi.checkpoints(snapshot.document.id),
    ])
      .then(([events, versions]) => {
        if (!active) {
          return;
        }
        setHistory((current) =>
          [
            ...new Map(
              [...current, ...events].map((item) => [item.commandId, item]),
            ).values(),
          ].sort((a, b) => b.revision - a.revision),
        );
        setCheckpoints((current) =>
          [
            ...new Map(
              [...current, ...versions].map((item) => [item.id, item]),
            ).values(),
          ].sort((a, b) => b.revision - a.revision),
        );
        setMore(events.length === 40);
        setMoreCheckpoints(versions.length === 40);
      })
      .catch((failure) => {
        if (active) {
          setError(getApiErrorMessage(failure));
        }
      });
    return () => {
      active = false;
    };
  }, [snapshot.document.id, snapshot.document.revision]);

  const loadMore = async (versions: boolean) => {
    try {
      if (versions) {
        const result = await canvasApi.checkpoints(
          snapshot.document.id,
          checkpoints.at(-1)?.revision,
        );
        setCheckpoints((current) => [...current, ...result]);
        setMoreCheckpoints(result.length === 40);
      } else {
        const result = await canvasApi.history(
          snapshot.document.id,
          history.at(-1)?.revision,
        );
        setHistory((current) => [...current, ...result]);
        setMore(result.length === 40);
      }
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };

  const saveCheckpoint = async () => {
    const body = {
      id: checkpointId,
      title: name.trim(),
      expectedRevision: store.getState().snapshot!.document.revision,
    };
    setSaving(true);
    setError(null);
    try {
      await canvasApi.saveCheckpoint(snapshot.document.id, {
        ...body,
        requestId: requests.forPayload(body),
      });
      await store.getState().load();
      setNaming(false);
      setName("");
      setCheckpointId(crypto.randomUUID());
      requests.clear();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="flex flex-col gap-3 p-3.5" aria-label="画布历史">
      <CanvasSegmented
        label="历史内容"
        value={view}
        onChange={setView}
        className="w-full"
        options={historyViewOptions}
      />
      <div hidden={view !== "business"}>
        <CanvasBusinessHistory />
      </div>
      <div className="flex flex-col gap-2.5" hidden={view !== "versions"}>
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-[13px] font-medium">命名版本</h3>
          <CanvasButton
            size="sm"
            variant="secondary"
            disabled={!editable}
            onClick={() => setNaming(true)}
          >
            <Bookmark />
            保存版本
          </CanvasButton>
        </div>
        {checkpoints.length > 0 ? (
          <div className="flex flex-col gap-1">
            {checkpoints.map((item) => (
              <CanvasButton
                key={item.id}
                variant="ghost"
                className="w-full justify-between"
                onClick={() => setPreview(item.id)}
                title={`${item.authorName} · ${canvasDate(item.createdAt)} · 画布 v${item.revision}`}
              >
                <span className="min-w-0 truncate text-left">{item.title}</span>
                <span className="shrink-0 rounded-[5px] bg-foreground/[0.06] px-1.5 py-0.5 text-[11px] tabular-nums text-muted-foreground dark:bg-white/[0.08]">
                  v{item.revision}
                </span>
              </CanvasButton>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-6 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
            <span className="grid size-8 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
              <Bookmark className="size-4" aria-hidden />
            </span>
            <p className="text-xs text-muted-foreground">
              为重要创作阶段保留命名版本，随时回看对比。
            </p>
          </div>
        )}
        {moreCheckpoints && (
          <CanvasButton
            size="sm"
            variant="secondary"
            onClick={() => void loadMore(true)}
          >
            更早的版本
          </CanvasButton>
        )}
      </div>
      <div className="flex flex-col gap-2" hidden={view !== "graph"}>
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-[13px] font-medium">画布操作记录</h3>
          {history.length > 0 && (
            <span className="rounded-[6px] bg-foreground/[0.05] px-1.5 py-0.5 text-[11px] tabular-nums text-muted-foreground dark:bg-white/[0.07]">
              v{snapshot.document.revision}
            </span>
          )}
        </div>
        {history.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            {history.map((item) => (
              <article
                key={item.commandId}
                className="flex flex-col gap-1.5 rounded-[12px] border border-foreground/[0.07] bg-background/70 p-2.5 dark:border-white/[0.08] dark:bg-white/[0.03]"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <p className="min-w-0 truncate text-[13px] font-medium">
                    {canvasEventNames[item.eventType] ?? item.eventType}
                  </p>
                  <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                    {canvasDate(item.createdAt)}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] text-muted-foreground">
                    {item.actorUserId === userId ? "我" : item.actorName}
                  </span>
                  <div className="flex items-center gap-1">
                    {item.nodeIds.length > 0 && (
                      <CanvasButton
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          store
                            .getState()
                            .focus(
                              item.nodeIds.filter((id) =>
                                snapshot.graph.nodes.some(
                                  (node) => node.id === id,
                                ),
                              ),
                            )
                        }
                      >
                        <LocateFixed />
                        定位
                      </CanvasButton>
                    )}
                    {item.canUndo && (
                      <CanvasButton
                        variant="ghost"
                        size="sm"
                        disabled={!editable}
                        onClick={() =>
                          void store
                            .getState()
                            .undoHistory(item.commandId)
                            .catch((failure) =>
                              setError(getApiErrorMessage(failure)),
                            )
                        }
                      >
                        <Undo2 />
                        撤销
                      </CanvasButton>
                    )}
                  </div>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-6 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
            <span className="grid size-8 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
              <History className="size-4" aria-hidden />
            </span>
            <p className="text-xs text-muted-foreground">暂无操作记录</p>
          </div>
        )}
        {more && (
          <CanvasButton
            size="sm"
            variant="secondary"
            onClick={() => void loadMore(false)}
          >
            更早的操作
          </CanvasButton>
        )}
      </div>
      {error && (
        <p
          role="alert"
          className="rounded-[10px] border border-destructive/20 bg-destructive/5 px-2.5 py-2 text-xs text-destructive"
        >
          {error}
        </p>
      )}
      <Dialog open={naming} onOpenChange={setNaming}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>保存命名版本</DialogTitle>
            <DialogDescription>
              保存画布结构和当前项目内容，以便之后回看。
            </DialogDescription>
          </DialogHeader>
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={200}
            placeholder="例如：第一集初稿"
            aria-label="版本名称"
            disabled={saving}
          />
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={saving}
              onClick={() => setNaming(false)}
            >
              取消
            </Button>
            <Button
              disabled={saving || !name.trim()}
              onClick={() => void saveCheckpoint()}
            >
              保存版本
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {preview && (
        <CanvasCheckpointPreview
          id={preview}
          onClose={() => setPreview(null)}
        />
      )}
    </section>
  );
}
