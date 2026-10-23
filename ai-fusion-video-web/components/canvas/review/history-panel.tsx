"use client";

import { useEffect, useState } from "react";
import { History, Bookmark, Undo2, LocateFixed } from "lucide-react";
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
import { useAuthStore } from "@/lib/store/auth-store";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { canvasDate, canvasEventNames } from "@/lib/canvas/labels";
import type { CanvasCheckpoint, CanvasHistory } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasCheckpointPreview } from "./checkpoint-preview";
import { CanvasBusinessHistory } from "./business-history";
import { CanvasSegmentedControl } from "../workspace/segmented-control";

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
        if (!active) return;
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
        if (active) setError(getApiErrorMessage(failure));
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
    <section className="space-y-6 p-4" aria-label="画布历史">
      <CanvasSegmentedControl
        label="历史内容"
        value={view}
        onChange={setView}
        className="w-full"
        options={[
          { value: "graph", label: "画布操作" },
          { value: "business", label: "项目内容" },
          { value: "versions", label: "命名版本" },
        ]}
      />
      <div hidden={view !== "business"}>
        <CanvasBusinessHistory />
      </div>
      <div className="space-y-3" hidden={view !== "versions"}>
        <div className="flex items-center gap-2">
          <Bookmark className="size-4" />
          <h3 className="flex-1 text-sm font-semibold">命名版本</h3>
          <Button
            size="xs"
            variant="outline"
            disabled={!editable}
            onClick={() => setNaming(true)}
          >
            保存版本
          </Button>
        </div>
        {checkpoints.map((item) => (
          <Button
            key={item.id}
            variant="ghost"
            className="w-full justify-start"
            onClick={() => setPreview(item.id)}
            title={`${item.authorName} · ${canvasDate(item.createdAt)} · 画布 v${item.revision}`}
          >
            <span className="truncate">{item.title}</span>
          </Button>
        ))}
        {!checkpoints.length && (
          <p className="text-xs text-muted-foreground">
            为重要创作阶段保留命名版本。
          </p>
        )}
        {moreCheckpoints && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => void loadMore(true)}
          >
            更早的版本
          </Button>
        )}
      </div>
      <div className="space-y-3" hidden={view !== "graph"}>
        <div className="flex items-center gap-2">
          <History className="size-4" />
          <h3 className="text-sm font-semibold">画布操作记录</h3>
        </div>
        {history.map((item) => (
          <article
            key={item.commandId}
            className="space-y-2 border-b border-border/20 pb-3"
          >
            <p className="text-sm">
              {canvasEventNames[item.eventType] ?? item.eventType}
            </p>
            <p className="text-xs text-muted-foreground">
              {item.actorUserId === userId ? "我" : item.actorName} ·{" "}
              {canvasDate(item.createdAt)}
            </p>
            <div className="flex gap-2">
              {item.nodeIds.length > 0 && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() =>
                    store
                      .getState()
                      .focus(
                        item.nodeIds.filter((id) =>
                          snapshot.graph.nodes.some((node) => node.id === id),
                        ),
                      )
                  }
                >
                  <LocateFixed />
                  定位
                </Button>
              )}
              {item.canUndo && (
                <Button
                  variant="ghost"
                  size="xs"
                  disabled={!editable}
                  onClick={() =>
                    void store
                      .getState()
                      .undoHistory(item.commandId)
                      .catch((failure) => setError(getApiErrorMessage(failure)))
                  }
                >
                  <Undo2 />
                  撤销画布编辑
                </Button>
              )}
            </div>
          </article>
        ))}
        {more && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => void loadMore(false)}
          >
            更早的操作
          </Button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
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
