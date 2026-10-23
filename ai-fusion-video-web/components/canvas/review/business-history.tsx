"use client";

import { useCallback, useEffect, useState } from "react";
import { Bot, History } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  canvasHistoryApi,
  type BusinessHistorySummary,
} from "@/lib/api/canvas-history";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { canvasDate, canvasTypeNames } from "@/lib/canvas/labels";
import { businessOperationNames } from "@/lib/canvas/history-presentation";
import { useCanvas } from "../canvas-provider";
import { BusinessHistoryDetail } from "./business-history-detail";

export function CanvasBusinessHistory() {
  const canvasId = useCanvas((state) => state.snapshot!.document.id);
  const contentRevision = useCanvas((state) => state.contentRevision);
  const [entries, setEntries] = useState<BusinessHistorySummary[]>([]);
  const [next, setNext] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    void canvasHistoryApi
      .list(canvasId, undefined, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setEntries(result.items);
        setNext(result.nextCursor);
        setError(null);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      });
    return () => controller.abort();
  }, [canvasId, contentRevision, refresh]);
  const more = useCallback(async () => {
    if (!next || loading) return;
    setLoading(true);
    try {
      const result = await canvasHistoryApi.list(canvasId, next);
      setEntries((current) => [
        ...new Map(
          [...current, ...result.items].map((entry) => [
            entry.commandId,
            entry,
          ]),
        ).values(),
      ]);
      setNext(result.nextCursor);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setLoading(false);
    }
  }, [canvasId, next, loading]);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <History className="size-4 text-primary" />
        <h3 className="text-sm font-semibold">项目内容变更</h3>
      </div>
      <p className="text-xs text-muted-foreground">
        查看项目内容的修改记录，包括你和助手的编辑。
      </p>
      {entries.map((entry) => (
        <article
          key={entry.commandId}
          className="space-y-2 border-b border-border/20 pb-3"
        >
          <p className="flex items-center gap-2 text-sm font-medium">
            {entry.origin === "agent" && (
              <Bot className="size-4 text-violet-500" />
            )}
            {businessOperationNames[entry.operation] ?? entry.operation}
          </p>
          <p className="text-xs text-muted-foreground">
            {entry.actorName} · {canvasDate(entry.committedAt)}
          </p>
          <p className="text-xs text-muted-foreground">
            {[
              ...new Set(
                entry.resources.map(
                  (item) => canvasTypeNames[item.resourceType],
                ),
              ),
            ].join("、")}{" "}
            ·{" "}
            {
              new Set(
                entry.resources.map(
                  (item) => `${item.resourceType}:${item.resourceId}`,
                ),
              ).size
            }{" "}
            项内容
          </p>
          <Button
            size="xs"
            variant="outline"
            onClick={() => setSelected(entry.commandId)}
          >
            查看变更
          </Button>
        </article>
      ))}
      {!entries.length && !error && (
        <p className="text-sm text-muted-foreground">暂无可查看的内容变更。</p>
      )}
      {next && (
        <Button
          variant="outline"
          size="sm"
          disabled={loading}
          onClick={() => void more()}
        >
          更早的变更
        </Button>
      )}
      {error && (
        <div className="space-y-2">
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setRefresh((value) => value + 1)}
          >
            重新读取
          </Button>
        </div>
      )}
      {selected && (
        <BusinessHistoryDetail
          commandId={selected}
          onClose={() => setSelected(null)}
          onRestored={() => {
            setSelected(null);
            setRefresh((value) => value + 1);
          }}
        />
      )}
    </div>
  );
}
