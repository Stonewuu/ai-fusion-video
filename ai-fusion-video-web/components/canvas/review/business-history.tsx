"use client";

import { useCallback, useEffect, useState } from "react";
import { Bot, History } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
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
        if (controller.signal.aborted) {
          return;
        }
        setEntries(result.items);
        setNext(result.nextCursor);
        setError(null);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) {
          setError(getApiErrorMessage(failure));
        }
      });
    return () => {
      controller.abort();
    };
  }, [canvasId, contentRevision, refresh]);

  const more = useCallback(async () => {
    if (!next || loading) {
      return;
    }
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
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-col gap-0.5">
        <h3 className="text-[13px] font-medium">项目内容变更</h3>
        <p className="text-xs text-muted-foreground">
          查看你与助手对项目内容的修改记录。
        </p>
      </div>
      {entries.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {entries.map((entry) => (
            <article
              key={entry.commandId}
              className="flex flex-col gap-1.5 rounded-[12px] border border-foreground/[0.07] bg-background/70 p-2.5 dark:border-white/[0.08] dark:bg-white/[0.03]"
            >
              <div className="flex items-center justify-between gap-2">
                <p className="flex min-w-0 items-center gap-1.5 truncate text-[13px] font-medium">
                  {entry.origin === "agent" && (
                    <Bot className="size-3.5 shrink-0 text-violet-500" />
                  )}
                  <span className="truncate">
                    {businessOperationNames[entry.operation] ?? entry.operation}
                  </span>
                </p>
                <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                  {canvasDate(entry.committedAt)}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground">
                {entry.actorName} ·{" "}
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
                项
              </p>
              <div>
                <CanvasButton
                  size="sm"
                  variant="secondary"
                  onClick={() => setSelected(entry.commandId)}
                >
                  查看变更
                </CanvasButton>
              </div>
            </article>
          ))}
        </div>
      )}
      {!entries.length && !error && (
        <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-6 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
          <span className="grid size-8 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
            <History className="size-4" aria-hidden />
          </span>
          <p className="text-xs text-muted-foreground">暂无可查看的内容变更</p>
        </div>
      )}
      {next && (
        <CanvasButton
          variant="secondary"
          size="sm"
          disabled={loading}
          onClick={() => void more()}
        >
          更早的变更
        </CanvasButton>
      )}
      {error && (
        <div className="flex flex-col items-start gap-2 rounded-[10px] border border-destructive/20 bg-destructive/5 p-2.5">
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
          <CanvasButton
            variant="secondary"
            size="sm"
            onClick={() => setRefresh((value) => value + 1)}
          >
            重新读取
          </CanvasButton>
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
