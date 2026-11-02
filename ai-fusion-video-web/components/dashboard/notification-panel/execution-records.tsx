"use client";

import { useEffect, useState } from "react";
import { CircleHelp, Loader2, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { executionRecordApi, executionRecordStatus, type ExecutionRecordRef, type ExecutionRecordSummary } from "@/lib/api/execution-records";
import { useAuthStore } from "@/lib/store/auth-store";
import { TaskListItem } from "./task-list-items";
import { formatDatetime } from "./utils";

export function ExecutionRecordSection({ selected, onSelect }: {
  selected: ExecutionRecordRef | null;
  onSelect: (record: ExecutionRecordRef) => void;
}) {
  const userId = useAuthStore((state) => state.user?.id);
  return userId ? <ExecutionRecordList key={userId} selected={selected} onSelect={onSelect} /> : null;
}

function ExecutionRecordList({ selected, onSelect }: {
  selected: ExecutionRecordRef | null;
  onSelect: (record: ExecutionRecordRef) => void;
}) {
  const [items, setItems] = useState<ExecutionRecordSummary[]>([]);
  const [page, setPage] = useState(0);
  const [revision, setRevision] = useState(0);
  const [more, setMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void executionRecordApi.list(page, controller.signal).then((result) => {
      if (controller.signal.aborted) {
        return;
      }
      setItems((previous) => {
        const merged = new Map((page === 0 ? [] : previous).map((item) => [`${item.kind}:${item.id}`, item]));
        result.items.forEach((item) => merged.set(`${item.kind}:${item.id}`, item));
        return Array.from(merged.values());
      });
      setMore(result.hasMore);
      setError(null);
    }).catch((failure) => {
      if (!controller.signal.aborted) {
        setError(getApiErrorMessage(failure));
      }
    }).finally(() => {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    });
    return () => controller.abort();
  }, [page, revision]);

  const reload = () => {
    setLoading(true);
    setPage(0);
    setRevision((value) => value + 1);
  };
  return (
    <section className="px-3 py-2" aria-label="异常任务记录">
      <div className="flex items-center justify-between gap-2 px-1 pb-2">
        <h4 className="text-xs font-medium text-muted-foreground">异常记录</h4>
        <Button variant="ghost" size="icon-xs" aria-label="刷新异常记录" disabled={loading} onClick={reload}>
          <RefreshCw className={loading ? "animate-spin" : undefined} />
        </Button>
      </div>
      {error && <p role="alert" className="px-1 py-2 text-xs text-destructive">{error}</p>}
      {!error && !loading && items.length === 0 && <p className="px-1 py-2 text-xs text-muted-foreground">暂无异常记录</p>}
      {items.map((item) => (
        <TaskListItem
          key={`${item.kind}:${item.id}`}
          label={item.title}
          subtitle={`${executionRecordStatus(item)} · ${formatDatetime(item.updatedAt)}`}
          icon={item.remoteUnknown ? <CircleHelp className="size-4 shrink-0 text-muted-foreground" /> : <XCircle className="size-4 shrink-0 text-destructive" />}
          selected={selected?.kind === item.kind && selected.id === item.id}
          onClick={() => onSelect(item)}
        />
      ))}
      {loading && items.length === 0 && <div role="status" aria-label="加载异常记录" className="flex justify-center p-3"><Loader2 className="size-4 animate-spin text-muted-foreground" /></div>}
      {more && <Button variant="ghost" size="sm" className="w-full" disabled={loading} onClick={() => {
        setLoading(true);
        setPage((value) => value + 1);
      }}>加载更多</Button>}
    </section>
  );
}
