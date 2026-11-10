"use client";

import { useEffect, useState } from "react";
import { executionRecordApi, type ExecutionRecordFilter, type ExecutionRecordSummary } from "@/lib/api/execution-records";

export function useExecutionRecords(userId: number | undefined) {
  const [query, setQuery] = useState({ page: 0, filter: "ALL" as ExecutionRecordFilter, days: 0, revision: 0 });
  const [items, setItems] = useState<ExecutionRecordSummary[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [matchedTotal, setMatchedTotal] = useState<number | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!userId) {
      return;
    }
    const controller = new AbortController();
    void executionRecordApi.list(query.page, controller.signal, { filter: query.filter, days: query.days }).then(result => {
      if (controller.signal.aborted) {
        return;
      }
      setItems(previous => {
        const merged = new Map((query.page === 0 ? [] : previous).map(item => [`${item.kind}:${item.id}`, item]));
        result.items.forEach(item => merged.set(`${item.kind}:${item.id}`, item));
        return Array.from(merged.values());
      });
      setTotal(result.total);
      setMatchedTotal(result.matchedTotal);
      setHasMore(result.hasMore);
      setError(null);
    }).catch(() => {
      if (!controller.signal.aborted) {
        setError("异常记录暂时无法加载，请重试");
      }
    }).finally(() => {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    });
    return () => controller.abort();
  }, [query, userId]);

  const changeFilters = (filters: { filter?: ExecutionRecordFilter; days?: number }) => {
    setItems([]);
    setMatchedTotal(null);
    setHasMore(false);
    setLoading(true);
    setError(null);
    setQuery(current => ({ ...current, ...filters, page: 0 }));
  };

  const reload = () => {
    setLoading(true);
    setError(null);
    setQuery(current => ({ ...current, page: 0, revision: current.revision + 1 }));
  };

  const loadMore = () => {
    setLoading(true);
    setQuery(current => error
      ? { ...current, revision: current.revision + 1 }
      : { ...current, page: current.page + 1 });
  };

  return { items, total, matchedTotal, hasMore, loading, error, query, changeFilters, reload, loadMore };
}

export type ExecutionRecordsState = ReturnType<typeof useExecutionRecords>;
