"use client";

import { Ban, CircleHelp, Loader2, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { executionRecordStatus, type ExecutionRecordFilter, type ExecutionRecordRef } from "@/lib/api/execution-records";
import { TaskListItem } from "./task-list-items";
import { formatDatetime } from "./utils";
import type { ExecutionRecordsState } from "./use-execution-records";

const states = [
  { value: "ALL", label: "全部状态" },
  { value: "REVIEW", label: "待确认" },
  { value: "FAILED", label: "执行失败" },
  { value: "CANCELLED", label: "已取消" },
];
const periods = [
  { value: "0", label: "全部时间" },
  { value: "7", label: "近 7 天" },
  { value: "30", label: "近 30 天" },
];

export function ExecutionRecordList({ records, selected, onSelect, showFilters }: {
  records: ExecutionRecordsState;
  selected: ExecutionRecordRef | null;
  onSelect: (record: ExecutionRecordRef) => void;
  showFilters: boolean;
}) {
  const { items, query, changeFilters, matchedTotal, loading, error, hasMore, reload, loadMore } = records;
  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="异常任务记录">
      <div className="shrink-0 px-3 pb-2">
        {showFilters && <div className="grid min-w-0 grid-cols-2 gap-2 p-1">
          <Select items={states} value={query.filter} onValueChange={value => changeFilters({ filter: value as ExecutionRecordFilter })}>
            <SelectTrigger className="w-full min-w-0" aria-label="异常状态"><SelectValue placeholder="全部状态" /></SelectTrigger>
            <SelectContent><SelectGroup>{states.map(item => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectGroup></SelectContent>
          </Select>
          <Select items={periods} value={String(query.days)} onValueChange={value => changeFilters({ days: Number(value) })}>
            <SelectTrigger className="w-full min-w-0" aria-label="异常记录时间"><SelectValue placeholder="全部时间" /></SelectTrigger>
            <SelectContent><SelectGroup>{periods.map(item => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectGroup></SelectContent>
          </Select>
        </div>}
        <div className="flex items-center justify-between gap-2 px-1">
          <span className="text-xs text-muted-foreground" aria-live="polite">{matchedTotal === null ? "" : `共 ${matchedTotal} 条`}</span>
          <Button variant="ghost" size="icon" aria-label="刷新异常记录" disabled={loading} onClick={reload}>
            <RefreshCw className={loading ? "animate-spin motion-reduce:animate-none" : undefined} />
          </Button>
        </div>
      </div>
      <OverlayScrollArea className="min-h-0 flex-1" aria-label="异常记录列表">
        <div className="flex flex-col gap-1 px-3 pb-3 pt-1">
          {error && <div role="alert" className="flex flex-col items-start gap-2 px-1 py-3">
            <p className="text-xs text-destructive">{error}</p>
            <Button variant="outline" onClick={items.length ? loadMore : reload}>重试</Button>
          </div>}
          {!error && !loading && items.length === 0 && <p className="px-1 py-8 text-center text-xs text-muted-foreground">
            {query.filter === "ALL" && query.days === 0 ? "暂无异常记录" : "没有符合条件的记录"}
          </p>}
          {items.map(item => (
            <TaskListItem key={`${item.kind}:${item.id}`} label={item.title}
              description={item.message}
              subtitle={`${executionRecordStatus(item)} · ${formatDatetime(item.updatedAt)}`}
              icon={item.status === "CANCELLED" ? <Ban className="size-4 shrink-0 text-muted-foreground" />
                : item.remoteUnknown || item.status === "NEEDS_REVIEW" ? <CircleHelp className="size-4 shrink-0 text-muted-foreground" />
                : <XCircle className="size-4 shrink-0 text-destructive" />}
              selected={selected?.kind === item.kind && selected.id === item.id} onClick={() => onSelect(item)} />
          ))}
          {loading && <div role="status" aria-label="加载异常记录" className="flex justify-center p-3">
            <Loader2 className="size-4 animate-spin text-muted-foreground motion-reduce:animate-none" />
          </div>}
          {hasMore && !error && <Button variant="ghost" className="w-full" disabled={loading} onClick={loadMore}>加载更多</Button>}
        </div>
      </OverlayScrollArea>
    </section>
  );
}
