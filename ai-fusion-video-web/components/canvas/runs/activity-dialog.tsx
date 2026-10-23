"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { canvasActivityApi, type CanvasActivity } from "@/lib/api/canvas-activity";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { CanvasActivitySummary } from "./activity-summary";

const localDay = (date: Date) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
function period(days: number) {
  const end = new Date();
  const start = new Date(); start.setDate(start.getDate() - days + 1);
  return { start: localDay(start), end: localDay(end) };
}
export function CanvasActivityDialog({ canvasId, onClose }: { canvasId: string; onClose: () => void }) {
  const [preset, setPreset] = useState("7");
  const [dates, setDates] = useState(() => period(7));
  const [query, setQuery] = useState(() => ({ ...period(7), reload: 0 }));
  const [result, setResult] = useState<CanvasActivity | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    const from = new Date(`${query.start}T00:00:00`);
    const until = new Date(`${query.end}T00:00:00`); until.setDate(until.getDate() + 1);
    void canvasActivityApi.get(canvasId, from.toISOString(), until.toISOString(), controller.signal)
      .then((data) => { if (!controller.signal.aborted) setResult(data); })
      .catch((failure) => { if (!controller.signal.aborted) setError(getApiErrorMessage(failure)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [canvasId, query]);
  const submit = (next = dates) => {
    setLoading(true); setError(null); setResult(null);
    setQuery({ ...next, reload: query.reload + 1 });
  };
  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>累计统计</DialogTitle>
          <DialogDescription>汇总这张画布在所选日期内发起的运行与交付。</DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-5 overflow-y-auto px-1 py-1">
          <CanvasSegmentedControl label="统计日期" value={preset} options={[
            { value: "7", label: "近 7 天" }, { value: "30", label: "近 30 天" }, { value: "90", label: "近 90 天" },
          ]} onChange={(value) => { const next = period(Number(value)); setPreset(value); setDates(next); submit(next); }} className="w-full" />
          <form className="grid items-end gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]" onSubmit={(event) => { event.preventDefault(); submit(); }}>
            <div className="min-w-0 flex-1 space-y-2"><Label htmlFor="activity-from">开始日期</Label><Input required id="activity-from" type="date" value={dates.start} onChange={(event) => { setDates({ ...dates, start: event.target.value }); setPreset(""); }} /></div>
            <div className="min-w-0 flex-1 space-y-2"><Label htmlFor="activity-until">结束日期</Label><Input required id="activity-until" type="date" min={dates.start} value={dates.end} onChange={(event) => { setDates({ ...dates, end: event.target.value }); setPreset(""); }} /></div>
            <Button type="submit" variant="outline" disabled={loading || !dates.start || !dates.end || dates.start > dates.end}>查询</Button>
          </form>
          {(dates.start !== query.start || dates.end !== query.end) && <p role="status" className="text-xs text-muted-foreground">日期已修改，点击查询更新统计。</p>}
          {loading && <p role="status" className="py-6 text-center text-sm text-muted-foreground">正在汇总运行与交付…</p>}
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
          {result && <><CanvasActivitySummary value={result} /><p className="text-xs text-muted-foreground">更新于 {new Date(result.observedAt).toLocaleString()}</p></>}
        </div>
        <DialogFooter className="shrink-0"><Button variant="outline" onClick={onClose}>关闭</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
