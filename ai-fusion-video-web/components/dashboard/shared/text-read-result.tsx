"use client";

import { useId, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { readTextPage, scriptContentSummaries, textReadSummary } from "./text-read-result-model";

export function TextReadResult({ data, toolName }: { data: unknown; toolName: string }) {
  const [open, setOpen] = useState(false);
  const contentId = useId();
  const resultRef = useRef<HTMLDivElement>(null);
  const page = readTextPage(data, toolName);
  if (!page) {
    return <p role="alert" className="text-xs text-muted-foreground">读取结果不完整，暂时无法确认读取范围。</p>;
  }
  const summary = textReadSummary(page);
  const segmentLabel = toolName === "read_tool_result" ? "本段内容" : page.sourceLabel === "剧本正文" ? "本段正文" : "本段原文";
  return (
    <div ref={resultRef} className="flex min-w-0 flex-col gap-2 text-xs">
      <p className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-medium text-foreground">{page.sourceLabel} · {summary.range}</span>
        <span className="text-muted-foreground">{summary.total}</span>
      </p>
      <p className="text-muted-foreground">{summary.detail}</p>
      {page.content.length > 0 && (
        <div className="p-1">
          <Button type="button" variant="ghost" size="xs" aria-expanded={open} aria-controls={contentId}
            onClick={() => setOpen((value) => !value)}>
            {open ? <ChevronDown /> : <ChevronRight />}
            {open ? `收起${segmentLabel}` : `查看${segmentLabel}`}
          </Button>
          {open && (
            <div id={contentId} className="mt-2 flex min-w-0 flex-col gap-2">
              <p className="whitespace-pre-wrap break-words rounded-lg border border-border/20 bg-background/70 p-3 leading-relaxed text-foreground/80">{page.content}</p>
              <Button type="button" variant="ghost" size="xs" className="self-start" onClick={() => {
                setOpen(false);
                requestAnimationFrame(() => resultRef.current?.scrollIntoView({ block: "nearest" }));
              }}>{`收起${segmentLabel}`}</Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function SavedTextResult({ totalChars }: { totalChars: number }) {
  return <p className="text-xs text-muted-foreground">已保存完整结果 · 共 {totalChars.toLocaleString("zh-CN")} 字，可分段读取。</p>;
}

export function ScriptContentSummary({ data }: { data: unknown }) {
  const entries = scriptContentSummaries(data);
  return entries.length > 0 ? (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
      {entries.map((entry) => <span key={entry.label}>{entry.label} {entry.totalChars.toLocaleString("zh-CN")} 字 · 可分段读取</span>)}
    </div>
  ) : null;
}
