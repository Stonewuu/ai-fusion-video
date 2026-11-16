"use client";

import { useEffect, useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { AlertCircle, Combine, Scissors, Trash2, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { ScriptSourceSegment } from "@/lib/api/script";
import {
  duplicateTitles,
  mergeSegment,
  splitSegment,
  titleKey,
} from "@/lib/scripts/source-import";
import { cn } from "@/lib/utils";

interface Props {
  source: string;
  segments: ScriptSourceSegment[];
  existingTitles: string[];
  selected: number;
  disabled: boolean;
  onSelect: (index: number) => void;
  onChange: (segments: ScriptSourceSegment[]) => void;
  onDelete: () => void;
  deletedTitle?: string;
  onUndoDelete?: () => void;
  onError: (message: string) => void;
}

export function SourcePreview({
  source,
  segments,
  existingTitles,
  selected,
  disabled,
  onSelect,
  onChange,
  onDelete,
  deletedTitle,
  onUndoDelete,
  onError,
}: Props) {
  const previewRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const undoRef = useRef<HTMLButtonElement>(null);
  const deleteRef = useRef<HTMLButtonElement>(null);
  const titleRef = useRef<HTMLInputElement>(null);
  const repeated = useMemo(
    () => duplicateTitles(segments.map((item) => item.title)),
    [segments],
  );
  const existing = useMemo(
    () => new Set(existingTitles.map(titleKey)),
    [existingTitles],
  );
  const virtualizer = useVirtualizer({
    count: segments.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 64,
    overscan: 5,
  });
  const item = segments[selected];
  useEffect(() => {
    if (segments.length > 0) {
      virtualizer.scrollToIndex(selected, { align: "auto" });
      if (restoreFocusRef.current) {
        titleRef.current?.focus();
        restoreFocusRef.current = false;
      } else if (document.activeElement === deleteRef.current) {
        deleteRef.current?.scrollIntoView({ block: "nearest" });
      }
    } else {
      undoRef.current?.focus();
    }
  }, [selected, segments.length, virtualizer]);
  const duplicateCount = segments.filter(
    (segment) =>
      repeated.has(titleKey(segment.title)) ||
      existing.has(titleKey(segment.title)),
  ).length;

  const adjust = (action: "split" | "merge") => {
    try {
      const offset = textRef.current?.selectionStart ?? 0;
      const boundary = item.start + offset;
      if (
        action === "split" &&
        /[\uD800-\uDBFF]/.test(source.charAt(boundary - 1))
      ) {
        throw new Error("请把光标移到完整字符之后");
      }
      onChange(
        action === "split"
          ? splitSegment(segments, selected, offset)
          : mergeSegment(segments, selected),
      );
      onError("");
    } catch (error) {
      onError(error instanceof Error ? error.message : "无法调整分集");
    }
  };

  return (
    <div
      ref={previewRef}
      tabIndex={-1}
      className="flex min-w-0 shrink-0 flex-col gap-3 md:min-h-0 md:flex-1"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className="font-medium">
          共 {segments.length} 集 · {source.length.toLocaleString()} 字
        </span>
        {duplicateCount > 0 && (
          <span className="flex items-center gap-1 text-muted-foreground">
            <AlertCircle className="size-4" />
            {duplicateCount} 个重名标题，可修改分集标题
          </span>
        )}
      </div>
      {onUndoDelete && (
        <div
          className="flex min-w-0 shrink-0 items-center gap-2 text-sm"
          role="status"
        >
          <span className="truncate text-muted-foreground">
            已删除「{deletedTitle}」
          </span>
          <Button
            ref={undoRef}
            variant="ghost"
            disabled={disabled}
            onClick={() => {
              // 先移走焦点，避免撤销按钮移除后弹窗将焦点重置到容器。
              previewRef.current?.focus();
              restoreFocusRef.current = true;
              onUndoDelete();
            }}
          >
            <Undo2 />
            撤销删除
          </Button>
        </div>
      )}
      {segments.length === 0 ? (
        <div className="flex min-h-48 flex-1 items-center justify-center text-sm text-muted-foreground">
          没有待导入的分集
        </div>
      ) : (
        <div className="grid shrink-0 gap-4 md:min-h-0 md:flex-1 md:grid-cols-[16rem_minmax(0,1fr)]">
          <div
            ref={scrollRef}
            className="h-36 overflow-y-auto rounded-lg border border-border/20 bg-background/70 p-1 md:h-full"
            aria-label="分集预览目录"
          >
            <div
              style={{
                height: virtualizer.getTotalSize(),
                position: "relative",
              }}
            >
              {virtualizer.getVirtualItems().map((row) => {
                const segment = segments[row.index];
                const key = titleKey(segment.title);
                const hint = existing.has(key)
                  ? "与现有分集重名"
                  : repeated.has(key)
                    ? "文件内标题重复"
                    : null;
                return (
                  <button
                    key={row.key}
                    type="button"
                    disabled={disabled}
                    onClick={() => onSelect(row.index)}
                    aria-current={selected === row.index ? "true" : undefined}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: row.size,
                      transform: `translateY(${row.start}px)`,
                    }}
                    className={cn(
                      "flex min-w-0 flex-col justify-center gap-1 rounded-xl px-3 text-left text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:opacity-50",
                      selected === row.index
                        ? "border border-border/50 bg-primary/10 font-medium text-foreground"
                        : "border border-transparent hover:bg-muted",
                    )}
                  >
                    <span className="w-full truncate">
                      {row.index + 1}. {segment.title || "未填写标题"}
                    </span>
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                      {(segment.end - segment.start).toLocaleString()} 字{" "}
                      {hint && (
                        <>
                          <AlertCircle className="size-3" />
                          {hint}
                        </>
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-3 p-1 md:min-h-0">
            <div className="flex min-w-0 flex-col gap-2">
              <label
                htmlFor="import-episode-title"
                className="text-sm font-medium"
              >
                第 {selected + 1} 集标题
              </label>
              <Input
                ref={titleRef}
                id="import-episode-title"
                value={item.title}
                disabled={disabled}
                maxLength={256}
                aria-invalid={!item.title.trim()}
                onChange={(event) =>
                  onChange(
                    segments.map((segment, index) =>
                      index === selected
                        ? { ...segment, title: event.target.value }
                        : segment,
                    ),
                  )
                }
              />
              {!item.title.trim() && (
                <p className="text-sm text-destructive">请输入分集标题</p>
              )}
            </div>
            <div className="flex min-w-0 flex-col gap-2 md:min-h-0 md:flex-1">
              <label
                htmlFor="import-episode-source"
                className="text-sm font-medium"
              >
                本集原文
              </label>
              <Textarea
                key={`${selected}:${item.end}`}
                ref={textRef}
                id="import-episode-source"
                readOnly
                disabled={disabled}
                value={source.slice(item.start, item.end)}
                className="field-sizing-fixed h-48 min-h-40 resize-none text-sm md:h-auto md:flex-1"
              />
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <Button
                variant="outline"
                disabled={disabled}
                onClick={() => adjust("split")}
                title="在上方原文中放置光标，再点击分集"
              >
                <Scissors />
                在光标处分集
              </Button>
              <Button
                variant="ghost"
                disabled={disabled || selected === segments.length - 1}
                onClick={() => adjust("merge")}
              >
                <Combine />
                合并下一集
              </Button>
              <Button
                ref={deleteRef}
                variant="destructive-ghost"
                disabled={disabled}
                onClick={onDelete}
                title="从本次导入中删除"
                className="sm:ml-auto"
              >
                <Trash2 />
                删除本集
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
