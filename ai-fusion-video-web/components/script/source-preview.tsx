"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  BookOpenText,
  ChevronLeft,
  ChevronRight,
  Combine,
  Scissors,
  Trash2,
  Undo2,
} from "lucide-react";
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
import { SourcePreviewDirectory } from "./source-preview-directory";

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
  const textRef = useRef<HTMLTextAreaElement>(null);
  const undoRef = useRef<HTMLButtonElement>(null);
  const titleRef = useRef<HTMLInputElement>(null);
  const [adjustError, setAdjustError] = useState("");
  const repeated = useMemo(
    () => duplicateTitles(segments.map((item) => item.title)),
    [segments],
  );
  const existing = useMemo(
    () => new Set(existingTitles.map(titleKey)),
    [existingTitles],
  );
  const item = segments[selected];
  useEffect(() => {
    if (segments.length === 0) {
      undoRef.current?.focus();
    } else if (restoreFocusRef.current) {
      titleRef.current?.focus();
      restoreFocusRef.current = false;
    }
  }, [selected, segments.length]);
  const duplicateCount = segments.filter(
    (segment) =>
      repeated.has(titleKey(segment.title)) ||
      existing.has(titleKey(segment.title)),
  ).length;
  const selectedDuplicate =
    item &&
    (existing.has(titleKey(item.title))
      ? "与现有分集重名，可修改标题后导入。"
      : repeated.has(titleKey(item.title))
        ? "本次导入中有相同标题，可修改标题以便区分。"
        : null);

  const select = (index: number) => {
    setAdjustError("");
    onSelect(index);
  };
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
      setAdjustError("");
      onError("");
    } catch (error) {
      setAdjustError(error instanceof Error ? error.message : "无法调整分集");
      textRef.current?.focus({ preventScroll: true });
    }
  };

  return (
    <div
      ref={previewRef}
      tabIndex={-1}
      className="flex min-w-0 shrink-0 flex-col outline-none md:min-h-0 md:flex-1"
    >
      <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-border/20 px-4 py-3 sm:px-6">
        <span className="text-sm">
          <strong className="text-lg font-semibold tabular-nums">
            {segments.length.toLocaleString()}
          </strong>{" "}
          <span className="text-muted-foreground">集待导入</span>
        </span>
        <span className="text-xs tabular-nums text-muted-foreground">
          共 {source.length.toLocaleString()} 字
        </span>
        {onUndoDelete && (
          <div
            role="status"
            className="flex min-w-0 items-center gap-2 sm:ml-auto"
          >
            <span className="max-w-48 truncate text-xs text-muted-foreground">
              已删除「{deletedTitle}」
            </span>
            <Button
              ref={undoRef}
              variant="ghost"
              disabled={disabled}
              onClick={() => {
                previewRef.current?.focus();
                restoreFocusRef.current = true;
                setAdjustError("");
                onUndoDelete();
              }}
            >
              <Undo2 />
              撤销删除
            </Button>
          </div>
        )}
      </div>
      {segments.length === 0 ? (
        <div className="flex min-h-64 flex-1 flex-col items-center justify-center gap-3 text-center">
          <BookOpenText className="size-8 text-muted-foreground/50" />
          <p className="text-sm font-medium">没有待导入的分集</p>
          <p className="text-xs text-muted-foreground">
            可撤销删除，或返回上一步重新选择原文。
          </p>
        </div>
      ) : (
        <div className="grid gap-4 p-4 md:min-h-0 md:flex-1 md:grid-cols-[17rem_minmax(0,1fr)] md:gap-0 md:p-0">
          <SourcePreviewDirectory
            segments={segments}
            selected={selected}
            disabled={disabled}
            repeated={repeated}
            existing={existing}
            duplicateCount={duplicateCount}
            onSelect={select}
          />
          <section
            aria-label="分集内容"
            className="flex min-w-0 flex-col gap-4 p-1 md:min-h-0 md:overflow-y-auto md:p-6"
          >
            <div className="flex shrink-0 items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-sm font-semibold">
                <BookOpenText className="size-4 text-primary" />第{" "}
                {selected + 1} 集
                <span className="text-xs font-normal text-muted-foreground">
                  / {segments.length} 集
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="上一集"
                  disabled={disabled || selected === 0}
                  onClick={() => select(selected - 1)}
                >
                  <ChevronLeft />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="下一集"
                  disabled={disabled || selected === segments.length - 1}
                  onClick={() => select(selected + 1)}
                >
                  <ChevronRight />
                </Button>
              </div>
            </div>
            <div className="flex min-w-0 shrink-0 flex-col gap-2">
              <label
                htmlFor="import-episode-title"
                className="text-xs font-medium text-muted-foreground"
              >
                分集标题
              </label>
              <Input
                ref={titleRef}
                id="import-episode-title"
                value={item.title}
                disabled={disabled}
                maxLength={256}
                aria-invalid={!item.title.trim()}
                aria-describedby={
                  !item.title.trim() || selectedDuplicate
                    ? "import-title-hint"
                    : undefined
                }
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
              {!item.title.trim() ? (
                <p id="import-title-hint" className="text-xs text-destructive">
                  请输入分集标题
                </p>
              ) : (
                selectedDuplicate && (
                  <p
                    id="import-title-hint"
                    className="flex items-start gap-1.5 text-xs text-muted-foreground"
                  >
                    <AlertCircle className="size-3.5 shrink-0" />
                    {selectedDuplicate}
                  </p>
                )
              )}
            </div>
            <div className="flex min-w-0 flex-col gap-2 md:min-h-48 md:flex-1">
              <div className="flex items-center justify-between gap-2">
                <label
                  htmlFor="import-episode-source"
                  className="text-xs font-medium text-muted-foreground"
                >
                  本集原文
                </label>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {(item.end - item.start).toLocaleString()} 字
                </span>
              </div>
              <Textarea
                key={`${selected}:${item.start}:${item.end}`}
                ref={textRef}
                id="import-episode-source"
                readOnly
                disabled={disabled}
                value={source.slice(item.start, item.end)}
                aria-describedby="import-boundary-hint"
                className="field-sizing-fixed h-64 min-h-40 resize-none rounded-lg bg-background p-4 text-sm leading-7 md:h-auto md:flex-1"
              />
            </div>
            <div className="flex shrink-0 flex-col gap-2">
              <p
                id="import-boundary-hint"
                className="text-xs text-muted-foreground"
              >
                在原文中放置光标，可从该位置拆为两集。
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  disabled={disabled}
                  onClick={() => adjust("split")}
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
                  variant="destructive-ghost"
                  disabled={disabled}
                  onClick={() => {
                    setAdjustError("");
                    onDelete();
                  }}
                  title="从本次导入中删除"
                  className="sm:ml-auto"
                >
                  <Trash2 />
                  删除本集
                </Button>
              </div>
              {adjustError && (
                <p role="alert" className="text-xs text-destructive">
                  {adjustError}
                </p>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
