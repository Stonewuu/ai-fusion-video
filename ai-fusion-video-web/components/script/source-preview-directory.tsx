"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { AlertCircle, ChevronDown, ListTree, Search, X } from "lucide-react";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { ScriptSourceSegment } from "@/lib/api/script";
import { titleKey } from "@/lib/scripts/source-import";
import { cn } from "@/lib/utils";

interface Props {
  segments: ScriptSourceSegment[];
  selected: number;
  disabled: boolean;
  repeated: Set<string>;
  existing: Set<string>;
  duplicateCount: number;
  onSelect: (index: number) => void;
}

export function SourcePreviewDirectory({
  segments,
  selected,
  disabled,
  repeated,
  existing,
  duplicateCount,
  onSelect,
}: Props) {
  const [query, setQuery] = useState("");
  const [onlyDuplicates, setOnlyDuplicates] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const mobileTriggerRef = useRef<HTMLButtonElement>(null);
  const filterDuplicates = onlyDuplicates && duplicateCount > 0;
  const lastFilter = useRef({ query, filterDuplicates });
  const entries = useMemo(() => {
    const search = titleKey(query);
    return segments.flatMap((segment, index) => {
      const key = titleKey(segment.title);
      const duplicate = repeated.has(key) || existing.has(key);
      const matches =
        !search || key.includes(search) || String(index + 1) === search;
      return matches && (!filterDuplicates || duplicate)
        ? [{ segment, index, key }]
        : [];
    });
  }, [segments, query, filterDuplicates, repeated, existing]);
  const virtualizer = useVirtualizer({
    count: entries.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 64,
    overscan: 5,
    paddingStart: 4,
    paddingEnd: 4,
  });
  useEffect(() => {
    const filterChanged =
      lastFilter.current.query !== query ||
      lastFilter.current.filterDuplicates !== filterDuplicates;
    lastFilter.current = { query, filterDuplicates };
    if (filterChanged) {
      virtualizer.scrollToOffset(0);
      return;
    }
    const position = entries.findIndex((entry) => entry.index === selected);
    if (position >= 0) {
      virtualizer.scrollToIndex(position, { align: "auto" });
    }
  }, [selected, entries, mobileOpen, virtualizer, query, filterDuplicates]);

  return (
    <aside className="flex min-h-0 min-w-0 flex-col rounded-xl border border-border/30 bg-muted/30 md:rounded-none md:border-0 md:border-r md:border-border/20">
      <div className="p-2 md:hidden">
        <Button
          ref={mobileTriggerRef}
          variant="ghost"
          className="w-full justify-between"
          aria-expanded={mobileOpen}
          aria-controls="source-preview-directory"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          <span className="flex min-w-0 items-center gap-2">
            <ListTree />
            <span className="truncate">分集目录 · 第 {selected + 1} 集</span>
          </span>
          <ChevronDown
            className={cn(
              "transition-transform duration-200 motion-reduce:transition-none",
              mobileOpen && "rotate-180",
            )}
          />
        </Button>
      </div>
      <div
        id="source-preview-directory"
        className={cn(
          "grid min-h-0 grid-rows-[0fr] transition-[grid-template-rows,opacity] duration-200 ease-out motion-reduce:transition-none md:flex md:flex-1 md:opacity-100",
          mobileOpen
            ? "visible grid-rows-[1fr] opacity-100"
            : "invisible opacity-0 md:visible",
        )}
      >
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="flex shrink-0 flex-col gap-3 p-3 md:p-4">
            <div className="hidden items-center justify-between md:flex">
              <h3 className="flex items-center gap-2 text-sm font-semibold">
                <ListTree className="size-4 text-muted-foreground" />
                分集目录
              </h3>
              <span className="text-xs tabular-nums text-muted-foreground">
                {segments.length.toLocaleString()} 集
              </span>
            </div>
            <div className="relative p-1">
              <Search
                aria-hidden
                className="pointer-events-none absolute top-3.5 left-3.5 size-4 text-muted-foreground"
              />
              <Input
                aria-label="搜索分集"
                placeholder="搜索标题或集数"
                value={query}
                disabled={disabled}
                onChange={(event) => setQuery(event.target.value)}
                className="pl-9"
              />
            </div>
            {duplicateCount > 0 && (
              <Button
                variant={filterDuplicates ? "secondary" : "ghost"}
                className="w-full justify-start"
                disabled={disabled}
                aria-pressed={filterDuplicates}
                onClick={() => setOnlyDuplicates(!filterDuplicates)}
              >
                <AlertCircle />
                {filterDuplicates
                  ? "显示全部分集"
                  : `查看 ${duplicateCount} 个重名标题`}
              </Button>
            )}
            {(query || filterDuplicates) && (
              <p className="text-xs text-muted-foreground" role="status">
                找到 {entries.length} 集
              </p>
            )}
          </div>
          <OverlayScrollArea
            className="h-52 md:h-auto md:flex-1"
            viewportRef={scrollRef}
            viewportClassName="scroll-p-2 [overflow-anchor:none]"
          >
            {entries.length === 0 ? (
              <div className="flex flex-col items-center gap-3 px-4 py-8 text-sm text-muted-foreground">
                <p>没有匹配的分集</p>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setQuery("");
                    setOnlyDuplicates(false);
                  }}
                >
                  <X />
                  清除筛选
                </Button>
              </div>
            ) : (
              <div className="px-2 pb-2">
                <div
                  style={{
                    height: virtualizer.getTotalSize(),
                    position: "relative",
                  }}
                >
                  {virtualizer.getVirtualItems().map((row) => {
                    const { segment, index, key } = entries[row.index];
                    const hint = existing.has(key)
                      ? "与现有分集重名"
                      : repeated.has(key)
                        ? "本次标题重复"
                        : null;
                    return (
                      <div
                        key={row.key}
                        className="absolute top-0 left-0 w-full px-1 py-0.5"
                        style={{
                          height: row.size,
                          transform: `translateY(${row.start}px)`,
                        }}
                      >
                        <button
                          type="button"
                          disabled={disabled}
                          onClick={() => {
                            onSelect(index);
                            mobileTriggerRef.current?.focus({
                              preventScroll: true,
                            });
                            setMobileOpen(false);
                          }}
                          aria-current={selected === index ? "true" : undefined}
                          aria-label={`第 ${index + 1} 集 ${segment.title || "未填写标题"}`}
                          title={segment.title}
                          className={cn(
                            "flex h-full w-full min-w-0 items-center gap-3 rounded-lg border px-3 text-left text-sm outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:opacity-50 motion-reduce:transition-none",
                            selected === index
                              ? "border-primary/30 bg-primary/10 text-foreground dark:bg-primary/20"
                              : "border-transparent hover:bg-muted",
                          )}
                        >
                          <span
                            className={cn(
                              "w-7 shrink-0 text-center text-xs tabular-nums",
                              selected === index
                                ? "font-semibold text-primary dark:text-foreground"
                                : "text-muted-foreground",
                            )}
                          >
                            {index + 1}
                          </span>
                          <span className="flex min-w-0 flex-1 flex-col gap-1">
                            <span
                              className={cn(
                                "truncate",
                                selected === index && "font-medium",
                              )}
                            >
                              {segment.title || "未填写标题"}
                            </span>
                            <span className="flex items-center gap-1 text-xs text-muted-foreground">
                              {(segment.end - segment.start).toLocaleString()}{" "}
                              字
                              {hint && (
                                <>
                                  <AlertCircle className="size-3 shrink-0" />
                                  <span className="truncate">{hint}</span>
                                </>
                              )}
                            </span>
                          </span>
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </OverlayScrollArea>
        </div>
      </div>
    </aside>
  );
}
