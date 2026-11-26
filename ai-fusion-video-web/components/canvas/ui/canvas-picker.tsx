"use client";

import { useId, useMemo, useRef, useState, type ReactNode } from "react";
import { Check, ChevronsUpDown, Search, X } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { CanvasInput } from "./canvas-field";

export interface CanvasPickerItem {
  value: string;
  label: string;
  description?: string;
  icon?: ReactNode;
  badge?: string;
  keywords?: string;
}

/**
 * 画布下拉选择器：32px 精密微边触发器，支持选项图标、副标题、徽标与键盘可搜索列表。
 */
export function CanvasPicker({
  label,
  value,
  items,
  placeholder,
  emptyIcon,
  searchPlaceholder = "搜索模型…",
  emptyText = "没有找到匹配项",
  side = "top",
  searchable,
  disabled = false,
  className,
  onChange,
}: {
  label: string;
  value: string | null;
  items: CanvasPickerItem[];
  placeholder: string;
  emptyIcon?: ReactNode;
  searchPlaceholder?: string;
  emptyText?: string;
  side?: "top" | "bottom";
  searchable?: boolean;
  disabled?: boolean;
  className?: string;
  onChange: (value: string) => void;
}) {
  const listboxId = useId();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const selected = items.find((item) => item.value === value) ?? null;
  const showSearch = searchable ?? items.length > 7;
  const filtered = useMemo(() => {
    const keyword = query.trim().toLocaleLowerCase();
    if (!keyword) {
      return items;
    }
    return items.filter((item) =>
      `${item.label} ${item.description ?? ""} ${item.keywords ?? ""}`
        .toLocaleLowerCase()
        .includes(keyword),
    );
  }, [items, query]);
  const highlighted = Math.min(activeIndex, Math.max(0, filtered.length - 1));
  const choose = (next: string) => {
    onChange(next);
    setOpen(false);
    setQuery("");
  };
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) {
          setQuery("");
          const current = items.findIndex((item) => item.value === value);
          setActiveIndex(current >= 0 ? current : 0);
        }
      }}
    >
      <PopoverTrigger
        render={
          <button
            type="button"
            role="combobox"
            aria-label={label}
            aria-controls={listboxId}
            aria-expanded={open}
            aria-haspopup="listbox"
            disabled={disabled}
            className={cn(
              "flex h-8 w-full min-w-0 items-center justify-between gap-2 rounded-[10px] border border-input bg-background/90 pl-2.5 pr-2 text-[13px] text-foreground shadow-[0_1px_2px_rgba(0,0,0,0.04)] outline-none transition-[color,background-color,border-color,box-shadow] duration-150 hover:bg-background focus-visible:border-primary/85 focus-visible:ring-[3px] focus-visible:ring-primary/20 disabled:cursor-not-allowed disabled:opacity-45 aria-expanded:border-primary/50 aria-expanded:ring-[3px] aria-expanded:ring-primary/15 dark:bg-white/[0.06] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_1px_2px_rgba(0,0,0,0.22)] dark:hover:bg-white/[0.1] dark:focus-visible:border-primary dark:focus-visible:ring-primary/35 motion-reduce:transition-none",
              className,
            )}
          />
        }
      >
        <span className="flex min-w-0 flex-1 items-center gap-2">
          {selected?.icon ?? emptyIcon ? (
            <span className="flex size-4 shrink-0 items-center justify-center">
              {selected?.icon ?? emptyIcon}
            </span>
          ) : null}
          <span
            className={cn(
              "min-w-0 flex-1 truncate text-left font-medium",
              !selected && "font-normal text-muted-foreground",
            )}
          >
            {selected?.label ?? placeholder}
          </span>
        </span>
        <ChevronsUpDown className="pointer-events-none size-3.5 shrink-0 text-muted-foreground/80" />
      </PopoverTrigger>
      <PopoverContent
        side={side}
        align="start"
        sideOffset={8}
        className="nodrag nopan nowheel flex w-(--anchor-width) min-w-80 max-w-[min(26rem,calc(100vw-2rem))] flex-col gap-1.5 rounded-[16px] border border-foreground/[0.1] bg-popover/96 p-1.5 shadow-[0_4px_16px_rgba(0,0,0,0.08),0_20px_48px_-12px_rgba(0,0,0,0.16)] backdrop-blur-2xl dark:border-white/[0.12] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_20px_48px_rgba(0,0,0,0.58)]"
        onPointerDown={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          event.stopPropagation();
          if (event.key === "ArrowDown") {
            event.preventDefault();
            const next = Math.min(filtered.length - 1, highlighted + 1);
            setActiveIndex(next);
            listRef.current
              ?.querySelector(`[data-picker-index="${next}"]`)
              ?.scrollIntoView({ block: "nearest" });
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            const next = Math.max(0, highlighted - 1);
            setActiveIndex(next);
            listRef.current
              ?.querySelector(`[data-picker-index="${next}"]`)
              ?.scrollIntoView({ block: "nearest" });
          } else if (event.key === "Enter" && filtered[highlighted]) {
            event.preventDefault();
            choose(filtered[highlighted].value);
          }
        }}
      >
        {showSearch && (
          <div className="relative p-0.5">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground/75" />
            <CanvasInput
              autoFocus
              name="canvas-picker-search"
              aria-label={searchPlaceholder}
              placeholder={searchPlaceholder}
              value={query}
              className="pl-8 pr-7"
              onChange={(event) => {
                setQuery(event.target.value);
                setActiveIndex(0);
              }}
            />
            {query && (
              <button
                type="button"
                aria-label="清除搜索"
                className="absolute right-2 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded-[6px] text-muted-foreground transition-colors hover:bg-foreground/[0.08] hover:text-foreground"
                onClick={() => {
                  setQuery("");
                  setActiveIndex(0);
                }}
              >
                <X className="size-3" />
              </button>
            )}
          </div>
        )}
        <div
          id={listboxId}
          ref={listRef}
          role="listbox"
          aria-label={label}
          className="flex max-h-64 flex-col gap-0.5 overflow-y-auto p-0.5"
        >
          {filtered.map((item, index) => {
            const isSelected = item.value === value;
            const isActive = index === highlighted;
            return (
              <button
                key={item.value}
                type="button"
                role="option"
                aria-selected={isSelected}
                data-picker-index={index}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-[10px] px-2 py-1.5 text-left outline-none transition-colors duration-100 focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transition-none",
                  isSelected
                    ? "bg-primary/[0.09] text-foreground"
                    : isActive
                      ? "bg-foreground/[0.06] text-foreground"
                      : "text-foreground/90 hover:bg-foreground/[0.05]",
                )}
                onPointerMove={() => setActiveIndex(index)}
                onClick={() => choose(item.value)}
              >
                {item.icon && (
                  <span className="grid size-7 shrink-0 place-items-center rounded-[8px] border border-foreground/[0.06] bg-background/80 shadow-[0_1px_1px_rgba(0,0,0,0.03)] dark:border-white/[0.08] dark:bg-white/[0.05]">
                    {item.icon}
                  </span>
                )}
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5">
                    <span className="min-w-0 truncate text-[13px] font-medium leading-tight">
                      {item.label}
                    </span>
                    {item.badge && (
                      <span className="shrink-0 rounded-[5px] bg-amber-500/12 px-1.5 py-0.5 text-[10px] font-medium leading-none text-amber-700 dark:text-amber-300">
                        {item.badge}
                      </span>
                    )}
                  </span>
                  {item.description && (
                    <span className="mt-0.5 block truncate font-mono text-[11px] leading-tight text-muted-foreground">
                      {item.description}
                    </span>
                  )}
                </span>
                {isSelected && (
                  <Check className="size-4 shrink-0 text-primary" />
                )}
              </button>
            );
          })}
          {!filtered.length && (
            <p className="px-3 py-6 text-center text-xs text-muted-foreground">
              {emptyText}
            </p>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
