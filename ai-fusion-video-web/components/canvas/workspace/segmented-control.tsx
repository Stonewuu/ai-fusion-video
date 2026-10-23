"use client";

import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface CanvasSegment<T extends string> {
  value: T;
  label: string;
  title?: string;
  icon?: LucideIcon;
  disabled?: boolean;
}

export function CanvasSegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
  compact = false,
  className,
}: {
  label: string;
  value: T;
  options: readonly CanvasSegment<T>[];
  onChange: (value: T) => void;
  compact?: boolean;
  className?: string;
}) {
  const index = options.findIndex((option) => option.value === value);
  return (
    <div
      className={cn(
        "relative inline-flex shrink-0 rounded-xl bg-muted/50",
        className,
      )}
    >
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-xl border border-border/30"
      />
      <div
        role="radiogroup"
        aria-label={label}
        className="relative grid w-full auto-cols-fr grid-flow-col"
        onKeyDown={(event) => {
          const direction = ["ArrowRight", "ArrowDown"].includes(event.key)
            ? 1
            : ["ArrowLeft", "ArrowUp"].includes(event.key)
              ? -1
              : 0;
          if (!direction && event.key !== "Home" && event.key !== "End") return;
          event.preventDefault();
          const available = options.filter((option) => !option.disabled);
          if (!available.length) return;
          const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")];
          const focused = buttons.indexOf(document.activeElement as HTMLButtonElement);
          const current = focused >= 0 ? focused : Math.max(0, available.findIndex((option) => option.value === value));
          const next =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? available.length - 1
                : (current + direction + available.length) % available.length;
          onChange(available[next].value);
          buttons[next]?.focus();
        }}
      >
        {index >= 0 && (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-y-0 left-0 rounded-xl border border-border/30 bg-card shadow-sm transition-transform duration-200 ease-out motion-reduce:transition-none"
            style={{
              width: `${100 / options.length}%`,
              transform: `translateX(${index * 100}%)`,
            }}
          />
        )}
        {options.map((option, optionIndex) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={value === option.value}
            aria-label={option.label}
            title={option.title ?? option.label}
            disabled={option.disabled}
            tabIndex={
              value === option.value ||
              (index < 0 &&
                optionIndex === options.findIndex((item) => !item.disabled))
                ? 0
                : -1
            }
            onClick={() => onChange(option.value)}
            className={cn(
              "relative flex h-9 min-w-9 items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
              compact ? "px-2 @2xl:px-3" : "px-3",
              value === option.value
                ? "font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option.icon && <option.icon className="size-4 shrink-0" />}
            <span className={compact ? "hidden @2xl:inline" : undefined}>
              {option.label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
