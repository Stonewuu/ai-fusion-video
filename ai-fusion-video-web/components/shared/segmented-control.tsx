"use client";

import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface Segment<T extends string | number> {
  value: T;
  label: string;
  title?: string;
  icon?: LucideIcon;
  disabled?: boolean;
}

export function SegmentedControl<T extends string | number>({
  label,
  value,
  options,
  onChange,
  compact = false,
  className,
}: {
  label: string;
  value: T;
  options: readonly Segment<T>[];
  onChange: (value: T) => void;
  compact?: boolean;
  className?: string;
}) {
  const index = options.findIndex((option) => option.value === value);
  const tabStop = index >= 0 && !options[index].disabled
    ? index
    : options.findIndex((option) => !option.disabled);
  return (
    <div
      data-slot="segmented-control"
      className={cn(
        "relative inline-flex h-9 max-w-full shrink-0 rounded-xl border border-input bg-muted p-0.5",
        className,
      )}
    >
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
          if (!direction && event.key !== "Home" && event.key !== "End") {
            return;
          }
          event.preventDefault();
          const available = options.filter((option) => !option.disabled);
          if (!available.length) {
            return;
          }
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
            className="pointer-events-none absolute inset-y-0 left-0 rounded-lg bg-background shadow-sm transition-transform duration-200 ease-out motion-reduce:transition-none"
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
            tabIndex={optionIndex === tabStop ? 0 : -1}
            onClick={() => onChange(option.value)}
            className={cn(
              "relative flex h-full min-w-8 items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm outline-none transition-colors duration-150 focus-visible:z-10 focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
              compact ? "px-2" : "px-3",
              value === option.value
                ? "font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option.icon && <option.icon aria-hidden="true" className="size-4 shrink-0" />}
            <span className={compact ? "sr-only" : undefined}>
              {option.label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
