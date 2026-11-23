"use client";

import type { LucideIcon } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/** 画布工具条中的分段切换：32px 高，与同排工具按钮齐平，仅显示图标时以提示说明各选项。 */
export function CanvasSegmented<T extends string>({
  label,
  value,
  options,
  iconOnly = false,
  className,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly {
    value: T;
    label: string;
    hint?: string;
    icon?: LucideIcon;
  }[];
  iconOnly?: boolean;
  className?: string;
  onChange: (value: T) => void;
}) {
  const index = options.findIndex((option) => option.value === value);
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        "relative grid h-8 shrink-0 auto-cols-fr grid-flow-col items-center rounded-[10px] bg-foreground/[0.055] p-[3px] dark:bg-white/[0.07]",
        className,
      )}
      onKeyDown={(event) => {
        const step = ["ArrowRight", "ArrowDown"].includes(event.key)
          ? 1
          : ["ArrowLeft", "ArrowUp"].includes(event.key)
            ? -1
            : 0;
        if (!step) {
          return;
        }
        event.preventDefault();
        const next = (Math.max(0, index) + step + options.length) % options.length;
        onChange(options[next].value);
        event.currentTarget
          .querySelectorAll<HTMLButtonElement>("button")
          [next]?.focus();
      }}
    >
      {index >= 0 && (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-y-[3px] left-[3px] rounded-[7px] bg-background shadow-[0_1px_2px_rgba(0,0,0,0.09),0_0_0_1px_rgba(0,0,0,0.04)] transition-transform duration-200 ease-out dark:bg-white/[0.14] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.12),0_1px_2px_rgba(0,0,0,0.3)] motion-reduce:transition-none"
          style={{
            width: `calc((100% - 6px) / ${options.length})`,
            transform: `translateX(${index * 100}%)`,
          }}
        />
      )}
      {options.map((option) => {
        const checked = option.value === value;
        const button = (
          <button
            type="button"
            role="radio"
            aria-checked={checked}
            aria-label={option.label}
            tabIndex={checked ? 0 : -1}
            className={cn(
              "relative flex h-[26px] min-w-[26px] items-center justify-center gap-1.5 rounded-[7px] px-2 text-xs outline-none transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transition-none [&_svg]:size-3.5 [&_svg]:shrink-0",
              checked
                ? "font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
            onClick={() => onChange(option.value)}
          >
            {option.icon && <option.icon aria-hidden />}
            {!iconOnly && option.label}
          </button>
        );
        return iconOnly ? (
          <Tooltip key={option.value}>
            <TooltipTrigger render={button} />
            <TooltipContent>{option.hint ?? option.label}</TooltipContent>
          </Tooltip>
        ) : (
          <span key={option.value} className="contents">
            {button}
          </span>
        );
      })}
    </div>
  );
}
