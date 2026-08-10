"use client";

import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import { ChevronUp, Clock3 } from "lucide-react";
import { cn } from "@/lib/utils";

interface GenerationDurationSliderProps {
  value: number;
  min: number;
  max: number;
  compact?: boolean;
  className?: string;
  onChange: (value: number) => void;
}

export function GenerationDurationSlider({
  value,
  min,
  max,
  compact = false,
  className,
  onChange,
}: GenerationDurationSliderProps) {
  const safeValue = Math.min(max, Math.max(min, value));
  const progress = max > min ? ((safeValue - min) / (max - min)) * 100 : 0;
  const trackStyle = {
    background: `linear-gradient(to right, var(--primary) 0%, var(--primary) ${progress}%, var(--muted) ${progress}%, var(--muted) 100%)`,
  };

  if (compact) {
    return (
      <PopoverPrimitive.Root>
        <PopoverPrimitive.Trigger
          type="button"
          aria-label={`调整视频时长，当前 ${safeValue} 秒`}
          className={cn(
            "group flex h-8 shrink-0 items-center gap-1.5 rounded-xl px-2.5 text-xs text-muted-foreground outline-none transition-colors hover:bg-background/80 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/30 data-[popup-open]:bg-background/80 data-[popup-open]:text-foreground",
            className,
          )}
        >
          <Clock3 className="size-3.5 shrink-0" />
          <span className="inline-flex w-10 justify-end tabular-nums whitespace-nowrap">
            {safeValue} 秒
          </span>
          <ChevronUp className="size-3 shrink-0 transition-transform group-data-[popup-open]:rotate-180" />
        </PopoverPrimitive.Trigger>
        <PopoverPrimitive.Portal>
          <PopoverPrimitive.Positioner
            side="top"
            sideOffset={8}
            align="end"
            className="isolate z-50"
          >
            <PopoverPrimitive.Popup
              initialFocus={false}
              className="w-64 origin-(--transform-origin) rounded-2xl border border-border/55 bg-popover/92 p-3.5 text-popover-foreground shadow-2xl backdrop-blur-xl data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 data-[side=top]:slide-in-from-bottom-2"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs font-medium">视频时长</span>
                <output className="inline-flex w-12 justify-end text-sm font-semibold tabular-nums whitespace-nowrap">
                  {safeValue}
                  <span className="ml-0.5 text-[10px] font-normal text-muted-foreground">
                    秒
                  </span>
                </output>
              </div>
              <input
                type="range"
                min={min}
                max={max}
                step={1}
                value={safeValue}
                aria-label="视频时长"
                onChange={(event) => onChange(Number(event.target.value))}
                style={trackStyle}
                className="mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full accent-primary outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
              />
              <div className="mt-1.5 flex justify-between text-[9px] tabular-nums text-muted-foreground">
                <span>{min} 秒</span>
                <span>{max} 秒</span>
              </div>
            </PopoverPrimitive.Popup>
          </PopoverPrimitive.Positioner>
        </PopoverPrimitive.Portal>
      </PopoverPrimitive.Root>
    );
  }

  return (
    <div
      className={cn(
        "rounded-xl border border-border/40 bg-background/55 p-2.5",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="generation-duration" className="text-[10px] text-muted-foreground">
          视频时长
        </label>
        <output
          htmlFor="generation-duration"
          className="inline-flex w-12 justify-end text-sm font-semibold tabular-nums whitespace-nowrap"
        >
          {safeValue}
          <span className="ml-0.5 text-[10px] font-normal text-muted-foreground">
            秒
          </span>
        </output>
      </div>
      <input
        id="generation-duration"
        type="range"
        min={min}
        max={max}
        step={1}
        value={safeValue}
        aria-label="视频时长"
        onChange={(event) => onChange(Number(event.target.value))}
        style={trackStyle}
        className="mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full accent-primary outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
      />
      <div className="mt-1 flex justify-between text-[9px] tabular-nums text-muted-foreground">
        <span>{min} 秒</span>
        <span>{max} 秒</span>
      </div>
    </div>
  );
}
