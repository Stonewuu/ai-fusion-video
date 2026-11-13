"use client";

import { useId } from "react";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import { ChevronUp, Clock3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { generationDurationScale } from "./generation-duration-scale";

interface GenerationDurationSliderProps {
  value: number;
  min: number;
  max: number;
  automatic?: boolean;
  options?: number[];
  compact?: boolean;
  className?: string;
  onChange: (value: number) => void;
}

export function GenerationDurationSlider({ value, min, max, automatic, options, compact = false, className, onChange }: GenerationDurationSliderProps) {
  const id = useId();
  const auto = automatic && value === -1;
  const scale = generationDurationScale(min, max, options);
  const fixed = auto ? scale.first : value;
  const label = auto ? "自动时长" : `${fixed} 秒`;
  const position = scale.position(fixed);
  const progress = scale.max > scale.min ? ((position - scale.min) / (scale.max - scale.min)) * 100 : 0;
  const content = <div className="flex min-w-0 flex-col gap-3 p-1">
    <div className="flex items-center justify-between gap-2">
      <label htmlFor={id} className="text-xs font-medium">视频时长</label>
      <output htmlFor={id} className="text-sm font-semibold tabular-nums">{label}</output>
    </div>
    {automatic && <div className="flex gap-2" role="group" aria-label="时长模式">
      <Button variant={auto ? "secondary" : "ghost"} aria-pressed={!!auto} onClick={() => onChange(-1)}>自动</Button>
      <Button variant={!auto ? "secondary" : "ghost"} aria-pressed={!auto} onClick={() => onChange(fixed)}>指定秒数</Button>
    </div>}
    {auto ? <p className="text-xs text-muted-foreground">由模型决定，最多 {max} 秒；运行额度按上限计算。</p> : <>
      <div className="flex min-h-9 items-center px-1">
        <input id={id} type="range" min={scale.min} max={scale.max} step={1} value={position}
          aria-label="视频时长" aria-valuetext={`${fixed} 秒`} disabled={scale.min === scale.max}
          onChange={(event) => onChange(scale.seconds(Number(event.target.value)))}
          style={{ background: `linear-gradient(to right, var(--primary) 0%, var(--primary) ${progress}%, var(--muted) ${progress}%, var(--muted) 100%)` }}
          className="h-2 w-full cursor-pointer appearance-none rounded-full accent-primary outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-default disabled:opacity-50" />
      </div>
      <div className="flex justify-between text-xs tabular-nums text-muted-foreground"><span>{scale.first} 秒</span><span>{scale.last} 秒</span></div>
    </>}
  </div>;

  if (!compact) {
    return <div className={cn("rounded-lg border border-border/20 bg-background/70 p-3", className)}>{content}</div>;
  }
  return <PopoverPrimitive.Root>
    <PopoverPrimitive.Trigger render={<Button variant="ghost" />} aria-label={`调整视频时长，当前${label}`} title={label}
      className={cn("w-28 shrink-0 sm:w-32", className)}>
      <Clock3 /><span className="min-w-0 flex-1 truncate tabular-nums">{auto ? "自动" : label}</span><ChevronUp />
    </PopoverPrimitive.Trigger>
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Positioner side="top" sideOffset={8} align="end" className="isolate z-50">
        <PopoverPrimitive.Popup initialFocus={false}
          className="w-72 origin-(--transform-origin) rounded-2xl border border-border/40 bg-popover/80 p-3 text-popover-foreground shadow-xl backdrop-blur-xl data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 data-[side=top]:slide-in-from-bottom-2 motion-reduce:transition-none">
          {content}
        </PopoverPrimitive.Popup>
      </PopoverPrimitive.Positioner>
    </PopoverPrimitive.Portal>
  </PopoverPrimitive.Root>;
}
