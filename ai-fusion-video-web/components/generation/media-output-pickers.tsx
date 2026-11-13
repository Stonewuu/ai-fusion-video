"use client";

import type { ReactNode } from "react";
import { Grid2X2, Proportions, Scan } from "lucide-react";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";

function ratioValue(value: string) {
  const [width, height] = value.split(":").map(Number);
  return width / height;
}

function ratioLabel(value: string) {
  return value === "auto" || value === "adaptive" ? "自适应" : value;
}

function RatioFrame({ value, compact = false }: { value: string; compact?: boolean }) {
  const aspect = ratioValue(value);
  const size = compact ? "size-4" : "size-7";
  if (!Number.isFinite(aspect) || aspect <= 0) {
    return <Scan aria-hidden className={size} strokeWidth={1.5} />;
  }
  const width = 24 * Math.min(1, aspect);
  const height = 24 * Math.min(1, 1 / aspect);
  return (
    <svg aria-hidden viewBox="0 0 32 32" className={size}>
      <rect x={(32 - width) / 2} y={(32 - height) / 2} width={width} height={height} rx="2.5"
        fill="currentColor" fillOpacity=".1" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function ResolutionGrid({ value }: { value: string }) {
  const level = /^(\d+)k$/i.exec(value);
  const height = /^(\d+)p$/i.exec(value);
  if (!level && !height) {
    return <Scan aria-hidden className="size-6" strokeWidth={1.5} />;
  }
  const divisions = Math.min(7, level ? Number(level[1]) + 2 : Math.ceil(Number(height![1]) / 360) + 1);
  return (
    <svg aria-hidden viewBox="0 0 26 26" className="size-6">
      <rect x="3" y="3" width="20" height="20" rx="3" fill="currentColor" fillOpacity=".08" stroke="currentColor" strokeWidth="1.5" />
      {Array.from({ length: divisions - 1 }, (_, index) => {
        const at = 3 + (20 * (index + 1)) / divisions;
        return <path key={index} d={`M ${at} 4 v18 M 4 ${at} h18`} stroke="currentColor" strokeOpacity=".55" strokeWidth="1" />;
      })}
    </svg>
  );
}

const itemClass = cn(
  "justify-center gap-1 rounded-lg border border-border/30 bg-background/70 px-2 text-sm font-medium tabular-nums text-foreground",
  "hover:border-border/50 aria-selected:border-primary/55 aria-selected:bg-primary/10 aria-selected:text-primary dark:aria-selected:text-chart-1",
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none",
  "[&>span:first-child]:items-center [&>span:last-child:not(:first-child)]:right-1 [&>span:last-child:not(:first-child)>svg]:size-3",
);

function OutputSelect({ title, value, items, compact, disabled, icon, onChange, children }: {
  title: string;
  value: string;
  items: { value: string; label: string }[];
  compact: boolean;
  disabled?: boolean;
  icon: ReactNode;
  onChange: (value: string) => void;
  children: ReactNode;
}) {
  const select = (
    <Select items={items} value={value || null} disabled={disabled || items.length <= 1} onValueChange={(next) => {
      if (next !== null) {
        onChange(next);
      }
    }}>
      <SelectTrigger aria-label={title} title={items.find((item) => item.value === value)?.label}
        className={cn("h-9 min-w-0 text-sm tabular-nums", compact ? "w-28 shrink-0 sm:w-32" : "w-full")}>
        {icon}
        <SelectValue placeholder={title} />
      </SelectTrigger>
      <SelectContent side={compact ? "top" : "bottom"} sideOffset={8} alignItemWithTrigger={false}
        className={cn(
          "w-[min(21rem,calc(100vw-1.5rem))] min-w-0 border border-border/40 bg-popover/95 p-2 text-sm shadow-xl",
          "[&_[role=option][aria-selected=true]:is(:focus,[data-highlighted])]:bg-primary/10",
        )}>
        <p className="flex items-center gap-2 px-1 pb-2 pt-1 text-sm font-semibold">
          <span aria-hidden className="grid size-7 place-items-center rounded-lg bg-primary/10 text-primary dark:text-chart-1">
            {icon}
          </span>
          {title}
        </p>
        {children}
      </SelectContent>
    </Select>
  );
  return compact ? select : <div className="flex min-w-0 flex-col gap-2"><span className="text-xs text-muted-foreground">{title}</span>{select}</div>;
}

interface PickerProps {
  value: string;
  compact?: boolean;
  disabled?: boolean;
  onChange: (value: string) => void;
}

export function AspectRatioPicker({ value, options, compact = false, disabled, onChange }: PickerProps & { options: string[] }) {
  const wide = options.filter((ratio) => ratioValue(ratio) > 1).sort((left, right) => ratioValue(left) - ratioValue(right));
  const tall = options.filter((ratio) => ratioValue(ratio) < 1).sort((left, right) => ratioValue(right) - ratioValue(left));
  const groups = [
    { title: "", options: options.filter((ratio) => !wide.includes(ratio) && !tall.includes(ratio)) },
    { title: "横向", options: wide },
    { title: "纵向", options: tall },
  ];
  return (
    <OutputSelect title="画面比例" value={value} items={options.map((ratio) => ({ value: ratio, label: ratioLabel(ratio) }))}
      compact={compact} disabled={disabled} icon={value ? <RatioFrame value={value} compact /> : <Proportions className="size-4" />} onChange={onChange}>
      {groups.filter((group) => group.options.length).map((group) => (
        <SelectGroup key={group.title} className="grid grid-cols-3 gap-2 p-1">
          {group.title && <SelectLabel className="col-span-full px-0 pb-0 pt-1 text-xs">{group.title}</SelectLabel>}
          {group.options.map((ratio) => (
            <SelectItem key={ratio} value={ratio} className={cn(itemClass,
              "min-h-12 py-2 [&>span:last-child:not(:first-child)]:top-1/2 [&>span:last-child:not(:first-child)]:-translate-y-1/2",
            )}>
              <span className="grid w-full grid-cols-[1.75rem_1fr] items-center gap-1">
                <span className="grid size-7 place-items-center text-primary dark:text-chart-1"><RatioFrame value={ratio} /></span>
                <span className="text-xs leading-4">{ratioLabel(ratio)}</span>
              </span>
            </SelectItem>
          ))}
        </SelectGroup>
      ))}
    </OutputSelect>
  );
}

export interface ResolutionOption {
  value: string;
  label: string;
  detail?: string;
}

export function ResolutionPicker({ value, options, compact = false, disabled, onChange }: PickerProps & { options: ResolutionOption[] }) {
  return (
    <OutputSelect title="分辨率" value={value} items={options} compact={compact} disabled={disabled}
      icon={<Grid2X2 aria-hidden className="size-4" />} onChange={onChange}>
      <SelectGroup className="grid grid-cols-3 gap-2 p-1">
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value} className={cn(itemClass, "py-3 [&>span:last-child:not(:first-child)]:top-1")}>
            <span className="flex w-full flex-col items-center gap-2">
              <span className="text-primary dark:text-chart-1"><ResolutionGrid value={option.value} /></span>
              <span className="text-sm font-semibold">{option.label}</span>
              {option.detail && <span className="text-xs font-normal text-muted-foreground">{option.detail}</span>}
            </span>
          </SelectItem>
        ))}
      </SelectGroup>
    </OutputSelect>
  );
}
