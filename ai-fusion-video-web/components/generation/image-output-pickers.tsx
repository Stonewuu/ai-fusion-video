"use client";

import type { ReactNode } from "react";
import { Grid2X2, Proportions, Scan } from "lucide-react";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { resolutionChoices, resolutionLabel, type ImageOutputCapabilities, type ImageOutputSelection } from "@/lib/image-output-sizes";
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

function ResolutionGrid({ tier }: { tier: string }) {
  const level = /^(\d+)k$/i.exec(tier);
  if (!level) {
    return <Scan aria-hidden className="size-5" strokeWidth={1.5} />;
  }
  const divisions = Math.min(Number(level[1]) + 1, 5);
  return (
    <svg aria-hidden viewBox="0 0 26 26" className="size-6">
      <rect x="3" y="3" width="20" height="20" rx="3" fill="currentColor" fillOpacity=".08" stroke="currentColor" strokeWidth="1.5" />
      {Array.from({ length: divisions - 1 }, (_, i) => {
        const at = 3 + (20 * (i + 1)) / divisions;
        return <path key={i} d={`M ${at} 4 v18 M 4 ${at} h18`} stroke="currentColor" strokeOpacity=".55" strokeWidth="1" />;
      })}
    </svg>
  );
}

const itemClass = cn(
  "justify-center gap-1 rounded-lg border border-border/30 bg-background/70 px-2 text-xs font-medium tabular-nums text-foreground",
  "hover:border-border/50 aria-selected:border-primary/55 aria-selected:bg-primary/10 aria-selected:text-primary dark:aria-selected:text-chart-1",
  "focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none",
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
    <Select items={items} value={value || null} disabled={disabled || !items.length} onValueChange={(next) => {
      if (next !== null) {
        onChange(next);
      }
    }}>
      <SelectTrigger aria-label={title} className={cn("text-xs", !compact && "w-full")}>
        {icon}
        <SelectValue placeholder={title} />
      </SelectTrigger>
      <SelectContent side={compact ? "top" : "bottom"} sideOffset={8} alignItemWithTrigger={false}
        className={cn(
          "w-[min(21rem,calc(100vw-1.5rem))] min-w-0 border border-border/40 bg-popover/95 p-2 text-xs shadow-xl",
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

interface Props {
  capabilities: ImageOutputCapabilities | null;
  form: ImageOutputSelection;
  onChange: (patch: Partial<ImageOutputSelection>) => void;
  compact?: boolean;
  disabled?: boolean;
}

export function ImageOutputPickers({ capabilities, form, onChange, compact = false, disabled = false }: Props) {
  const ratios = capabilities?.supportedAspectRatios ?? [];
  const wide = ratios.filter((value) => ratioValue(value) > 1).sort((a, b) => ratioValue(a) - ratioValue(b));
  const tall = ratios.filter((value) => ratioValue(value) < 1).sort((a, b) => ratioValue(b) - ratioValue(a));
  const groups = [
    { title: "", options: ratios.filter((value) => !wide.includes(value) && !tall.includes(value)) },
    { title: "横向", options: wide },
    { title: "纵向", options: tall },
  ];
  const resolutions = resolutionChoices(capabilities, form.ratio);
  return (
    <>
      <OutputSelect title="画面比例" value={form.ratio} items={ratios.map((value) => ({ value, label: ratioLabel(value) }))}
        compact={compact} disabled={disabled} icon={form.ratio ? <RatioFrame value={form.ratio} compact /> : <Proportions className="size-4" />} onChange={(ratio) => onChange({ ratio })}>
        {groups.filter(({ options }) => options.length).map(({ title, options }) => (
          <SelectGroup key={title} className="grid grid-cols-3 gap-1.5 p-1">
            {title && <SelectLabel className="col-span-full px-0 pb-0 pt-1.5 text-xs">{title}</SelectLabel>}
            {options.map((value) => (
              <SelectItem key={value} value={value} className={cn(itemClass,
                "min-h-10 py-1 [&>span:last-child:not(:first-child)]:top-1/2 [&>span:last-child:not(:first-child)]:-translate-y-1/2",
              )}>
                <span className="grid w-full grid-cols-[1.75rem_1fr] items-center gap-1.5">
                  <span className="grid size-7 place-items-center text-primary dark:text-chart-1"><RatioFrame value={value} /></span>
                  <span className="leading-4">{ratioLabel(value)}</span>
                </span>
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </OutputSelect>
      <OutputSelect title="分辨率" value={form.resolution}
        items={resolutions.map((value) => ({ value, label: resolutionLabel(value) }))}
        compact={compact} disabled={disabled} icon={<Grid2X2 aria-hidden className="size-4" />} onChange={(resolution) => onChange({ resolution })}>
        <SelectGroup className="grid grid-cols-3 gap-2 p-1">
          {resolutions.map((tier) => {
            const size = capabilities?.imageSizes?.[tier]?.[form.ratio];
            return (
              <SelectItem key={tier} value={tier} className={cn(itemClass, "py-3 [&>span:last-child:not(:first-child)]:top-1")}>
                <span className="flex w-full flex-col items-center gap-1.5">
                  <span className="text-primary dark:text-chart-1"><ResolutionGrid tier={tier} /></span>
                  <span className="text-sm font-semibold">{resolutionLabel(tier)}</span>
                  {size && <span className="text-[11px] font-normal text-muted-foreground">{size.replace(/[x*]/i, " × ")}</span>}
                </span>
              </SelectItem>
            );
          })}
        </SelectGroup>
      </OutputSelect>
    </>
  );
}
