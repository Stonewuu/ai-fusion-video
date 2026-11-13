"use client";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface GenerationQuickSelectOption {
  value: string;
  label: string;
  menuLabel?: string;
}

interface GenerationQuickSelectProps {
  value: string;
  placeholder: string;
  options: GenerationQuickSelectOption[];
  className?: string;
  contentClassName?: string;
  onChange: (value: string) => void;
}

const TRIGGER_CLASS =
  "h-9 w-28 min-w-0 shrink-0 text-sm tabular-nums sm:w-32";

export function GenerationQuickSelect({
  value,
  placeholder,
  options,
  className,
  contentClassName,
  onChange,
}: GenerationQuickSelectProps) {
  const selected = options.find((option) => option.value === value) ?? options[0];

  return (
    <Select
      disabled={options.length <= 1}
      value={selected?.value ?? value}
      onValueChange={(nextValue) => onChange(String(nextValue))}
      items={options.map((option) => ({
        value: option.value,
        label: option.label,
      }))}
    >
      <SelectTrigger aria-label={placeholder} title={selected?.label || placeholder} className={cn(TRIGGER_CLASS, className)}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent
        side="top"
        sideOffset={8}
        align="start"
        alignItemWithTrigger={false}
        className={cn("w-auto min-w-28 text-sm", contentClassName)}
      >
        <SelectGroup>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value} className="text-sm">
              {option.menuLabel ?? option.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
