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
  "h-8 rounded-xl border-transparent bg-transparent px-2.5 text-xs hover:bg-background/80";

export function GenerationQuickSelect({
  value,
  placeholder,
  options,
  className,
  contentClassName,
  onChange,
}: GenerationQuickSelectProps) {
  const selected = options.find((option) => option.value === value) ?? options[0];

  if (options.length <= 1) {
    return (
      <div
        className={cn(
          "flex min-w-0 items-center text-muted-foreground",
          TRIGGER_CLASS,
          className,
        )}
        title={selected?.label || placeholder}
      >
        <span className="truncate">{selected?.label || placeholder}</span>
      </div>
    );
  }

  return (
    <Select
      value={selected?.value ?? value}
      onValueChange={(nextValue) => onChange(String(nextValue))}
      items={options.map((option) => ({
        value: option.value,
        label: option.label,
      }))}
    >
      <SelectTrigger className={cn(TRIGGER_CLASS, className)}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent
        side="top"
        sideOffset={8}
        align="start"
        alignItemWithTrigger={false}
        className={cn("w-auto min-w-24", contentClassName)}
      >
        <SelectGroup>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.menuLabel ?? option.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
