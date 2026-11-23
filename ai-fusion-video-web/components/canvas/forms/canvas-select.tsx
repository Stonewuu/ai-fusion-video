"use client";

import type { ReactNode } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface CanvasSelectItem {
  value: string;
  label: string;
  description?: string;
  icon?: ReactNode;
  badge?: string;
}

export function CanvasSelect({
  value,
  items,
  onChange,
  placeholder,
  label,
  disabled = false,
}: {
  value: string | null;
  items: CanvasSelectItem[];
  onChange: (value: string) => void;
  placeholder: string;
  label: string;
  disabled?: boolean;
}) {
  const selectItems = items.map((item) => ({
    value: item.value,
    label: item.label,
  }));
  const selected = items.find((item) => item.value === value) ?? null;
  return (
    <Select
      value={value}
      items={selectItems}
      disabled={disabled}
      onValueChange={(next) => {
        if (next !== null) {
          onChange(next);
        }
      }}
    >
      <SelectTrigger className="w-full text-sm" aria-label={label}>
        {selected?.icon && (
          <span className="flex size-4 shrink-0 items-center justify-center">
            {selected.icon}
          </span>
        )}
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="text-sm">
        <SelectGroup>
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value} className="text-sm">
              {item.icon && (
                <span className="grid size-6 shrink-0 place-items-center rounded-md bg-muted/70">
                  {item.icon}
                </span>
              )}
              <span className="min-w-0 flex-1 leading-tight">
                <span className="flex items-center gap-1.5">
                  <span className="min-w-0 truncate">{item.label}</span>
                  {item.badge && (
                    <span className="shrink-0 rounded bg-amber-500/12 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:text-amber-300">
                      {item.badge}
                    </span>
                  )}
                </span>
                {item.description && (
                  <span className="mt-0.5 block truncate font-mono text-[11px] text-muted-foreground">
                    {item.description}
                  </span>
                )}
              </span>
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
