"use client";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function CanvasSelect({
  value,
  items,
  onChange,
  placeholder,
  label,
  disabled = false,
}: {
  value: string | null;
  items: { value: string; label: string }[];
  onChange: (value: string) => void;
  placeholder: string;
  label: string;
  disabled?: boolean;
}) {
  return (
    <Select
      value={value}
      items={items}
      disabled={disabled}
      onValueChange={(next) => {
        if (next !== null) onChange(next);
      }}
    >
      <SelectTrigger className="w-full text-sm" aria-label={label}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="text-sm">
        <SelectGroup>
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value} className="text-sm">
              {item.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
