"use client";

import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";

export function CanvasToolbarToggle({
  label,
  checked,
  onCheckedChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <Button
      variant={checked ? "secondary" : "outline"}
      size="default"
      aria-pressed={checked}
      disabled={disabled}
      onClick={() => onCheckedChange(!checked)}
    >
      <Check aria-hidden className={checked ? undefined : "invisible"} />
      {label}
    </Button>
  );
}
