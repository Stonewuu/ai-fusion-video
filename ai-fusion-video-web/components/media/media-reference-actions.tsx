"use client";

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

export function MediaReferenceActions({ label, onRemove }: {
  label: string;
  onRemove: () => void;
}) {
  return (
    <div className="absolute inset-x-1 top-1 flex justify-end gap-2">
      <Button variant="secondary" size="icon-xs" title={"移除" + label} aria-label={"移除" + label} onClick={onRemove}>
        <X />
      </Button>
    </div>
  );
}
