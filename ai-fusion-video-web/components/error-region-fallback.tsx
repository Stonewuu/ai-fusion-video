"use client";

import { RefreshCw, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ErrorRegionFallback({
  label,
  onRetry,
}: {
  label: string;
  onRetry: () => void;
}) {
  return (
    <div className="fixed inset-x-3 top-3 z-50 flex items-center justify-between gap-3 rounded-2xl border border-border/40 bg-popover/80 p-3 shadow-xl backdrop-blur-xl">
      <div className="flex min-w-0 items-center gap-2 text-sm">
        <TriangleAlert className="size-4 shrink-0 text-destructive" aria-hidden="true" />
        <span className="truncate">{label}</span>
      </div>
      <Button type="button" size="sm" onClick={onRetry}>
        <RefreshCw />
        重试
      </Button>
    </div>
  );
}
