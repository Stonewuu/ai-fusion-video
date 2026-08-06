"use client";

import { House, RefreshCw, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ErrorRecoveryPanelProps {
  title: string;
  description: string;
  details?: string;
  onRetry: () => void;
  onGoHome?: () => void;
  className?: string;
}

export function ErrorRecoveryPanel({
  title,
  description,
  details,
  onRetry,
  onGoHome,
  className,
}: ErrorRecoveryPanelProps) {
  return (
    <div
      className={cn(
        "flex w-full grow items-center justify-center px-4 pb-6",
        className
      )}
    >
      <div className="w-full max-w-lg rounded-xl border border-border/30 bg-card/50 p-5 backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-destructive/30 bg-destructive/10 text-destructive">
            <TriangleAlert className="size-5" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <h1 className="text-base font-semibold">{title}</h1>
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
        {details ? (
          <p className="mt-4 max-h-24 overflow-auto rounded-lg border border-border/20 bg-background/70 p-3 font-mono text-xs text-muted-foreground">
            {details}
          </p>
        ) : null}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button type="button" onClick={onRetry}>
            <RefreshCw />
            重试
          </Button>
          {onGoHome ? (
            <Button type="button" variant="outline" onClick={onGoHome}>
              <House />
              返回仪表盘
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
