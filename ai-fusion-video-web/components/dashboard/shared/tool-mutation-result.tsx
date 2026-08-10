"use client";

import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { buildToolMutationPresentation } from "./tool-mutation-result-model";

export function ToolMutationResult({
  data,
  toolName,
}: {
  data: unknown;
  toolName: string;
}) {
  const presentation = buildToolMutationPresentation(data, toolName);

  return (
    <div className="space-y-1">
      <p className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        {presentation.failed ? (
          <AlertTriangle className="size-3.5 shrink-0 text-destructive" />
        ) : (
          <CheckCircle2 className="size-3.5 shrink-0 text-green-500" />
        )}
        {presentation.message}
      </p>
      {presentation.summary ? (
        <p className="text-[11px] text-foreground/80">{presentation.summary}</p>
      ) : null}
      {presentation.id !== undefined || presentation.counts.length > 0 ? (
        <p className="text-[10px] text-muted-foreground/60">
          {presentation.id !== undefined
            ? `${presentation.label} ID: ${String(presentation.id)}`
            : null}
          {presentation.id !== undefined && presentation.counts.length > 0
            ? " · "
            : null}
          {presentation.counts.join(" · ")}
        </p>
      ) : null}
    </div>
  );
}
