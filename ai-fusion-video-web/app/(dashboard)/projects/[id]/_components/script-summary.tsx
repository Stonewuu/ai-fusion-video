"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function ScriptSummary({ summary }: { summary: string | null }) {
  const [expanded, setExpanded] = useState(false);
  if (!summary?.trim()) return null;
  const shortened = !expanded && summary.length > 240;
  return (
    <section aria-label="剧本摘要" className="space-y-2">
      <h4 className="text-xs font-medium text-muted-foreground">剧本摘要</h4>
      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground">
        {shortened ? `${summary.slice(0, 240)}…` : summary}
      </p>
      {summary.length > 240 && (
        <Button variant="ghost" size="xs" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
          {expanded ? "收起摘要" : "展开摘要"}
        </Button>
      )}
    </section>
  );
}
