import { CheckCircle2, CircleAlert, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { canvasToolPresentation } from "@/lib/canvas/tool-presentation";

export function CanvasToolResult({ toolName, data }: { toolName: string; data: unknown }) {
  const result = canvasToolPresentation(toolName, data);
  const Icon = result.tone === "success" ? CheckCircle2 : result.tone === "info" ? Info : CircleAlert;
  return (
    <div className="flex flex-col gap-3 text-sm">
      <p className={cn("flex items-start gap-2 font-medium", result.tone === "success" && "text-emerald-700 dark:text-emerald-300",
        result.tone === "info" && "text-primary", result.tone === "warning" && "text-amber-700 dark:text-amber-300", result.tone === "error" && "text-destructive")}>
        <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />{result.summary}
      </p>
      {!!result.facts.length && <p className="text-xs leading-relaxed text-muted-foreground">{result.facts.join(" · ")}</p>}
      {!!result.items.length && <ul className="space-y-2 border-t border-border/20 pt-3">{result.items.map((item, index) => (
        <li key={index} className="flex items-start justify-between gap-3 text-xs"><span className="min-w-0 break-words">{item.title}</span>{item.detail && <span className="shrink-0 text-muted-foreground">{item.detail}</span>}</li>
      ))}</ul>}
    </div>
  );
}
