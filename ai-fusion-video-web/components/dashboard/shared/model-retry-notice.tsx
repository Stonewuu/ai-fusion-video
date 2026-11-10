import { RotateCcw } from "lucide-react";

export function ModelRetryNotice({ text }: { text: string }) {
  return (
    <p role="status" className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
      <RotateCcw className="size-3.5 shrink-0" />
      {text}
    </p>
  );
}
