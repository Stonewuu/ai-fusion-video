"use client";
import { Button } from "@/components/ui/button";
import type { useCanvasBusinessCommand } from "./use-business-command";

export function CanvasBusinessCommandState({
  command,
  onReload,
}: {
  command: ReturnType<typeof useCanvasBusinessCommand>;
  onReload: () => void;
}) {
  if (!command.error && !command.pending && !command.busy) return null;
  return (
    <div
      role={command.error ? "alert" : "status"}
      className="space-y-2 rounded-lg border border-border/20 bg-background/70 p-3"
    >
      <p
        className={
          command.error
            ? "text-sm text-destructive"
            : "text-sm text-muted-foreground"
        }
      >
        {command.error ?? (command.busy ? "正在提交…" : "上次操作尚未确认")}
      </p>
      {!command.busy && (
        <div className="flex flex-wrap gap-2">
          {command.pending && (
            <Button size="sm" variant="outline" onClick={command.retry}>
              再次确认
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              command.discard();
              onReload();
            }}
          >
            放弃重试并刷新
          </Button>
        </div>
      )}
    </div>
  );
}
