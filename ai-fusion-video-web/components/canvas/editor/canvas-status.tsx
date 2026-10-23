"use client";
import { useStore as useFlowStore } from "@xyflow/react";
import { useCanvas } from "../canvas-provider";
import { Button } from "@/components/ui/button";
import { CircleHelp } from "lucide-react";

export function CanvasViewportStatus({
  mode,
  onHelp,
}: {
  mode: "select" | "pan";
  onHelp: () => void;
}) {
  const zoom = useFlowStore((state) => Math.round(state.transform[2] * 100));
  const count = useCanvas((state) => state.snapshot!.graph.nodes.length);
  return (
    <footer className="@container pointer-events-none absolute inset-x-3 bottom-3 z-20 flex items-center justify-between gap-2 text-xs text-muted-foreground">
      <div className="pointer-events-auto flex items-center gap-2 rounded-2xl border border-border/40 bg-popover/90 p-1 shadow-sm backdrop-blur-xl">
        <Button variant="ghost" size="xs" onClick={onHelp}>
          <CircleHelp />
          操作指南
        </Button>
        <span className="hidden pr-2 @2xl:inline">
          {mode === "select"
            ? "拖动空白处框选 · 空格拖动画布"
            : "拖动画布 · 点击“选择”编辑节点"}
        </span>
      </div>
      <span
        role="status"
        aria-live="off"
        aria-label="画布缩放与节点数量"
        className="flex shrink-0 items-center self-stretch rounded-2xl border border-border/40 bg-popover/90 px-3 shadow-sm backdrop-blur-xl"
      >
        {zoom}% · {count} 个节点
      </span>
    </footer>
  );
}
