"use client";
import { useReactFlow, useStore as useFlowStore } from "@xyflow/react";
import { useReducedMotion } from "framer-motion";
import { CircleHelp, Minus, Plus } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasBarDivider, CanvasFloatingBar } from "../ui/canvas-surface";
import {
  CANVAS_MAX_ZOOM,
  CANVAS_MIN_ZOOM,
} from "@/lib/canvas/viewport-limits";
import { useCanvas } from "../canvas-provider";

export function CanvasViewportStatus({
  mode,
  onHelp,
}: {
  mode: "select" | "pan";
  onHelp: () => void;
}) {
  const flow = useReactFlow();
  const reduced = useReducedMotion();
  const scale = useFlowStore((state) => state.transform[2]);
  const count = useCanvas((state) => state.snapshot!.graph.nodes.length);
  const zoom = Math.round(scale * 100);
  const duration = reduced ? 0 : 200;
  return (
    <footer className="@container pointer-events-none absolute inset-x-3 bottom-3 z-20 flex items-center justify-between gap-2 text-xs text-muted-foreground">
      <CanvasFloatingBar>
        <CanvasButton onClick={onHelp}>
          <CircleHelp />
          操作指南
        </CanvasButton>
        <span className="hidden px-2 @2xl:inline">
          {mode === "select"
            ? "拖动空白处框选 · 空格拖动画布"
            : "拖动画布 · 点击“选择”编辑节点"}
        </span>
      </CanvasFloatingBar>
      <CanvasFloatingBar role="group" aria-label="缩放控制">
        <CanvasIconButton
          label="缩小"
          disabled={scale <= CANVAS_MIN_ZOOM}
          onClick={() => void flow.zoomOut({ duration })}
        >
          <Minus />
        </CanvasIconButton>
        <CanvasButton
          className="min-w-16 tabular-nums"
          aria-label={`当前缩放 ${zoom}%，点击恢复 100%`}
          title="恢复 100%"
          onClick={() => void flow.zoomTo(1, { duration })}
        >
          {zoom}%
        </CanvasButton>
        <CanvasIconButton
          label="放大"
          disabled={scale >= CANVAS_MAX_ZOOM}
          onClick={() => void flow.zoomIn({ duration })}
        >
          <Plus />
        </CanvasIconButton>
        <CanvasBarDivider />
        <span role="status" aria-live="off" className="whitespace-nowrap px-2">
          {count} 个节点
        </span>
      </CanvasFloatingBar>
    </footer>
  );
}
