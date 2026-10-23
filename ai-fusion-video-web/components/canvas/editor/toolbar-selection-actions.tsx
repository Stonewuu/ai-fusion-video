"use client";

import { useEffect, useRef, useState } from "react";
import { useStore as useFlowStore } from "@xyflow/react";
import { shallow } from "zustand/shallow";
import type { CanvasSelectionAction } from "./build-context-actions";
import { CanvasSelectionActionButton } from "./selection-action-button";
import { selectionToolbarPosition } from "@/lib/canvas/selection-toolbar";

export function CanvasSelectionActions({
  nodeIds,
  count,
  edgeCount,
  actions,
}: {
  nodeIds: string[];
  count: number;
  edgeCount: number;
  actions: CanvasSelectionAction[];
}) {
  const root = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const bounds = useFlowStore((state) => {
    const nodes = nodeIds
      .map((id) => state.nodeLookup.get(id))
      .filter((node) => node && !node.hidden);
    if (!nodes.length) return null;
    const [x, y, zoom] = state.transform;
    const left = Math.min(
      ...nodes.map((node) => node!.internals.positionAbsolute.x),
    );
    const top = Math.min(
      ...nodes.map((node) => node!.internals.positionAbsolute.y),
    );
    const right = Math.max(
      ...nodes.map(
        (node) =>
          node!.internals.positionAbsolute.x + (node!.measured.width ?? 0),
      ),
    );
    const bottom = Math.max(
      ...nodes.map(
        (node) =>
          node!.internals.positionAbsolute.y + (node!.measured.height ?? 0),
      ),
    );
    return {
      left: left * zoom + x,
      top: top * zoom + y,
      right: right * zoom + x,
      bottom: bottom * zoom + y,
      width: state.width,
      height: state.height,
    };
  }, shallow);
  const shown = !!bounds && actions.length > 0;
  useEffect(() => {
    if (!shown || !root.current) return;
    const observer = new ResizeObserver(() => {
      const element = root.current;
      if (!element) return;
      const next = { width: element.offsetWidth, height: element.offsetHeight };
      setSize((current) =>
        current.width === next.width && current.height === next.height
          ? current
          : next,
      );
    });
    observer.observe(root.current);
    return () => observer.disconnect();
  }, [shown]);
  if (!shown || !bounds) return null;
  const position = selectionToolbarPosition(bounds, size);
  return (
    <div
      ref={root}
      role="toolbar"
      aria-label="所选节点操作"
      data-canvas-selection-toolbar
      className="nodrag nopan nowheel absolute z-30 flex w-max items-center gap-2 overflow-x-auto rounded-xl border border-border/30 bg-card p-1 shadow-md"
      style={{
        ...position,
        maxWidth: Math.max(0, bounds.width - 16),
        visibility: size.width ? "visible" : "hidden",
      }}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <span className="shrink-0 px-2 text-xs text-muted-foreground">
        {count ? "已选 " + count + " 个节点" : ""}
        {count && edgeCount ? " · " : ""}
        {edgeCount ? edgeCount + " 条连线" : ""}
      </span>
      {actions.map((action) => (
        <CanvasSelectionActionButton key={action.id} action={action} />
      ))}
    </div>
  );
}
