"use client";
import { useCallback, useEffect, useState, type RefObject } from "react";
import { getViewportForBounds, type ReactFlowInstance, type Viewport } from "@xyflow/react";
import { toast } from "sonner";
import type { CanvasFlowNode } from "./canvas-node";

export function useCanvasViewport(
  key: string,
  flow: ReactFlowInstance<CanvasFlowNode>,
  canvasArea: RefObject<HTMLDivElement | null>,
  reduced: boolean,
  hasNodes: boolean,
  focusRequest: { ids: string[]; sequence: number } | null,
) {
  const [initialView] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return { fit: hasNodes };
      const viewport: Viewport = JSON.parse(raw);
      if (
        ![viewport.x, viewport.y, viewport.zoom].every(Number.isFinite) ||
        viewport.zoom < 0.15 ||
        viewport.zoom > 2
      )
        throw new Error();
      return { fit: false, viewport };
    } catch {
      return {
        fit: false,
        error: "保存的视图无法读取，请点击“适应画布”重新定位",
      };
    }
  });

  const fitNodes = useCallback((ids?: string[]) => {
    const area = canvasArea.current;
    if (!area || !flow.viewportInitialized) return;
    const requested = ids ? new Set(ids) : null;
    const nodes = flow.getNodes().filter((node) => !node.hidden && (!requested || requested.has(node.id)));
    if (!nodes.length) return;
    // fitView waits for every node to be measured, including culled nodes that cannot
    // mount until the view moves. Use their declared bounds and measured visible sizes.
    const viewport = getViewportForBounds(
      flow.getNodesBounds(nodes), area.clientWidth, area.clientHeight, 0.15, 1, ids ? 0.35 : 0.18,
    );
    void flow.setViewport(viewport, { duration: reduced ? 0 : 200 });
  }, [canvasArea, flow, reduced]);

  useEffect(() => {
    if (!initialView.fit || !flow.viewportInitialized) return;
    const frame = requestAnimationFrame(() => fitNodes());
    return () => cancelAnimationFrame(frame);
  }, [initialView.fit, flow.viewportInitialized, fitNodes]);

  useEffect(() => {
    if (!focusRequest) return;
    const frame = requestAnimationFrame(() => fitNodes(focusRequest.ids));
    return () => cancelAnimationFrame(frame);
  }, [focusRequest, fitNodes]);

  return {
    initialView,
    fit: () => fitNodes(),
    saveViewport: (viewport: Viewport) => {
      try {
        localStorage.setItem(key, JSON.stringify(viewport));
      } catch {
        toast.error("当前浏览器无法保存画布视图");
      }
    },
  };
}
