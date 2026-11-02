import type { ReactFlowInstance } from "@xyflow/react";
import { canvasNodeGeometry } from "@/lib/canvas/flow-geometry";
import type { CanvasGeometry } from "@/lib/canvas/layout/types";
import type { CanvasCatalog, CanvasGraph } from "@/lib/canvas/types";

export function captureLayoutLabels(flow: ReactFlowInstance) {
  return Object.fromEntries(flow.getEdges().filter((edge) => !edge.id.startsWith("relation:")
    && (typeof edge.label === "string" || typeof edge.label === "number"))
    .map((edge) => [edge.id, String(edge.label)]));
}

export function captureLayoutGeometry(graph: CanvasGraph, catalog: CanvasCatalog, flow: ReactFlowInstance): CanvasGeometry {
  const capabilities = new Map(catalog.nodes.map((node) => [node.kind, node]));
  const geometry: CanvasGeometry = {};
  for (const node of graph.nodes) {
    const internal = flow.getInternalNode(node.id);
    const declared = canvasNodeGeometry(node, capabilities.get(node.kind)!, node.layout.collapsed);
    const width = internal?.measured.width ?? declared.initialWidth;
    const height = internal?.measured.height ?? declared.initialHeight;
    const handleMap = (output: boolean) => {
      const measured = output ? internal?.internals.handleBounds?.source : internal?.internals.handleBounds?.target;
      const handles = measured ?? declared.handles!.filter((handle) => handle.type === (output ? "source" : "target"));
      return Object.fromEntries(handles.filter((handle) => handle.id)
        .map((handle) => [handle.id!, handle.y + handle.height! / 2]));
    };
    geometry[node.id] = { width, height, inputs: handleMap(false), outputs: handleMap(true) };
  }
  return geometry;
}
