"use client";

import { useMemo } from "react";
import type { Edge } from "@xyflow/react";
import { canvasSelectionChain } from "@/lib/canvas/selection-chain";
import { cn } from "@/lib/utils";
import type { CanvasFlowNode } from "./canvas-node";

export function useCanvasChainHighlight(
  nodes: CanvasFlowNode[],
  edges: Edge[],
  selection: readonly string[],
) {
  const chain = useMemo(
    () => selection.length ? canvasSelectionChain(edges, selection) : null,
    [edges, selection],
  );
  const highlightedNodes = useMemo(() => {
    if (!chain) {
      return nodes;
    }
    return nodes.map((node) => chain.nodeIds.has(node.id)
      ? { ...node, className: cn(node.className, "canvas-chain-node") }
      : node);
  }, [chain, nodes]);
  const highlightedEdges = useMemo<Edge[]>(() => {
    if (!chain) {
      return edges;
    }
    return edges.map<Edge>((edge) => chain.edgeIds.has(edge.id)
      ? {
          ...edge,
          className: cn(edge.className, "canvas-chain-edge"),
          style: {
            ...edge.style,
            strokeWidth: edge.selected ? 3 : 2.5,
            vectorEffect: "non-scaling-stroke",
          },
        }
      : edge);
  }, [chain, edges]);

  return {
    active: chain !== null,
    nodes: highlightedNodes,
    edges: highlightedEdges,
  };
}
