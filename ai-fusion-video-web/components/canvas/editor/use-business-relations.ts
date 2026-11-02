"use client";
import { useMemo, useState } from "react";
import type { Edge } from "@xyflow/react";
import {
  canvasBusinessRelations,
  type CanvasRelationAvailability,
  type CanvasRelationScope,
} from "@/lib/canvas/relations";
import type { CanvasGraph } from "@/lib/canvas/types";
import { useCanvasLibrary } from "../library/use-library";

export function useCanvasBusinessRelations(
  graph: CanvasGraph,
  hidden: Set<string>,
) {
  const [scope, setScope] = useState<CanvasRelationScope>(
    "hidden",
  );
  const hasResourcePair = useMemo(
    () => graph.nodes.filter((node) => node.resource).length >= 2,
    [graph.nodes],
  );
  const library = useCanvasLibrary(hasResourcePair);
  const layoutRelations = useMemo(
    () => canvasBusinessRelations(graph, library.entries, "all"),
    [graph, library.entries],
  );
  const allRelations = useMemo(
    () => layoutRelations.filter((relation) => !hidden.has(relation.source) && !hidden.has(relation.target)),
    [layoutRelations, hidden],
  );
  const available = useMemo<CanvasRelationAvailability>(() => ({
    structure: allRelations.some((relation) => relation.category === "structure"),
    assets: allRelations.some((relation) => relation.category === "assets"),
  }), [allRelations]);
  const edges = useMemo<Edge[]>(
    () =>
      scope === "hidden"
        ? []
        : allRelations.filter((relation) => scope === "all" || relation.category === scope).map(
            (relation) => ({
              ...relation,
              type: "default",
              sourceHandle: "annotation-out",
              targetHandle: "annotation-in",
              selectable: false,
              reconnectable: false,
              focusable: false,
              style: {
                stroke: "var(--muted-foreground)",
                strokeDasharray: "4 5",
              },
              labelStyle: { fill: "var(--muted-foreground)" },
              labelBgStyle: { fill: "var(--card)" },
            }),
          ),
    [allRelations, scope],
  );
  return {
    scope,
    setScope,
    available,
    layoutRelations,
    edges,
    error: hasResourcePair ? library.error : null,
    loading: library.loading,
    refresh: library.refresh,
  };
}
