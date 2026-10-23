"use client";
import { useMemo, useState } from "react";
import type { Edge } from "@xyflow/react";
import { canvasBusinessRelations, type CanvasRelationScope } from "@/lib/canvas/relations";
import type { CanvasGraph } from "@/lib/canvas/types";
import { useCanvasLibrary } from "../library/use-library";

export function useCanvasBusinessRelations(
  graph: CanvasGraph,
  hidden: Set<string>,
) {
  const [scope, setScope] = useState<CanvasRelationScope>(
    "hidden",
  );
  const library = useCanvasLibrary(scope !== "hidden");
  const edges = useMemo<Edge[]>(
    () =>
      scope === "hidden"
        ? []
        : canvasBusinessRelations(graph, library.entries, scope).map(
            (relation) => ({
              ...relation,
              type: "default",
              sourceHandle: "annotation-out",
              targetHandle: "annotation-in",
              selectable: false,
              reconnectable: false,
              focusable: false,
              hidden:
                hidden.has(relation.source) || hidden.has(relation.target),
              style: {
                stroke: "var(--muted-foreground)",
                strokeDasharray: "4 5",
              },
              labelStyle: { fill: "var(--muted-foreground)" },
              labelBgStyle: { fill: "var(--card)" },
            }),
          ),
    [graph, library.entries, scope, hidden],
  );
  return {
    scope,
    setScope,
    edges,
    error: library.error,
    loading: library.loading,
  };
}
