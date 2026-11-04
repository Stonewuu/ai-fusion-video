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
  const projectRelations = useMemo(
    () => canvasBusinessRelations(graph, library.entries, "all"),
    [graph, library.entries],
  );
  const allRelations = useMemo(
    () => projectRelations.filter((relation) => !hidden.has(relation.source) && !hidden.has(relation.target)),
    [projectRelations, hidden],
  );
  const available = useMemo<CanvasRelationAvailability>(() => ({
    structure: allRelations.some((relation) => relation.category === "structure"),
    assets: allRelations.some((relation) => relation.category === "assets"),
  }), [allRelations]);
  const visibleRelations = useMemo(
    () => allRelations.filter((relation) => scope === "all" || relation.category === scope),
    [allRelations, scope],
  );
  const edges = useMemo<Edge[]>(
    () => visibleRelations.map((relation) => ({
      ...relation,
      type: "default",
      sourceHandle: "annotation-out",
      targetHandle: "annotation-in",
      selectable: false,
      reconnectable: false,
      focusable: false,
      style: { stroke: "var(--muted-foreground)", strokeDasharray: "4 5" },
      labelStyle: { fill: "var(--muted-foreground)" },
      labelBgStyle: { fill: "var(--card)" },
    })),
    [visibleRelations],
  );
  const error = hasResourcePair ? library.error : null;
  const arrangeUnavailable = scope === "hidden" ? null : error ?? (library.loading ? "正在读取项目关系" : null);
  return {
    scope,
    setScope,
    available,
    visibleRelations,
    arrangeUnavailable,
    edges,
    error,
    loading: library.loading,
    refresh: library.refresh,
  };
}
