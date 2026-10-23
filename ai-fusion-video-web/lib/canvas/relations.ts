import type { CanvasGraph } from "./types";
import type { CanvasLibraryEntry } from "./resource-library";

export type CanvasRelationScope = "hidden" | "all" | "structure" | "assets";
export type CanvasRelationCategory = "structure" | "assets";

export function toggleCanvasRelation(scope: CanvasRelationScope, category: CanvasRelationCategory): CanvasRelationScope {
  const enabled = scope === "all" || scope === category;
  const other = category === "structure" ? "assets" : "structure";
  const otherEnabled = scope === "all" || scope === other;
  return enabled ? (otherEnabled ? other : "hidden") : (otherEnabled ? "all" : category);
}

export function canvasBusinessRelations(
  graph: CanvasGraph,
  library: CanvasLibraryEntry[],
  scope: "all" | "structure" | "assets",
) {
  const byResource = new Map<string, string[]>();
  for (const node of graph.nodes) {
    if (!node.resource) continue;
    const key = `${node.resource.type}:${node.resource.id}`;
    byResource.set(key, [...(byResource.get(key) ?? []), node.id]);
  }
  return library.flatMap((entry) => {
    const targets = byResource.get(`${entry.ref.type}:${entry.ref.id}`) ?? [];
    const refs = [
      ...(entry.parent && scope !== "assets"
        ? [{ key: entry.parent, label: "包含" }]
        : []),
      ...(scope !== "structure" ? (entry.relations ?? []) : []),
    ];
    return refs.flatMap((ref) =>
      (byResource.get(ref.key) ?? []).flatMap((source) =>
        targets
          .filter((target) => target !== source)
          .map((target) => ({
            id: `relation:${source}:${target}:${ref.key}`,
            source,
            target,
            label: ref.label,
          })),
      ),
    );
  });
}
