import type { CanvasNode } from "./types";

/** 每个图版本只构建一次分组层次，拖动时复用。 */
export function canvasGraphHierarchy(nodes: CanvasNode[]) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const childCounts = new Map<string, number>();
  const depths = new Map<string, number>();
  for (const node of nodes) {
    if (node.layout.parentId) {
      childCounts.set(
        node.layout.parentId,
        (childCounts.get(node.layout.parentId) ?? 0) + 1,
      );
    }
    const chain: CanvasNode[] = [];
    const visited = new Set<string>();
    let current: CanvasNode | undefined = node;
    while (current && !depths.has(current.id)) {
      if (visited.has(current.id)) {
        throw new Error("画布分组形成循环");
      }
      visited.add(current.id);
      chain.push(current);
      const parentId: string | null = current.layout.parentId;
      current = parentId ? byId.get(parentId) : undefined;
      if (parentId && !current) {
        throw new Error("画布分组不存在");
      }
    }
    let depth = current ? depths.get(current.id)! : -1;
    for (const ancestor of chain.reverse()) {
      depths.set(ancestor.id, ++depth);
    }
  }
  const ordered = [...nodes].sort(
    (a, b) => depths.get(a.id)! - depths.get(b.id)!,
  );
  return { byId, childCounts, ordered };
}
