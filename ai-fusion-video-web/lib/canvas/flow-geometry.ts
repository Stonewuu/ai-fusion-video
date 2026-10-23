import { Position, type NodeHandle } from "@xyflow/react";
import type { CanvasCapability, CanvasNode } from "./types";

/** Build hierarchy once per graph revision, rather than scanning every node per drag frame. */
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
      if (visited.has(current.id)) throw new Error("画布分组形成循环");
      visited.add(current.id);
      chain.push(current);
      const parentId: string | null = current.layout.parentId;
      current = parentId ? byId.get(parentId) : undefined;
      if (parentId && !current) throw new Error("画布分组不存在");
    }
    let depth = current ? depths.get(current.id)! : -1;
    for (const ancestor of chain.reverse()) depths.set(ancestor.id, ++depth);
  }
  const ordered = [...nodes].sort(
    (a, b) => depths.get(a.id)! - depths.get(b.id)!,
  );
  return { byId, childCounts, ordered };
}

/**
 * Offscreen nodes need bounds and port anchors before their DOM exists. React Flow replaces
 * these initial dimensions/handles with ResizeObserver measurements when a node enters view.
 * No node body is simplified or collapsed as part of viewport culling.
 */
export function canvasNodeGeometry(
  node: CanvasNode,
  capability: CanvasCapability,
  collapsed: boolean,
  measured?: { width?: number; height?: number },
) {
  const width = measured?.width ?? node.layout.width ?? 300;
  const ports = [...capability.inputs, ...capability.outputs];
  const rich =
    !!node.resource ||
    ["image", "video", "audio", "document"].includes(node.kind);
  const height =
    node.kind === "group"
      ? collapsed
        ? 64
        : (measured?.height ?? node.layout.height ?? 400)
      : (measured?.height ?? (rich ? 320 : 160) + ports.length * 32);
  const handle = (
    id: string,
    output: boolean,
    centerY: number,
  ): NodeHandle => ({
    id,
    type: output ? "source" : "target",
    position: output ? Position.Right : Position.Left,
    x: output ? width - 4 : -4,
    y: centerY - 4,
    width: 8,
    height: 8,
  });
  const handles = [
    handle("annotation-out", true, height / 2),
    handle("annotation-in", false, height / 2),
  ];
  if (node.kind !== "group" || !collapsed) {
    ports.forEach((port, index) =>
      handles.push(
        handle(
          port.id,
          index >= capability.inputs.length,
          height - (ports.length - index) * 32 + 16,
        ),
      ),
    );
  }
  return {
    initialWidth: width,
    initialHeight: height,
    // Once mounted, retain React Flow's actual DOM handle bounds across graph updates.
    // Declared anchors are only needed before the first measurement.
    handles:
      measured?.width !== undefined && measured?.height !== undefined
        ? undefined
        : handles,
  };
}
