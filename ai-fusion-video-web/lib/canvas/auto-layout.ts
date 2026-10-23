import type { CanvasEdge, CanvasGraph, CanvasNode } from "./types";
import { worldPosition } from "./graph";

export interface CanvasNodeGeometry {
  width: number;
  height: number;
  inputs: Record<string, number>;
  outputs: Record<string, number>;
}
export type CanvasGeometry = Record<string, CanvasNodeGeometry>;
type Point = { x: number; y: number };
type Link = {
  source: string;
  target: string;
  sourcePort: string | null;
  targetPort: string | null;
};

/** Layer a DAG from left to right, then give each continuous path its own lane. */
export function arrangeCanvasPaths(
  graph: CanvasGraph,
  nodes: CanvasNode[],
  geometry: CanvasGeometry,
): Map<string, Point> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const all = new Map(graph.nodes.map((node) => [node.id, node]));
  const original = new Map(
    nodes.map((node) => [node.id, worldPosition(node, graph.nodes)]),
  );
  const size = (id: string) => ({
    width: geometry[id]?.width ?? byId.get(id)!.layout.width ?? 300,
    height: geometry[id]?.height ?? byId.get(id)!.layout.height ?? 280,
  });
  const endpoint = (id: string): string | null => {
    let current: string | null = id;
    while (current) {
      if (byId.has(current)) return current;
      current = all.get(current)?.layout.parentId ?? null;
    }
    return null;
  };
  const toLink = (edge: CanvasEdge): Link | null => {
    const source = endpoint(edge.sourceNodeId);
    const target = endpoint(edge.targetNodeId);
    if (!source || !target || source === target) return null;
    return {
      source,
      target,
      sourcePort: source === edge.sourceNodeId ? edge.kind === "execution" ? edge.sourcePort : "annotation-out" : null,
      targetPort: target === edge.targetNodeId ? edge.kind === "execution" ? edge.targetPort : "annotation-in" : null,
    };
  };
  const links = graph.edges.filter((edge) => edge.kind === "execution").map(toLink).filter((link): link is Link => !!link);
  const outgoing = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  const incoming = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  for (const link of links) {
    outgoing.get(link.source)!.add(link.target);
    incoming.get(link.target)!.add(link.source);
  }
  // 来源与注释也组成可读的创作链；它们不能倒转执行顺序或制造布局循环。
  for (const edge of graph.edges.filter((edge) => edge.kind !== "execution")) {
    const link = toLink(edge);
    if (!link || outgoing.get(link.source)!.has(link.target)) continue;
    const pending = [link.target];
    const visited = new Set<string>();
    while (pending.length) {
      const id = pending.pop()!;
      if (visited.has(id)) continue;
      visited.add(id);
      pending.push(...outgoing.get(id)!);
    }
    if (visited.has(link.source)) continue;
    links.push(link);
    outgoing.get(link.source)!.add(link.target);
    incoming.get(link.target)!.add(link.source);
  }
  const compare = (a: string, b: string) =>
    original.get(a)!.y - original.get(b)!.y ||
    original.get(a)!.x - original.get(b)!.x ||
    a.localeCompare(b);
  const degree = new Map(
    nodes.map((node) => [node.id, incoming.get(node.id)!.size]),
  );
  const levels = new Map(nodes.map((node) => [node.id, 0]));
  const ready = nodes
    .map((node) => node.id)
    .filter((id) => !degree.get(id))
    .sort(compare);
  const ordered: string[] = [];
  while (ready.length) {
    const id = ready.shift()!;
    ordered.push(id);
    for (const target of outgoing.get(id)!) {
      levels.set(target, Math.max(levels.get(target)!, levels.get(id)! + 1));
      degree.set(target, degree.get(target)! - 1);
      if (!degree.get(target)) {
        ready.push(target);
        ready.sort(compare);
      }
    }
  }
  if (ordered.length !== nodes.length)
    throw new Error("选区的连线跨分组形成循环，请展开分组并选择具体节点后整理");
  const distance = new Map(nodes.map((node) => [node.id, 0]));
  for (const id of [...ordered].reverse())
    for (const target of outgoing.get(id)!)
      distance.set(id, Math.max(distance.get(id)!, distance.get(target)! + 1));

  const components: string[][] = [];
  const unseen = new Set(ordered);
  for (const seed of [...ordered].sort(compare)) {
    if (!unseen.delete(seed)) continue;
    const component = [seed];
    for (let cursor = 0; cursor < component.length; cursor++) {
      const id = component[cursor];
      for (const neighbor of [...outgoing.get(id)!, ...incoming.get(id)!])
        if (unseen.delete(neighbor)) component.push(neighbor);
    }
    components.push(component);
  }
  const result = new Map<string, Point>();
  const left = Math.min(...[...original.values()].map((point) => point.x));
  let rowTop = Math.min(...[...original.values()].map((point) => point.y));
  const portY = (id: string, port: string | null, output: boolean) => {
    const ports = output ? geometry[id]?.outputs : geometry[id]?.inputs;
    return port && ports?.[port] !== undefined
      ? ports[port]
      : size(id).height / 2;
  };
  for (const component of components.filter(
    (component) => component.length > 1,
  )) {
    const widths = new Map<number, number>();
    for (const id of component)
      widths.set(
        levels.get(id)!,
        Math.max(widths.get(levels.get(id)!) ?? 0, size(id).width),
      );
    const columns = new Map<number, number>();
    let columnLeft = left;
    for (const level of [...widths.keys()].sort((a, b) => a - b)) {
      columns.set(level, columnLeft);
      columnLeft += widths.get(level)! + 96;
    }
    const remaining = new Set(component);
    while (remaining.size) {
      const seed = [...remaining].sort(
        (a, b) =>
          levels.get(a)! - levels.get(b)! ||
          distance.get(b)! - distance.get(a)! ||
          compare(a, b),
      )[0];
      const path: { id: string; y: number }[] = [{ id: seed, y: 0 }];
      remaining.delete(seed);
      while (true) {
        const previous = path[path.length - 1];
        const next = [...outgoing.get(previous.id)!]
          .filter((id) => remaining.has(id))
          .sort(
            (a, b) => distance.get(b)! - distance.get(a)! || compare(a, b),
          )[0];
        if (!next) break;
        const link = links.find(
          (link) => link.source === previous.id && link.target === next,
        )!;
        path.push({
          id: next,
          y:
            previous.y +
            portY(previous.id, link.sourcePort, true) -
            portY(next, link.targetPort, false),
        });
        remaining.delete(next);
      }
      const minY = Math.min(...path.map((item) => item.y));
      const bottom = Math.max(
        ...path.map((item) => item.y + size(item.id).height),
      );
      for (const item of path)
        result.set(item.id, {
          x: columns.get(levels.get(item.id)!)!,
          y: rowTop + item.y - minY,
        });
      rowTop += bottom - minY + 80;
    }
    rowTop += 32;
  }
  // Loose media and notes share compact rows instead of creating a tall column.
  const loose = components.filter((component) => component.length === 1).flat();
  const columnCount = Math.ceil(Math.sqrt(loose.length));
  let looseX = left;
  let rowHeight = 0;
  loose.forEach((id, index) => {
    if (index && index % columnCount === 0) {
      rowTop += rowHeight + 80;
      looseX = left;
      rowHeight = 0;
    }
    result.set(id, { x: looseX, y: rowTop });
    looseX += size(id).width + 96;
    rowHeight = Math.max(rowHeight, size(id).height);
  });
  return result;
}
