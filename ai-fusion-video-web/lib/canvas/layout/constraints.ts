import { topologicalOrder } from "./model";
import { LAYER_GAP, NODE_GAP, LayoutConstraintError, overlaps, type Box, type LayoutLink, type LayoutScope, type Point } from "./types";

// 分组收缩为一个矩形后可能产生回路。此处只对能保持先后顺序的边施加横向约束，回边仍保留并参与路由。
function directionLinks(scope: LayoutScope) {
  const units = new Map(scope.units.map((unit) => [unit.id, unit]));
  const outgoing = new Map(scope.units.map((unit) => [unit.id, new Set<string>()]));
  const result: LayoutLink[] = [];
  const candidates = scope.links.filter((link) => link.kind !== "reference"
    && (units.get(link.source)!.movable || units.get(link.target)!.movable))
    .sort((a, b) => Number(b.kind === "execution") - Number(a.kind === "execution") || a.id.localeCompare(b.id));
  for (const link of candidates) {
    const queue = [link.target];
    const visited = new Set(queue);
    for (let index = 0; index < queue.length; index++) {
      for (const next of outgoing.get(queue[index])!) {
        if (!visited.has(next)) {
          visited.add(next);
          queue.push(next);
        }
      }
    }
    if (!visited.has(link.source)) {
      outgoing.get(link.source)!.add(link.target);
      result.push(link);
    }
  }
  return result;
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

export function constrainPlacement(scope: LayoutScope, proposed: Map<string, Point>) {
  const moving = scope.units.filter((unit) => unit.movable);
  const anchors = scope.units.filter((unit) => !unit.movable);
  const shift = anchors.length ? {
    x: median(anchors.map((node) => node.x - proposed.get(node.id)!.x)),
    y: median(anchors.map((node) => node.y - proposed.get(node.id)!.y)),
  } : {
    x: Math.round(Math.min(...moving.map((node) => node.x))) - Math.min(...moving.map((node) => proposed.get(node.id)!.x)),
    y: Math.round(Math.min(...moving.map((node) => node.y))) - Math.min(...moving.map((node) => proposed.get(node.id)!.y)),
  };
  const nodes = new Map(scope.units.map((node) => [node.id, node]));
  const positions = new Map(scope.units.map((node) => [node.id, node.movable ? {
    x: Math.round(proposed.get(node.id)!.x + shift.x),
    y: Math.round(proposed.get(node.id)!.y + shift.y),
  } : { x: node.x, y: node.y }]));
  const links = directionLinks(scope);
  const ordered = topologicalOrder(scope.units.map((node) => node.id), links);
  const next = new Map(scope.units.map((node) => [node.id, [] as string[]]));
  const previous = new Map(scope.units.map((node) => [node.id, [] as string[]]));
  for (const link of links) {
    next.get(link.source)!.push(link.target);
    previous.get(link.target)!.push(link.source);
  }
  const upper = new Map(scope.units.map((node) => [node.id, node.movable
    ? scope.bounds ? scope.bounds.x + scope.bounds.width - node.width : Infinity : node.x]));
  for (const id of [...ordered].reverse()) {
    const node = nodes.get(id)!;
    if (node.movable) {
      upper.set(id, Math.min(upper.get(id)!, ...next.get(id)!.map((target) => upper.get(target)! - node.width - LAYER_GAP)));
    }
  }
  for (const id of ordered) {
    const node = nodes.get(id)!;
    if (!node.movable) {
      continue;
    }
    const lower = Math.max(scope.bounds?.x ?? -Infinity,
      ...previous.get(id)!.map((source) => positions.get(source)!.x + nodes.get(source)!.width + LAYER_GAP));
    if (lower > upper.get(id)!) {
      throw new LayoutConstraintError("当前选区在固定节点或分组之间没有足够的横向空间，请扩大选区、解锁节点或扩大分组后重试");
    }
    positions.get(id)!.x = Math.max(lower, Math.min(positions.get(id)!.x, upper.get(id)!));
  }
  const occupied: Box[] = [...scope.obstacles, ...scope.units.filter((unit) => !unit.movable)];
  for (const node of [...moving].sort((a, b) => positions.get(a.id)!.y - positions.get(b.id)!.y || a.id.localeCompare(b.id))) {
    const position = positions.get(node.id)!;
    const minY = scope.bounds?.y ?? -Infinity;
    const maxY = scope.bounds ? scope.bounds.y + scope.bounds.height - node.height : Infinity;
    const candidates = new Set([
      Math.max(minY, Math.min(position.y, maxY)),
      ...occupied.flatMap((box) => [box.y - node.height - NODE_GAP, box.y + box.height + NODE_GAP]),
    ]);
    const y = [...candidates].filter((value) => value >= minY && value <= maxY)
      .sort((a, b) => Math.abs(a - position.y) - Math.abs(b - position.y) || a - b)
      .find((value) => !occupied.some((box) => overlaps({ ...node, x: position.x, y: value }, box, NODE_GAP)));
    if (y === undefined) {
      throw new LayoutConstraintError("分组内没有足够的空间容纳选区，请扩大分组或减少本次整理的节点");
    }
    position.y = y;
    occupied.push({ ...node, ...position });
  }
  return Object.fromEntries(moving.map((node) => [node.id, positions.get(node.id)!]));
}
