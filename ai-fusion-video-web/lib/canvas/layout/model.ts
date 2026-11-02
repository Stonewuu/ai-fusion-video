import { canvasGraphHierarchy } from "../hierarchy";
import type { CanvasNode } from "../types";
import {
  boundsOf,
  type Box,
  type LayoutLink,
  type LayoutRequest,
  type LayoutScope,
  type LayoutUnit,
  type Point,
} from "./types";

export function layoutLinks(request: LayoutRequest): LayoutLink[] {
  return [
    ...request.graph.edges.map((edge): LayoutLink => ({
      id: edge.id,
      source: edge.sourceNodeId,
      target: edge.targetNodeId,
      sourcePort: edge.kind === "execution" ? edge.sourcePort! : "annotation-out",
      targetPort: edge.kind === "execution" ? edge.targetPort! : "annotation-in",
      kind: edge.kind === "execution" ? "execution" : "reference",
      label: request.labels?.[edge.id] ?? (edge.kind === "execution" && edge.inputIndex !== null && edge.inputIndex > 0
        ? String(edge.inputIndex + 1) : ""),
    })),
    ...request.relations.map((relation): LayoutLink => ({
      ...relation,
      sourcePort: "annotation-out",
      targetPort: "annotation-in",
      kind: relation.category === "structure" ? "structure" : "reference",
    })),
  ].sort((a, b) => a.id.localeCompare(b.id));
}

export function layoutWorldBoxes(request: LayoutRequest, positions: Record<string, Point> = {}) {
  const hierarchy = canvasGraphHierarchy(request.graph.nodes);
  const boxes = new Map<string, Box>();
  for (const node of hierarchy.ordered) {
    const parent = node.layout.parentId ? boxes.get(node.layout.parentId)! : { x: 0, y: 0 };
    const size = request.geometry[node.id];
    if (!size || !Number.isFinite(size.width) || !Number.isFinite(size.height)
      || size.width <= 0 || size.height <= 0) {
      throw new Error(`无法读取「${node.title}」的尺寸，请等待节点加载后重试`);
    }
    boxes.set(node.id, {
      ...size,
      ...(positions[node.id] ?? { x: parent.x + node.layout.x, y: parent.y + node.layout.y }),
    });
  }
  return { ...hierarchy, boxes };
}

export function topologicalOrder(ids: string[], links: LayoutLink[]) {
  const incoming = new Map(ids.map((id) => [id, 0]));
  const outgoing = new Map(ids.map((id) => [id, [] as string[]]));
  for (const link of links) {
    outgoing.get(link.source)!.push(link.target);
    incoming.set(link.target, incoming.get(link.target)! + 1);
  }
  const queue = ids.filter((id) => incoming.get(id) === 0).sort();
  const ordered: string[] = [];
  for (let index = 0; index < queue.length; index++) {
    const id = queue[index];
    ordered.push(id);
    for (const target of outgoing.get(id)!) {
      incoming.set(target, incoming.get(target)! - 1);
      if (incoming.get(target) === 0) {
        queue.push(target);
      }
    }
  }
  if (ordered.length !== ids.length) {
    throw new Error("执行连线形成循环，无法确定节点的先后顺序");
  }
  return ordered;
}

export function buildLayoutScopes(request: LayoutRequest): LayoutScope[] {
  const { boxes, byId, ordered } = layoutWorldBoxes(request);
  const links = layoutLinks(request);
  topologicalOrder(ordered.map((node) => node.id), links.filter((link) => link.kind === "execution"));
  const selected = new Set(request.selection.length ? request.selection : ordered.map((node) => node.id));
  const hidden = new Set(request.hidden);
  const ancestors = (node: CanvasNode) => {
    const result: string[] = [];
    let parent = node.layout.parentId;
    while (parent) {
      result.push(parent);
      parent = byId.get(parent)!.layout.parentId;
    }
    return result;
  };
  const parents = new Map(ordered.map((node) => [node.id, ancestors(node)]));
  const locked = new Set(ordered.filter((node) => node.layout.locked).map((node) => node.id));
  const protectedGroups = new Set([...locked].flatMap((id) => parents.get(id)!));
  const movable = ordered.filter((node) => selected.has(node.id) && !hidden.has(node.id)
    && !locked.has(node.id) && !protectedGroups.has(node.id)
    && !parents.get(node.id)!.some((id) => selected.has(id) || locked.has(id)));
  if (!movable.length) {
    throw new Error("没有可整理的节点；锁定节点及包含锁定节点的分组会保持原位");
  }
  if (movable.length > 500) {
    throw new Error("一次最多整理 500 个节点，请缩小选区后重试");
  }
  const scopes: LayoutScope[] = [];
  const parentIds = [...new Set(movable.map((node) => node.layout.parentId))];
  for (const parentId of parentIds) {
    const movingIds = new Set(movable.filter((node) => node.layout.parentId === parentId).map((node) => node.id));
    const siblings = ordered.filter((node) => node.layout.parentId === parentId);
    const siblingIds = new Set(siblings.map((node) => node.id));
    const endpoint = (id: string) => [id, ...parents.get(id)!].find((item) => siblingIds.has(item)) ?? id;
    const scopedLinks = links.flatMap((link): LayoutLink[] => {
      const source = endpoint(link.source);
      const target = endpoint(link.target);
      if (source === target || (!siblingIds.has(source) && !siblingIds.has(target))) {
        return [];
      }
      return [{
        ...link, source, target,
        sourcePort: source === link.source ? link.sourcePort : "annotation-out",
        targetPort: target === link.target ? link.targetPort : "annotation-in",
      }];
    });
    const unitIds = new Set([...siblingIds, ...scopedLinks.flatMap((link) => [link.source, link.target])]);
    const units: LayoutUnit[] = [...unitIds].sort().map((id) => {
      const node = byId.get(id)!;
      const box = boxes.get(id)!;
      // 分组整体移动。展开分组的占地包含超出边框的子节点，避免盖住相邻链路。
      const descendants = node.kind === "group" && !node.layout.collapsed
        ? ordered.filter((child) => parents.get(child.id)!.includes(id)).map((child) => boxes.get(child.id)!) : [];
      const occupied = boundsOf([box, ...descendants]);
      if (movingIds.has(id) && (occupied.x < box.x || occupied.y < box.y)) {
        throw new Error(`「${node.title}」的子节点超出分组左侧或顶部，请先调整分组边界`);
      }
      return {
        id, x: box.x, y: box.y,
        width: Math.max(box.width, occupied.x + occupied.width - box.x),
        height: Math.max(box.height, occupied.y + occupied.height - box.y),
        movable: movingIds.has(id),
        inputs: request.geometry[id].inputs,
        outputs: request.geometry[id].outputs,
      };
    });
    const obstacles = ordered.filter((node) => !hidden.has(node.id) && !unitIds.has(node.id)
      && !parents.get(node.id)!.some((id) => unitIds.has(id))
      && ![...movingIds].some((id) => parents.get(id)!.includes(node.id))).map((node) => boxes.get(node.id)!);
    const container = parentId ? boxes.get(parentId)! : null;
    scopes.push({
      units, links: scopedLinks, obstacles,
      bounds: container ? {
        x: container.x + 24, y: container.y + 64,
        width: container.width - 48, height: container.height - 88,
      } : null,
    });
  }
  return scopes;
}
