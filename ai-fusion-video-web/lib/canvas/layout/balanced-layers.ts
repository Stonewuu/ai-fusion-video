import { LAYER_GAP, NODE_GAP, type LayoutScope, type LayoutUnit, type Point } from "./types";

export function balancedLayers(scope: LayoutScope, positions: Map<string, Point>) {
  const units = new Map(scope.units.map((node) => [node.id, node]));
  const incoming = new Map(scope.units.map((node) => [node.id, new Set<string>()]));
  const outgoing = new Map(scope.units.map((node) => [node.id, new Set<string>()]));
  // ELK 已经处理了辅助关系的回路，沿它确定的方向安排层次；回边仍交给路由器绘制。
  for (const link of scope.links) {
    if (positions.get(link.source)!.x < positions.get(link.target)!.x) {
      outgoing.get(link.source)!.add(link.target);
      incoming.get(link.target)!.add(link.source);
    }
  }
  const distance = new Map<string, number>();
  for (const node of [...scope.units].sort((a, b) => positions.get(b.id)!.x - positions.get(a.id)!.x)) {
    distance.set(node.id, 1 + Math.max(0, ...[...outgoing.get(node.id)!].map((id) => distance.get(id)!)));
  }
  const area = scope.units.reduce((sum, node) => sum + (node.width + LAYER_GAP) * (node.height + NODE_GAP), 0);
  const height = Math.max(Math.sqrt(area / 1.6), ...scope.units.map((node) => node.height));
  const remaining = new Map([...incoming].map(([id, sources]) => [id, sources.size]));
  let ready = scope.units.filter((node) => remaining.get(node.id) === 0);
  const columns: { right: number; nodes: LayoutUnit[]; height: number }[] = [];
  while (ready.length) {
    ready.sort((a, b) => distance.get(b.id)! - distance.get(a.id)!
      || positions.get(a.id)!.y - positions.get(b.id)!.y || a.id.localeCompare(b.id));
    const column = { right: 0, nodes: [] as LayoutUnit[], height: -NODE_GAP };
    const deferred: LayoutUnit[] = [];
    for (const node of ready) {
      if (column.nodes.length && column.height + NODE_GAP + node.height > height) {
        deferred.push(node);
      } else {
        column.nodes.push(node);
        column.height += NODE_GAP + node.height;
      }
    }
    for (const node of column.nodes) {
      for (const id of outgoing.get(node.id)!) {
        remaining.set(id, remaining.get(id)! - 1);
        if (remaining.get(id) === 0) {
          deferred.push(units.get(id)!);
        }
      }
    }
    column.nodes.sort((a, b) => positions.get(a.id)!.y - positions.get(b.id)!.y || a.id.localeCompare(b.id));
    columns.push(column);
    ready = deferred;
  }
  const order = new Map<string, number>();
  const updateOrder = (column: (typeof columns)[number]) => {
    column.nodes.forEach((node, index) => order.set(node.id, (index + 0.5) / column.nodes.length));
  };
  columns.forEach(updateOrder);
  for (let pass = 0; pass < 4; pass++) {
    const adjacent = pass % 2 === 0 ? incoming : outgoing;
    for (const column of pass % 2 === 0 ? columns : [...columns].reverse()) {
      const barycenter = (id: string) => {
        const peers = [...adjacent.get(id)!];
        return peers.length ? peers.reduce((sum, peer) => sum + order.get(peer)!, 0) / peers.length : order.get(id)!;
      };
      const centers = new Map(column.nodes.map((node) => [node.id, barycenter(node.id)]));
      column.nodes.sort((a, b) => centers.get(a.id)! - centers.get(b.id)! || a.id.localeCompare(b.id));
      updateOrder(column);
    }
  }
  return columns;
}
