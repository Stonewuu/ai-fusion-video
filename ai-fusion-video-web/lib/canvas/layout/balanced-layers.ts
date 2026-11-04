import { LAYER_GAP, NODE_GAP, type LayoutScope, type LayoutUnit, type Point } from "./types";
import { groupParallelInputs, type InputDestination } from "./input-groups";
import { topologicalOrder } from "./model";

export function balancedLayers(scope: LayoutScope, positions: Map<string, Point>,
  destinations: Map<string, InputDestination>, layoutWidth: number) {
  const units = new Map(scope.units.map((node) => [node.id, node]));
  const incoming = new Map(scope.units.map((node) => [node.id, new Set<string>()]));
  const outgoing = new Map(scope.units.map((node) => [node.id, new Set<string>()]));
  for (const link of scope.links) {
    outgoing.get(link.source)!.add(link.target);
    incoming.get(link.target)!.add(link.source);
  }
  const distance = new Map<string, number>();
  for (const id of topologicalOrder([...units.keys()], scope.links).reverse()) {
    distance.set(id, 1 + Math.max(0, ...[...outgoing.get(id)!].map((target) => distance.get(target)!)));
  }
  const area = scope.units.reduce((sum, node) => sum + (node.width + LAYER_GAP) * (node.height + NODE_GAP), 0);
  const height = Math.max(Math.sqrt(area / 1.6), ...scope.units.map((node) => node.height));
  const remaining = new Map([...incoming].map(([id, sources]) => [id, sources.size]));
  let ready = scope.units.filter((node) => remaining.get(node.id) === 0);
  const columns: { nodes: LayoutUnit[]; height: number }[] = [];
  while (ready.length) {
    ready.sort((a, b) => distance.get(b.id)! - distance.get(a.id)!
      || positions.get(a.id)!.y - positions.get(b.id)!.y || a.id.localeCompare(b.id));
    const column = { nodes: [] as LayoutUnit[], height: -NODE_GAP };
    const deferred: LayoutUnit[] = [];
    for (const group of groupParallelInputs(ready, destinations)) {
      const groupHeight = group.reduce((sum, node) => sum + NODE_GAP + node.height, -NODE_GAP);
      // 同一节点的并列输入尽量同列，只有整列会比画布还高时才分开排。
      const batches = groupHeight <= Math.max(height, layoutWidth) ? [group] : group.map((node) => [node]);
      for (const batch of batches) {
        const addedHeight = batch.reduce((sum, node) => sum + NODE_GAP + node.height, 0);
        if (column.nodes.length && column.height + addedHeight > height) {
          deferred.push(...batch);
        } else {
          column.nodes.push(...batch);
          column.height += addedHeight;
        }
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
