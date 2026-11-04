import { boundsOf, LAYER_GAP, NODE_GAP, type LayoutLink, type LayoutScope, type LayoutUnit, type Point } from "./types";
import { balancedLayers } from "./balanced-layers";
import { groupParallelInputs, type InputDestination } from "./input-groups";
import { alignConnectedRows } from "./connected-rows";
import { topologicalOrder } from "./model";
import { placeColumn } from "./column";

// 保留 ELK 排好的层内顺序，在相邻层之间对齐端口并收紧空白。
// 单调回归同时满足整列的不重叠约束，不逐个挪节点制造越来越高的新行。
export function compactPlacement(scope: LayoutScope, input: Map<string, Point>, balanced = false) {
  const units = new Map(scope.units.map((node) => [node.id, node]));
  const incoming = new Map(scope.units.map((node) => [node.id, [] as LayoutLink[]]));
  const outgoing = new Map(scope.units.map((node) => [node.id, [] as LayoutLink[]]));
  for (const link of scope.links) {
    incoming.get(link.target)!.push(link);
    outgoing.get(link.source)!.push(link);
  }
  const destinations = new Map<string, InputDestination>();
  for (const node of scope.units) {
    const links = outgoing.get(node.id)!;
    if (node.movable && links.length === 1 && !incoming.get(node.id)!.length) {
      const link = links[0];
      destinations.set(node.id, {
        target: link.target, portY: units.get(link.target)!.inputs[link.targetPort].y, index: link.inputIndex ?? 0,
      });
    }
  }
  const ranks = new Map<string, number>();
  let columns: { nodes: LayoutUnit[]; height: number }[] = [];
  const ordered = topologicalOrder([...units.keys()], scope.links);
  for (const id of ordered) {
    const rank = Math.max(0, ...incoming.get(id)!.map((link) => ranks.get(link.source)! + 1));
    ranks.set(id, rank);
    const column = columns[rank] ??= { nodes: [], height: 0 };
    column.nodes.push(units.get(id)!);
  }
  for (const column of columns) {
    column.nodes.sort((a, b) => input.get(a.id)!.y - input.get(b.id)!.y || a.id.localeCompare(b.id));
    column.height = column.nodes.reduce((sum, node) => sum + node.height + NODE_GAP, -NODE_GAP);
  }
  const original = boundsOf(scope.units.map((node) => ({ ...node, ...input.get(node.id)! })));
  const flowWidth = columns.reduce((sum, column) => sum + Math.max(...column.nodes.map((node) => node.width)) + LAYER_GAP, -LAYER_GAP);
  const tallest = Math.max(...columns.map((column) => column.height));
  // 一层的节点已经堆得比整条流程还高时，再比较将并列节点分列的排法。
  if (balanced && tallest > flowWidth) {
    columns = balancedLayers(scope, input, destinations, flowWidth);
  } else if (balanced) {
    // 先放路径较短的起点，再按各条入线的起点顺序安排下游。与 ELK 的层内顺序一起比较。
    const remaining = new Map<string, number>();
    for (const id of [...ordered].reverse()) {
      remaining.set(id, Math.max(0, ...outgoing.get(id)!.map((link) => remaining.get(link.target)! + 1)));
    }
    columns[0].nodes.sort((a, b) => remaining.get(a.id)! - remaining.get(b.id)!);
    const lane = new Map(columns[0].nodes.map((node, i) => [node.id, i]));
    for (const column of columns.slice(1)) {
      for (const node of column.nodes) {
        const sources = [...new Set(incoming.get(node.id)!.map((link) => link.source))];
        lane.set(node.id, sources.reduce((sum, id) => sum + lane.get(id)!, 0) / sources.length);
      }
      column.nodes.sort((a, b) => lane.get(a.id)! - lane.get(b.id)!);
    }
  }
  for (const column of columns) {
    column.nodes = groupParallelInputs(column.nodes, destinations).flat();
  }
  const height = Math.max(...columns.map((column) => column.height));
  const budget = Math.max(height, Math.min(original.height, height + Math.max(...scope.units.map((node) => node.height))));
  const positions = new Map<string, Point>();
  const nodeColumn = new Map<string, number>();
  let x = 0;
  columns.forEach((column, index) => {
    let y = (budget - column.height) / 2;
    for (const node of column.nodes) {
      positions.set(node.id, { x, y });
      nodeColumn.set(node.id, index);
      y += node.height + NODE_GAP;
    }
    const edgeCount = column.nodes.reduce((sum, node) => sum + outgoing.get(node.id)!.length, 0);
    x += Math.max(...column.nodes.map((node) => node.width)) + Math.max(LAYER_GAP, Math.min(240, edgeCount * 4));
  });
  for (let pass = 0; pass < 4; pass++) {
    const forward = pass % 2 === 0;
    for (const column of forward ? columns : [...columns].reverse()) {
      const targets = column.nodes.map((node) => {
        const links = (forward ? incoming : outgoing).get(node.id)!;
        let total = 0;
        let weight = 0;
        for (const link of links) {
          const other = units.get(forward ? link.source : link.target)!;
          if (nodeColumn.get(other.id) === nodeColumn.get(node.id)) {
            continue;
          }
          const localPort = forward ? node.inputs[link.targetPort] : node.outputs[link.sourcePort];
          const otherPort = forward ? other.outputs[link.sourcePort] : other.inputs[link.targetPort];
          total += positions.get(other.id)!.y + otherPort.y - localPort.y;
          weight++;
        }
        if (!weight) {
          total = positions.get(node.id)!.y;
          weight = 1;
        }
        return { total, weight };
      });
      placeColumn(column.nodes, positions, targets, 0, budget - column.height);
    }
  }
  alignConnectedRows(scope, positions, columns.map((column) => column.nodes), incoming);
  return positions;
}
