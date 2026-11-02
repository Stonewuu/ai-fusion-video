import { boundsOf, LAYER_GAP, NODE_GAP, type LayoutScope, type LayoutUnit, type Point } from "./types";
import { balancedLayers } from "./balanced-layers";

// 保留 ELK 排好的层内顺序，在相邻层之间对齐端口并收紧空白。
// 单调回归同时满足整列的不重叠约束，不逐个挪节点制造越来越高的新行。
export function compactPlacement(scope: LayoutScope, input: Map<string, Point>, balanced = false) {
  let columns: { right: number; nodes: LayoutUnit[]; height: number }[] = [];
  for (const node of [...scope.units].sort((a, b) => input.get(a.id)!.x - input.get(b.id)!.x || a.id.localeCompare(b.id))) {
    const point = input.get(node.id)!;
    let column = columns.at(-1);
    if (!column || point.x >= column.right) {
      column = { right: point.x + node.width, nodes: [], height: 0 };
      columns.push(column);
    }
    column.right = Math.max(column.right, point.x + node.width);
    column.nodes.push(node);
  }
  for (const column of columns) {
    column.nodes.sort((a, b) => input.get(a.id)!.y - input.get(b.id)!.y || a.id.localeCompare(b.id));
    column.height = column.nodes.reduce((sum, node) => sum + node.height + NODE_GAP, -NODE_GAP);
  }
  if (balanced) {
    columns = balancedLayers(scope, input);
  }
  const original = boundsOf(scope.units.map((node) => ({ ...node, ...input.get(node.id)! })));
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
    const outgoing = scope.links.filter((link) => column.nodes.some((node) => node.id === link.source)).length;
    x += Math.max(...column.nodes.map((node) => node.width)) + Math.max(LAYER_GAP, Math.min(240, outgoing * 4));
  });
  const units = new Map(scope.units.map((node) => [node.id, node]));
  const incoming = new Map(scope.units.map((node) => [node.id, scope.links.filter((link) => link.target === node.id)]));
  const outgoing = new Map(scope.units.map((node) => [node.id, scope.links.filter((link) => link.source === node.id)]));
  for (let pass = 0; pass < 4; pass++) {
    const forward = pass % 2 === 0;
    for (const column of forward ? columns : [...columns].reverse()) {
      const blocks: { start: number; end: number; total: number; weight: number }[] = [];
      const offsets: number[] = [];
      let offset = 0;
      column.nodes.forEach((node, index) => {
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
          const importance = link.kind === "execution" ? 4 : link.kind === "structure" ? 2 : 1;
          total += (positions.get(other.id)!.y + otherPort - localPort) * importance;
          weight += importance;
        }
        if (!weight) {
          total = positions.get(node.id)!.y;
          weight = 1;
        }
        offsets.push(offset);
        blocks.push({ start: index, end: index, total: total - offset * weight, weight });
        while (blocks.length > 1) {
          const b = blocks.at(-1)!;
          const a = blocks.at(-2)!;
          if (a.total / a.weight <= b.total / b.weight) {
            break;
          }
          blocks.pop();
          blocks.pop();
          blocks.push({ start: a.start, end: b.end, total: a.total + b.total, weight: a.weight + b.weight });
        }
        offset += node.height + NODE_GAP;
      });
      for (const block of blocks) {
        const top = Math.max(0, Math.min(budget - column.height, block.total / block.weight));
        for (let index = block.start; index <= block.end; index++) {
          positions.get(column.nodes[index].id)!.y = top + offsets[index];
        }
      }
    }
  }
  return positions;
}
