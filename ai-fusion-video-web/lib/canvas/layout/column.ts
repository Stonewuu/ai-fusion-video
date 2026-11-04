import { NODE_GAP, type LayoutUnit, type Point } from "./types";

// 在给定顺序和间距下，让整列尽量靠近各自的目标高度。重叠的相邻节点一起挪动。
export function placeColumn(nodes: LayoutUnit[], positions: Map<string, Point>,
  targets: { total: number; weight: number }[], minTop = -Infinity, maxTop = Infinity) {
  const blocks: { start: number; end: number; total: number; weight: number }[] = [];
  const offsets: number[] = [];
  let offset = 0;
  nodes.forEach((node, index) => {
    const { total, weight } = targets[index];
    offsets.push(offset);
    blocks.push({ start: index, end: index, total: total - offset * weight, weight });
    while (blocks.length > 1) {
      const b = blocks.at(-1)!;
      const a = blocks.at(-2)!;
      if (a.total / a.weight <= b.total / b.weight) {
        break;
      }
      blocks.splice(-2, 2, { start: a.start, end: b.end, total: a.total + b.total, weight: a.weight + b.weight });
    }
    offset += node.height + NODE_GAP;
  });
  for (const block of blocks) {
    const top = Math.max(minTop, Math.min(maxTop, block.total / block.weight));
    for (let i = block.start; i <= block.end; i++) {
      positions.get(nodes[i].id)!.y = top + offsets[i];
    }
  }
}
