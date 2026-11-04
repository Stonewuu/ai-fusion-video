import type { CanvasOperation } from "../types";
import { layoutWorldBoxes } from "./model";
import type { LayoutRequest, LayoutResult } from "./types";

export function autoLayoutOperations(request: LayoutRequest, result: Pick<LayoutResult, "positions">): CanvasOperation[] {
  const { boxes } = layoutWorldBoxes(request, result.positions);
  const operations: CanvasOperation[] = [];
  for (const node of request.graph.nodes) {
    const point = result.positions[node.id];
    if (!point) {
      continue;
    }
    const parent = node.layout.parentId ? boxes.get(node.layout.parentId)! : { x: 0, y: 0 };
    const x = point.x - parent.x;
    const y = point.y - parent.y;
    if (Math.abs(x - node.layout.x) < 0.01 && Math.abs(y - node.layout.y) < 0.01) {
      continue;
    }
    operations.push({
      type: "canvas.node.layout",
      nodeId: node.id,
      expectedLayoutVersion: node.layoutVersion,
      layout: { ...node.layout, x, y },
    });
  }
  if (operations.length > 500) {
    throw new Error("一次最多整理 500 个节点，请缩小选区后重试");
  }
  return operations;
}
