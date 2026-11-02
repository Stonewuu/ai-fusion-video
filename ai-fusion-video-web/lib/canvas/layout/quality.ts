import { boundsOf, type EdgeRoute, type LayoutRequest, type LayoutResult } from "./types";
import { layoutWorldBoxes } from "./model";

export function routeCrossings(routes: Record<string, EdgeRoute>) {
  const horizontal: { id: string; y: number; left: number; right: number }[] = [];
  const vertical: { id: string; x: number; top: number; bottom: number }[] = [];
  for (const [id, { points }] of Object.entries(routes)) {
    for (let index = 1; index < points.length; index++) {
      const a = points[index - 1];
      const b = points[index];
      if (a.y === b.y) {
        horizontal.push({ id, y: a.y, left: Math.min(a.x, b.x), right: Math.max(a.x, b.x) });
      } else {
        vertical.push({ id, x: a.x, top: Math.min(a.y, b.y), bottom: Math.max(a.y, b.y) });
      }
    }
  }
  let count = 0;
  for (const h of horizontal) {
    for (const v of vertical) {
      if (h.id !== v.id && v.x > h.left && v.x < h.right && h.y > v.top && h.y < v.bottom) {
        count++;
      }
    }
  }
  return count;
}

export function layoutQuality(request: LayoutRequest, result: LayoutResult) {
  const { boxes } = layoutWorldBoxes(request, result.positions);
  const original = layoutWorldBoxes(request).boxes;
  const bounds = boundsOf(Object.keys(result.positions).map((id) => boxes.get(id)!));
  const length = Object.values(result.routes).reduce((total, route) => total + route.points.slice(1)
    .reduce((sum, point, index) => sum + Math.abs(point.x - route.points[index].x) + Math.abs(point.y - route.points[index].y), 0), 0);
  const movement = Object.entries(result.positions).reduce((sum, [id, point]) => sum
    + Math.abs(point.x - original.get(id)!.x) + Math.abs(point.y - original.get(id)!.y), 0);
  const crossings = routeCrossings(result.routes);
  const edgeCount = Math.max(1, Object.keys(result.routes).length);
  const nodeArea = Object.keys(result.positions).reduce((sum, id) => sum + boxes.get(id)!.width * boxes.get(id)!.height, 0);
  const area = bounds.width * bounds.height;
  // 交叉、平均线长和空白共同评分，不能为了少几处交叉把整张画布拉成长条。
  const score = crossings / edgeCount * 4 + length / edgeCount / 400
    + area / nodeArea * 2 + Math.max(bounds.height / bounds.width - 1.5, 0) * 8;
  return [Math.round(score * 1000), crossings, Math.round(length), Math.round(area), Math.round(movement)];
}

export function compareQuality(a: number[], b: number[]) {
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) {
      return a[index] - b[index];
    }
  }
  return 0;
}
