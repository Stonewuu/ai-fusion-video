import { boundsOf, type EdgeRoute, type LayoutRequest, type LayoutResult } from "./types";
import { layoutWorldBoxes } from "./model";
import { crossingCount, traceBounds } from "./crossings";

export function routeCrossings(routes: Record<string, EdgeRoute>) {
  const traces = Object.values(routes).map((route) => traceBounds(route.points));
  let count = 0;
  for (let i = 0; i < traces.length; i++) {
    for (let j = 0; j < i; j++) {
      count += crossingCount(traces[i], traces[j]);
    }
  }
  return count;
}

export function layoutQuality(request: LayoutRequest, result: LayoutResult) {
  const { boxes } = layoutWorldBoxes(request, result.positions);
  const original = layoutWorldBoxes(request).boxes;
  const bounds = boundsOf(Object.keys(result.positions).map((id) => boxes.get(id)!));
  const length = Object.values(result.routes).reduce((total, route) => total + route.points.slice(1)
    .reduce((sum, point, index) => sum + Math.hypot(point.x - route.points[index].x, point.y - route.points[index].y), 0), 0);
  const movement = Object.entries(result.positions).reduce((sum, [id, point]) => sum
    + Math.abs(point.x - original.get(id)!.x) + Math.abs(point.y - original.get(id)!.y), 0);
  const crossings = routeCrossings(result.routes);
  const obstructions = Object.values(result.routes).reduce((sum, route) => sum + route.obstructions, 0);
  const edgeCount = Math.max(1, Object.keys(result.routes).length);
  const nodeArea = Object.keys(result.positions).reduce((sum, id) => sum + boxes.get(id)!.width * boxes.get(id)!.height, 0);
  const area = bounds.width * bounds.height;
  // 遮挡和交叉都按画布上显示的曲线计算，同时避免把整张画布拉成长条。
  const score = (obstructions * 8 + crossings * 4) / edgeCount + length / edgeCount / 400
    + area / nodeArea * 2 + Math.max(bounds.height / bounds.width - 1.5, 0) * 8;
  return [Math.round(score * 1000), obstructions, crossings, Math.round(length), Math.round(area), Math.round(movement)];
}

export function compareQuality(a: number[], b: number[]) {
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) {
      return a[index] - b[index];
    }
  }
  return 0;
}
