import { layoutLinks, layoutWorldBoxes } from "./model";
import { RouteGrid, simplifyRoute } from "./route-grid";
import { ROUTE_MARGIN, LayoutConstraintError, overlaps, type Box, type EdgeRoute, type LayoutRequest, type Point } from "./types";

function labelPosition(points: Point[], text: string, obstacles: Box[], labels: Box[]) {
  if (!text) {
    return null;
  }
  const width = text.length * 12 + 16;
  const height = 24;
  const segments = points.slice(1).map((point, index) => ({ a: points[index], b: point }))
    .sort((a, b) => Math.hypot(b.a.x - b.b.x, b.a.y - b.b.y) - Math.hypot(a.a.x - a.b.x, a.a.y - a.b.y));
  for (const offset of [0, 20, -20, 36, -36]) {
    for (const { a, b } of segments) {
      for (const fraction of [0.5, 0.25, 0.75]) {
        const center = { x: a.x + (b.x - a.x) * fraction, y: a.y + (b.y - a.y) * fraction };
        if (a.y === b.y) {
          center.y += offset;
        } else {
          center.x += offset;
        }
        const box = { x: center.x - width / 2, y: center.y - height / 2, width, height };
        if (![...obstacles, ...labels].some((other) => overlaps(box, other, 4))) {
          labels.push(box);
          return center;
        }
      }
    }
  }
  throw new LayoutConstraintError("连线标签附近没有足够的空间，请扩大选区后重新整理");
}

export function routeCanvasEdges(request: LayoutRequest, positions: Record<string, Point> = {}) {
  const { boxes, byId } = layoutWorldBoxes(request, positions);
  const hidden = new Set(request.hidden);
  const links = layoutLinks(request).filter((link) => !hidden.has(link.source) && !hidden.has(link.target));
  const ancestors = (id: string) => {
    const result: string[] = [];
    let parent = byId.get(id)!.layout.parentId;
    while (parent) {
      result.push(parent);
      parent = byId.get(parent)!.layout.parentId;
    }
    return result;
  };
  const endpoints = new Map(links.map((link) => {
    const source = boxes.get(link.source)!;
    const target = boxes.get(link.target)!;
    const sourceY = request.geometry[link.source].outputs[link.sourcePort];
    const targetY = request.geometry[link.target].inputs[link.targetPort];
    if (!Number.isFinite(sourceY) || !Number.isFinite(targetY)) {
      throw new Error("部分连线的输入或输出端口已不存在，请先修正连线");
    }
    const start = { x: source.x + source.width, y: source.y + sourceY };
    const end = { x: target.x, y: target.y + targetY };
    return [link.id, { start, end, from: { ...start, x: start.x + ROUTE_MARGIN }, to: { ...end, x: end.x - ROUTE_MARGIN } }];
  }));
  if (!links.length) {
    return {};
  }
  const grids = new Map<string, { grid: RouteGrid; obstacles: Box[] }>();
  const labels: Box[] = [];
  const routes: Record<string, EdgeRoute> = {};
  const allEndpoints = [...endpoints.values()].flatMap(({ from, to }) => [from, to]);
  for (const link of [...links].sort((a, b) => Number(b.kind === "execution") - Number(a.kind === "execution") || a.id.localeCompare(b.id))) {
    // 端点所在的父分组是容器，连线需要穿过其边界；其余分组仍需避让。
    const containers = new Set([...ancestors(link.source), ...ancestors(link.target)]);
    const gridKey = [...containers].sort().join(",");
    let context = grids.get(gridKey);
    if (!context) {
      const obstacles = [...boxes].filter(([id]) => !hidden.has(id) && !containers.has(id)).map(([, box]) => box);
      context = { grid: new RouteGrid(obstacles, allEndpoints), obstacles };
      if (grids.size >= 2) {
        grids.delete(grids.keys().next().value!);
      }
      grids.set(gridKey, context);
    }
    const { start, end, from, to } = endpoints.get(link.id)!;
    const points = simplifyRoute([start, ...context.grid.route(from, to), end]);
    routes[link.id] = { points, label: labelPosition(points, link.label, context.obstacles, labels) };
  }
  return routes;
}
