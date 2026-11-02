import type { EdgeRoute, Point } from "./types";

export function roundedRoutePath(points: Point[]) {
  let path = `M ${points[0].x} ${points[0].y}`;
  for (let index = 1; index < points.length - 1; index++) {
    const previous = points[index - 1];
    const corner = points[index];
    const next = points[index + 1];
    const before = Math.hypot(corner.x - previous.x, corner.y - previous.y);
    const after = Math.hypot(next.x - corner.x, next.y - corner.y);
    const radius = Math.min(8, before / 2, after / 2);
    const entry = { x: corner.x + (previous.x - corner.x) * radius / before, y: corner.y + (previous.y - corner.y) * radius / before };
    const exit = { x: corner.x + (next.x - corner.x) * radius / after, y: corner.y + (next.y - corner.y) * radius / after };
    path += ` L ${entry.x} ${entry.y} Q ${corner.x} ${corner.y} ${exit.x} ${exit.y}`;
  }
  const end = points.at(-1)!;
  return `${path} L ${end.x} ${end.y}`;
}

export function routeMatchesEndpoints(route: EdgeRoute, source: Point, target: Point) {
  const start = route.points[0];
  const end = route.points.at(-1)!;
  return Math.abs(start.x - source.x) < 1 && Math.abs(start.y - source.y) < 1
    && Math.abs(end.x - target.x) < 1 && Math.abs(end.y - target.y) < 1;
}
