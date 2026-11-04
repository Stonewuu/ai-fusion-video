import { layoutLinks, layoutWorldBoxes } from "./model";
import { overlaps, type Box, type EdgeRoute, type LayoutRequest, type Point } from "./types";

const middle = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

function sampleCubic(start: Point, a: Point, b: Point, end: Point, points: Point[], depth = 0) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  const nearChord = (p: Point) => Math.abs(dx * (p.y - start.y) - dy * (p.x - start.x)) <= length * 0.5
    && p.x >= Math.min(start.x, end.x) - 0.5 && p.x <= Math.max(start.x, end.x) + 0.5;
  // 相距很远的固定节点也只采样最多 128 段，限制密集画布比较交叉的开销。
  if (depth === 7 || (nearChord(a) && nearChord(b))) {
    points.push(end);
    return;
  }
  const ab = middle(a, b);
  const first = middle(start, a);
  const last = middle(b, end);
  const left = middle(first, ab);
  const right = middle(ab, last);
  const split = middle(left, right);
  sampleCubic(start, first, left, split, points, depth + 1);
  sampleCubic(split, right, last, end, points, depth + 1);
}

// 与 React Flow 的 BezierEdge 共用向右输出、从左输入的曲率约定。采样只用于比较节点位置。
export function sampleCanvasCurve(start: Point, end: Point) {
  const dx = end.x - start.x;
  const offset = dx >= 0 ? dx / 2 : 0.25 * 25 * Math.sqrt(-dx);
  const points = [start];
  sampleCubic(start, { x: start.x + offset, y: start.y }, { x: end.x - offset, y: end.y }, end, points);
  return points;
}

function crossesBox(a: Point, b: Point, box: Box) {
  if (Math.max(a.x, b.x) <= box.x || Math.min(a.x, b.x) >= box.x + box.width
    || Math.max(a.y, b.y) <= box.y || Math.min(a.y, b.y) >= box.y + box.height) {
    return false;
  }
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 || dy === 0) {
    return true;
  }
  const x1 = (box.x - a.x) / dx;
  const x2 = (box.x + box.width - a.x) / dx;
  const y1 = (box.y - a.y) / dy;
  const y2 = (box.y + box.height - a.y) / dy;
  return Math.max(0, Math.min(x1, x2), Math.min(y1, y2)) < Math.min(1, Math.max(x1, x2), Math.max(y1, y2));
}

export function measureLayoutCurves(request: LayoutRequest, positions: Record<string, Point> = {}) {
  const { boxes, ordered } = layoutWorldBoxes(request, positions);
  const hidden = new Set(request.hidden);
  const parents = new Map<string, string[]>();
  for (const node of ordered) {
    const parent = node.layout.parentId;
    parents.set(node.id, parent ? [...parents.get(parent)!, parent] : []);
  }
  const visible = [...boxes].filter(([id]) => !hidden.has(id));
  const routes: Record<string, EdgeRoute> = {};
  for (const link of layoutLinks(request)) {
    if (hidden.has(link.source) || hidden.has(link.target)) {
      continue;
    }
    const output = request.geometry[link.source].outputs[link.sourcePort];
    const input = request.geometry[link.target].inputs[link.targetPort];
    if (!output || !input) {
      throw new Error("部分连线的输入或输出端口已不存在，请先修正连线");
    }
    const source = boxes.get(link.source)!;
    const target = boxes.get(link.target)!;
    const start = { x: source.x + output.x, y: source.y + output.y };
    const end = { x: target.x + input.x, y: target.y + input.y };
    const points = sampleCanvasCurve(start, end);
    const label = link.label ? middle(start, end) : null;
    const width = link.label.length * 12 + 16;
    const labelBox = label && { x: label.x - width / 2, y: label.y - 12, width, height: 24 };
    const connected = new Set([link.source, link.target, ...parents.get(link.source)!, ...parents.get(link.target)!]);
    const obstructions = visible.filter(([id, box]) => !connected.has(id)
      && ((labelBox && overlaps(labelBox, box, 4)) || points.some((p, i) => i > 0 && crossesBox(points[i - 1], p, box)))).length;
    routes[link.id] = { points, label, obstructions };
  }
  return routes;
}
