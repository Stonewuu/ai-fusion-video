import type { Point } from "./types";

export function traceBounds(points: Point[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return { points, left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys) };
}

export function crossingCount(a: ReturnType<typeof traceBounds>, b: ReturnType<typeof traceBounds>) {
  if (a.right < b.left || b.right < a.left || a.bottom < b.top || b.bottom < a.top) {
    return 0;
  }
  let count = 0;
  for (let i = 1; i < a.points.length; i++) {
    const p = a.points[i - 1];
    const q = a.points[i];
    const dx = q.x - p.x;
    const dy = q.y - p.y;
    for (let j = 1; j < b.points.length; j++) {
      const r = b.points[j - 1];
      const s = b.points[j];
      const ex = s.x - r.x;
      const ey = s.y - r.y;
      const det = dx * ey - dy * ex;
      if (!det) {
        continue;
      }
      const t = ((r.x - p.x) * ey - (r.y - p.y) * ex) / det;
      const u = ((r.x - p.x) * dy - (r.y - p.y) * dx) / det;
      // 每段仅计起点，避免曲线在采样点相交时重复计数。共用端口不算交叉。
      if (t >= 0 && t < 1 && u >= 0 && u < 1 && (i > 1 || t > 0) && (j > 1 || u > 0)) {
        count++;
      }
    }
  }
  return count;
}
