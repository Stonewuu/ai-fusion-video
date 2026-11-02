import { ROUTE_MARGIN, LayoutConstraintError, type Box, type Point } from "./types";

const key = (value: number) => value;

class MinHeap {
  private values: { state: number; cost: number }[] = [];

  push(state: number, cost: number) {
    const value = { state, cost };
    let index = this.values.length;
    this.values.push(value);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.values[parent].cost <= cost) {
        break;
      }
      this.values[index] = this.values[parent];
      index = parent;
    }
    this.values[index] = value;
  }

  pop() {
    const first = this.values[0];
    const last = this.values.pop();
    if (this.values.length && last) {
      let index = 0;
      while (index * 2 + 1 < this.values.length) {
        let child = index * 2 + 1;
        if (child + 1 < this.values.length && this.values[child + 1].cost < this.values[child].cost) {
          child++;
        }
        if (last.cost <= this.values[child].cost) {
          break;
        }
        this.values[index] = this.values[child];
        index = child;
      }
      this.values[index] = last;
    }
    return first;
  }
}

export function simplifyRoute(points: Point[]) {
  const result: Point[] = [];
  for (const point of points) {
    const previous = result.at(-1);
    if (previous?.x === point.x && previous.y === point.y) {
      continue;
    }
    const before = result.at(-2);
    if (before && previous && ((before.x === previous.x && previous.x === point.x)
      || (before.y === previous.y && previous.y === point.y))) {
      result.pop();
    }
    result.push(point);
  }
  return result;
}

export class RouteGrid {
  private xs: number[];
  private ys: number[];
  private xIndex: Map<number, number>;
  private yIndex: Map<number, number>;
  private horizontal: Uint8Array;
  private vertical: Uint8Array;
  private usageH: Uint16Array;
  private usageV: Uint16Array;
  private distance: Float64Array;
  private previous: Int32Array;

  constructor(boxes: Box[], endpoints: Point[]) {
    const padded = boxes.map((box) => ({
      x: key(box.x - ROUTE_MARGIN), y: key(box.y - ROUTE_MARGIN),
      right: key(box.x + box.width + ROUTE_MARGIN), bottom: key(box.y + box.height + ROUTE_MARGIN),
    }));
    const xs = [...padded.flatMap((box) => [box.x, box.right]), ...endpoints.map((point) => key(point.x))];
    const ys = [...padded.flatMap((box) => [box.y, box.bottom]), ...endpoints.map((point) => key(point.y))];
    this.xs = [...new Set([...xs, Math.min(...xs) - 48, Math.max(...xs) + 48])].sort((a, b) => a - b);
    this.ys = [...new Set([...ys, Math.min(...ys) - 48, Math.max(...ys) + 48])].sort((a, b) => a - b);
    this.xIndex = new Map(this.xs.map((x, index) => [x, index]));
    this.yIndex = new Map(this.ys.map((y, index) => [y, index]));
    const count = this.xs.length * this.ys.length;
    if (count > 750000) {
      throw new Error("当前画布的连线计算范围过大，请分成多个画布后整理");
    }
    this.horizontal = new Uint8Array(count);
    this.vertical = new Uint8Array(count);
    this.usageH = new Uint16Array(count);
    this.usageV = new Uint16Array(count);
    this.distance = new Float64Array(count * 2);
    this.previous = new Int32Array(count * 2);
    const width = this.xs.length;
    for (const box of padded) {
      const left = this.xIndex.get(box.x)!;
      const right = this.xIndex.get(box.right)!;
      const top = this.yIndex.get(box.y)!;
      const bottom = this.yIndex.get(box.bottom)!;
      for (let y = top + 1; y < bottom; y++) {
        this.horizontal.fill(1, y * width + left, y * width + right);
      }
      for (let y = top; y < bottom; y++) {
        this.vertical.fill(1, y * width + left + 1, y * width + right);
      }
    }
  }

  route(start: Point, end: Point): Point[] {
    const width = this.xs.length;
    const cell = (point: Point) => this.yIndex.get(key(point.y))! * width + this.xIndex.get(key(point.x))!;
    const first = cell(start);
    const last = cell(end);
    const heuristic = (index: number) => Math.abs(this.xs[index % width] - end.x)
      + Math.abs(this.ys[Math.floor(index / width)] - end.y);
    this.distance.fill(Infinity);
    this.previous.fill(-1);
    const heap = new MinHeap();
    for (const direction of [0, 1]) {
      this.distance[first * 2 + direction] = 0;
      heap.push(first * 2 + direction, heuristic(first));
    }
    let finish = -1;
    let entry = heap.pop();
    while (entry) {
      const current = entry.state >> 1;
      const direction = entry.state % 2;
      if (entry.cost > this.distance[entry.state] + heuristic(current) + 0.001) {
        entry = heap.pop();
        continue;
      }
      if (current === last) {
        finish = entry.state;
        break;
      }
      const x = current % width;
      const y = Math.floor(current / width);
      const visit = (target: number, nextDirection: number, segment: number, length: number) => {
        const blocked = nextDirection === 0 ? this.horizontal : this.vertical;
        if (blocked[segment]) {
          return;
        }
        const usage = nextDirection === 0 ? this.usageH : this.usageV;
        const crossing = nextDirection === 0 ? this.usageV[target] : this.usageH[target];
        const cost = this.distance[entry!.state] + length + (direction === nextDirection ? 0 : 24)
          + Math.min(usage[segment], 4) * 8 + Math.min(crossing, 4) * 24;
        const state = target * 2 + nextDirection;
        if (cost + 0.001 < this.distance[state]) {
          this.distance[state] = cost;
          this.previous[state] = entry!.state;
          heap.push(state, cost + heuristic(target));
        }
      };
      if (x > 0) {
        visit(current - 1, 0, current - 1, this.xs[x] - this.xs[x - 1]);
      }
      if (x + 1 < width) {
        visit(current + 1, 0, current, this.xs[x + 1] - this.xs[x]);
      }
      if (y > 0) {
        visit(current - width, 1, current - width, this.ys[y] - this.ys[y - 1]);
      }
      if (y + 1 < this.ys.length) {
        visit(current + width, 1, current, this.ys[y + 1] - this.ys[y]);
      }
      entry = heap.pop();
    }
    if (finish === -1) {
      throw new LayoutConstraintError("部分连线端口被其他节点遮挡，请先移开重叠节点后重试");
    }
    const cells: number[] = [];
    for (let state = finish; state !== -1; state = this.previous[state]) {
      cells.push(state >> 1);
    }
    cells.reverse();
    for (let index = 1; index < cells.length; index++) {
      const a = cells[index - 1];
      const b = cells[index];
      const horizontal = Math.floor(a / width) === Math.floor(b / width);
      (horizontal ? this.usageH : this.usageV)[Math.min(a, b)]++;
    }
    return simplifyRoute(cells.map((index) => ({ x: this.xs[index % width], y: this.ys[Math.floor(index / width)] })));
  }
}
