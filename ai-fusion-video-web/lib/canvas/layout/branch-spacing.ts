import { LAYER_GAP, type LayoutScope, type Point } from "./types";

export function balanceBranchSpacing(scope: LayoutScope, positions: Map<string, Point>) {
  const units = new Map(scope.units.map((unit) => [unit.id, unit]));
  const incoming = new Map(scope.units.map((unit) => [unit.id, new Set<string>()]));
  const outgoing = new Map(scope.units.map((unit) => [unit.id, new Set<string>()]));
  for (const link of scope.links) {
    if (positions.get(link.source)!.x + units.get(link.source)!.width < positions.get(link.target)!.x) {
      incoming.get(link.target)!.add(link.source);
      outgoing.get(link.source)!.add(link.target);
    }
  }
  const middle = scope.units.filter((unit) => unit.movable && incoming.get(unit.id)!.size && outgoing.get(unit.id)!.size)
    .sort((a, b) => positions.get(a.id)!.x - positions.get(b.id)!.x || a.id.localeCompare(b.id));
  // 较短的分支也可以占用长分支留下的横向空间。保持两端位置，在前后节点之间平分空白。
  for (let pass = 0; pass < 4; pass++) {
    for (const unit of pass % 2 === 0 ? middle : [...middle].reverse()) {
      const point = positions.get(unit.id)!;
      const left = Math.max(...[...incoming.get(unit.id)!].map((id) => positions.get(id)!.x + units.get(id)!.width));
      const right = Math.min(...[...outgoing.get(unit.id)!].map((id) => positions.get(id)!.x)) - unit.width;
      let low = Math.max(left + LAYER_GAP, scope.bounds?.x ?? -Infinity);
      let high = Math.min(right - LAYER_GAP, scope.bounds ? scope.bounds.x + scope.bounds.width - unit.width : Infinity);
      for (const other of scope.units) {
        const box = positions.get(other.id)!;
        if (point.y + unit.height <= box.y || box.y + other.height <= point.y) {
          continue;
        }
        if (box.x + other.width <= point.x) {
          low = Math.max(low, box.x + other.width + LAYER_GAP);
        } else if (point.x + unit.width <= box.x) {
          high = Math.min(high, box.x - unit.width - LAYER_GAP);
        }
      }
      if (low <= high) {
        point.x = Math.max(low, Math.min(Math.round((left + right) / 2), high));
      }
    }
  }
}
