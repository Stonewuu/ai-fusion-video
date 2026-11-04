import { LAYER_GAP, LayoutConstraintError, type Box, type LayoutLink, type LayoutUnit } from "./types";

// 按端口确定方向。只有遇到回路或固定边界时，才需要区分必须保持先后的执行依赖。
export function forwardLayoutLinks(units: LayoutUnit[], links: LayoutLink[], bounds: Box | null) {
  const nodes = new Map(units.map((unit) => [unit.id, unit]));
  const incoming = new Map(units.map((unit) => [unit.id, new Set<string>()]));
  const outgoing = new Map(units.map((unit) => [unit.id, new Set<string>()]));
  const left = new Map(units.map((unit) => [unit.id, unit.movable ? bounds?.x ?? -Infinity : unit.x]));
  const right = new Map(units.map((unit) => [unit.id, unit.movable
    ? bounds ? bounds.x + bounds.width - unit.width : Infinity : unit.x]));

  const sorted = [...links].sort((a, b) => a.source.localeCompare(b.source) || a.target.localeCompare(b.target)
    || a.sourcePort.localeCompare(b.sourcePort) || a.targetPort.localeCompare(b.targetPort)
    || (a.inputIndex ?? 0) - (b.inputIndex ?? 0) || a.id.localeCompare(b.id));
  const accepted = new Set<string>();
  const reaches = (start: string, end: string) => {
    const visited = new Set([start]);
    for (const id of visited) {
      if (id === end) {
        return true;
      }
      for (const next of outgoing.get(id)!) {
        visited.add(next);
      }
    }
    return false;
  };
  const tighten = (start: string, value: number, forward: boolean) => {
    const limits = forward ? left : right;
    const adjacent = forward ? outgoing : incoming;
    const queue = [{ id: start, value }];
    for (let i = 0; i < queue.length; i++) {
      const item = queue[i];
      if (forward ? item.value <= limits.get(item.id)! : item.value >= limits.get(item.id)!) {
        continue;
      }
      limits.set(item.id, item.value);
      for (const next of adjacent.get(item.id)!) {
        const span = nodes.get(forward ? item.id : next)!.width + LAYER_GAP;
        queue.push({ id: next, value: item.value + (forward ? span : -span) });
      }
    }
  };
  for (const link of [...sorted.filter((edge) => edge.execution), ...sorted.filter((edge) => !edge.execution)]) {
    const { source, target } = link;
    const from = nodes.get(source)!;
    if (!outgoing.get(source)!.has(target)) {
      if (reaches(target, source)) {
        continue;
      }
      const span = from.width + LAYER_GAP;
      if (left.get(source)! + span > right.get(target)!) {
        if (link.execution && (from.movable || nodes.get(target)!.movable)) {
          throw new LayoutConstraintError("当前选区在固定节点或分组之间没有足够的横向空间，请扩大选区、解锁节点或扩大分组后重试");
        }
        continue;
      }
      outgoing.get(source)!.add(target);
      incoming.get(target)!.add(source);
      tighten(target, left.get(source)! + span, true);
      tighten(source, right.get(target)! - span, false);
    }
    accepted.add(link.id);
  }
  const ports = new Set<string>();
  return sorted.filter((link) => {
    const key = JSON.stringify([link.source, link.sourcePort, link.target, link.targetPort, link.inputIndex ?? 0]);
    if (!accepted.has(link.id) || ports.has(key)) {
      return false;
    }
    ports.add(key);
    return true;
  });
}
