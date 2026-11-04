import { placeColumn } from "./column";
import { NODE_GAP, type Box, type LayoutLink, type LayoutScope, type LayoutUnit, type Point } from "./types";

// 分支和汇合只有在两端都能选出同一条延续路径时才并入一行。
function continuation(peers: Set<string>, opposite: Map<string, Set<string>>) {
  if (peers.size === 1) {
    return [...peers][0];
  }
  const exclusive = [...peers].filter((peer) => opposite.get(peer)!.size === 1);
  return exclusive.length === 1 ? exclusive[0] : undefined;
}

export function alignConnectedRows(scope: LayoutScope, positions: Map<string, Point>, columns: LayoutUnit[][],
  inputLinks: Map<string, LayoutLink[]>) {
  const units = new Map(scope.units.map((node) => [node.id, node]));
  const incoming = new Map(scope.units.map((node) => [node.id, new Set<string>()]));
  const outgoing = new Map(scope.units.map((node) => [node.id, new Set<string>()]));
  for (const link of scope.links) {
    outgoing.get(link.source)!.add(link.target);
    incoming.get(link.target)!.add(link.source);
  }
  const successor = new Map([...outgoing].map(([id, peers]) => [id, continuation(peers, incoming)]));
  const predecessor = new Map([...incoming].map(([id, peers]) => [id, continuation(peers, outgoing)]));
  const next = new Map<string, { target: string; offset: number; count: number }>();
  const joined = new Set<string>();
  for (const link of scope.links) {
    if (successor.get(link.source) !== link.target || predecessor.get(link.target) !== link.source) {
      continue;
    }
    const offset = units.get(link.source)!.outputs[link.sourcePort].y - units.get(link.target)!.inputs[link.targetPort].y;
    const step = next.get(link.source);
    if (step) {
      step.offset += offset;
      step.count++;
    } else {
      next.set(link.source, { target: link.target, offset, count: 1 });
      joined.add(link.target);
    }
  }
  if (!next.size) {
    return;
  }
  const rows = scope.units.filter((node) => !joined.has(node.id)).map((start) => {
    const members: { node: LayoutUnit; offset: number }[] = [];
    let node = start;
    let offset = 0;
    let preferred = 0;
    while (true) {
      members.push({ node, offset });
      preferred += positions.get(node.id)!.y - offset;
      const step = next.get(node.id);
      if (!step) {
        break;
      }
      offset += step.offset / step.count;
      node = units.get(step.target)!;
    }
    return { members, preferred: preferred / members.length };
  }).sort((a, b) => a.preferred - b.preferred || a.members[0].node.id.localeCompare(b.members[0].node.id));

  const placed: Box[] = [];
  for (const { members, preferred } of rows) {
    const blocked: { top: number; bottom: number }[] = [];
    for (const { node, offset } of members) {
      const x = positions.get(node.id)!.x;
      for (const box of placed) {
        if (x < box.x + box.width + NODE_GAP && box.x < x + node.width + NODE_GAP) {
          blocked.push({ top: box.y - offset - node.height - NODE_GAP, bottom: box.y + box.height - offset + NODE_GAP });
        }
      }
    }
    const bands: typeof blocked = [];
    for (const band of blocked.sort((a, b) => a.top - b.top)) {
      const last = bands.at(-1);
      if (last && band.top < last.bottom) {
        last.bottom = Math.max(last.bottom, band.bottom);
      } else {
        bands.push(band);
      }
    }
    const conflict = bands.find((band) => band.top < preferred && preferred < band.bottom);
    const base = conflict ? (preferred - conflict.top < conflict.bottom - preferred ? conflict.top : conflict.bottom) : preferred;
    for (const { node, offset } of members) {
      const point = positions.get(node.id)!;
      point.y = base + offset;
      placed.push({ ...point, width: node.width, height: node.height });
    }
  }
  // 连续路径定好后，将分支末端或汇合点移到它们各条入线的中间。
  for (const column of columns) {
    const targets = column.filter((node) => !next.has(node.id) && !joined.has(node.id) && inputLinks.get(node.id)!.length);
    const centers = targets.map((node) => {
      const links = inputLinks.get(node.id)!;
      const total = links.reduce((sum, link) => sum + positions.get(link.source)!.y
        + units.get(link.source)!.outputs[link.sourcePort].y - node.inputs[link.targetPort].y, 0);
      return { total, weight: links.length };
    });
    placeColumn(targets, positions, centers);
  }
}
