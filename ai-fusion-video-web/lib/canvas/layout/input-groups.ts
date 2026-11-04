import type { LayoutUnit } from "./types";

export interface InputDestination {
  target: string;
  portY: number;
  index: number;
}

export function groupParallelInputs(nodes: LayoutUnit[], destinations: Map<string, InputDestination>) {
  const groups = new Map<string, LayoutUnit[]>();
  for (const node of nodes) {
    const input = destinations.get(node.id);
    const key = input ? `target:${input.target}` : `node:${node.id}`;
    const group = groups.get(key);
    if (group) {
      group.push(node);
    } else {
      groups.set(key, [node]);
    }
  }
  for (const group of groups.values()) {
    group.sort((a, b) => {
      const left = destinations.get(a.id)!;
      const right = destinations.get(b.id)!;
      return left.portY - right.portY || left.index - right.index || a.id.localeCompare(b.id);
    });
  }
  return [...groups.values()];
}
