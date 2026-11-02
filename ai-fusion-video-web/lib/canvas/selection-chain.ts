interface CanvasChainEdge {
  id: string;
  source: string;
  target: string;
}

export function canvasSelectionChain(
  edges: readonly CanvasChainEdge[],
  selectedNodeIds: readonly string[],
) {
  const nodeIds = new Set(selectedNodeIds);
  const edgeIds = new Set<string>();
  const incoming = new Map<string, CanvasChainEdge[]>();
  const outgoing = new Map<string, CanvasChainEdge[]>();

  for (const edge of edges) {
    const parents = incoming.get(edge.target);
    if (parents) {
      parents.push(edge);
    } else {
      incoming.set(edge.target, [edge]);
    }
    const children = outgoing.get(edge.source);
    if (children) {
      children.push(edge);
    } else {
      outgoing.set(edge.source, [edge]);
    }
  }

  const traverse = (
    adjacency: Map<string, CanvasChainEdge[]>,
    nextNode: (edge: CanvasChainEdge) => string,
  ) => {
    const visited = new Set(selectedNodeIds);
    const pending = [...visited];
    for (let index = 0; index < pending.length; index++) {
      const connections = adjacency.get(pending[index]);
      if (!connections) {
        continue;
      }
      for (const edge of connections) {
        const next = nextNode(edge);
        edgeIds.add(edge.id);
        nodeIds.add(next);
        if (!visited.has(next)) {
          visited.add(next);
          pending.push(next);
        }
      }
    }
  };

  // 两个方向分别从选中节点出发，避免从父节点再向下扩展到兄弟分支。
  traverse(incoming, (edge) => edge.source);
  traverse(outgoing, (edge) => edge.target);
  return { nodeIds, edgeIds };
}
