import type { LayoutRelation, LayoutRequest } from "./types";

export function layoutRelationsKey(relations: LayoutRelation[]) {
  return JSON.stringify(relations
    .map(({ id, source, target, label }) => [id, source, target, label])
    .sort(([left], [right]) => left.localeCompare(right)));
}

export function layoutRequestKey(request: LayoutRequest) {
  return JSON.stringify([
    request.graph.nodes.map((node) => [node.id, node.layout]),
    request.graph.edges.map((edge) => [edge.id, edge.kind, edge.sourceNodeId, edge.sourcePort, edge.targetNodeId, edge.targetPort, edge.inputIndex]),
    request.geometry, request.relations, request.hidden, request.labels,
  ]);
}
