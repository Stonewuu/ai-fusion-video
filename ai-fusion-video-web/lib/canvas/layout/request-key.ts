import type { LayoutRequest } from "./types";

export function layoutRequestKey(request: LayoutRequest) {
  return JSON.stringify([
    request.graph.nodes.map((node) => [node.id, node.layout]),
    request.graph.edges.map((edge) => [edge.id, edge.kind, edge.sourceNodeId, edge.sourcePort, edge.targetNodeId, edge.targetPort, edge.inputIndex]),
    request.geometry, request.relations, request.hidden, request.labels,
  ]);
}
