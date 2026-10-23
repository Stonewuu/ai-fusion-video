import { copyNodes } from "./graph";
import type { CanvasGraph } from "./types";

export const CANVAS_CLIPBOARD = "application/x-fusion-canvas";
export function encodeCanvasClipboard(graph: CanvasGraph, ids: string[]) {
  const operations = copyNodes(graph, ids, { x: 0, y: 0 });
  if (operations.length > 500) throw new Error("一次最多复制 500 个节点与连线");
  const fragment: CanvasGraph = {
    ...graph,
    nodes: operations.flatMap((operation) =>
      operation.type === "canvas.node.create" ? [operation.node] : [],
    ),
    edges: operations.flatMap((operation) =>
      operation.type === "canvas.edge.create" ? [operation.edge] : [],
    ),
  };
  return JSON.stringify({ format: "afv-canvas-1", graph: fragment });
}

export function decodeCanvasClipboard(value: string): CanvasGraph {
  if (value.length > 8 * 1024 * 1024)
    throw new Error("复制内容超过 8 MiB，请缩小选择范围");
  const parsed = JSON.parse(value);
  const graph = parsed?.graph;
  if (
    parsed?.format !== "afv-canvas-1" ||
    !graph ||
    graph.schemaVersion !== 1 ||
    !Array.isArray(graph.nodes) ||
    !Array.isArray(graph.edges) ||
    graph.nodes.length + graph.edges.length > 500 ||
    (graph.projectId !== null && !Number.isSafeInteger(graph.projectId))
  )
    throw new Error("剪贴板中的画布片段无效");
  for (const node of graph.nodes) {
    if (
      !node ||
      typeof node.id !== "string" ||
      typeof node.kind !== "string" ||
      typeof node.title !== "string" ||
      !node.layout ||
      !Number.isFinite(node.layout.x) ||
      !Number.isFinite(node.layout.y) ||
      !node.config ||
      typeof node.config !== "object" ||
      Array.isArray(node.config)
    )
      throw new Error("剪贴板节点格式无效");
  }
  for (const edge of graph.edges)
    if (
      !edge ||
      typeof edge.id !== "string" ||
      typeof edge.sourceNodeId !== "string" ||
      typeof edge.targetNodeId !== "string"
    )
      throw new Error("剪贴板连线格式无效");
  return graph;
}
