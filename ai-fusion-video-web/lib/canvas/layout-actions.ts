import type { CanvasGraph, CanvasNode, CanvasOperation } from "./types";
import { worldPosition } from "./graph";
import { arrangeCanvasPaths, type CanvasGeometry } from "./auto-layout";

export function deleteCanvasSelection(
  graph: CanvasGraph,
  nodeIds: string[],
  edgeIds: string[],
): CanvasOperation[] {
  const nodes = new Set(nodeIds);
  return [
    ...graph.edges
      .filter(
        (edge) =>
          edgeIds.includes(edge.id) &&
          !nodes.has(edge.sourceNodeId) &&
          !nodes.has(edge.targetNodeId),
      )
      .map((edge) => ({
        type: "canvas.edge.delete" as const,
        edgeId: edge.id,
        expectedVersion: edge.version,
      })),
    ...graph.nodes
      .filter((node) => nodes.has(node.id))
      .sort((a, b) => depth(b, graph) - depth(a, graph))
      .map((node) => ({
        type: "canvas.node.delete" as const,
        nodeId: node.id,
        expectedConfigVersion: node.configVersion,
        expectedLayoutVersion: node.layoutVersion,
      })),
  ];
}

function depth(node: CanvasNode, graph: CanvasGraph) {
  let result = 0;
  let parent = node.layout.parentId;
  while (parent) {
    result++;
    parent =
      graph.nodes.find((item) => item.id === parent)?.layout.parentId ?? null;
  }
  return result;
}
function hasSelectedParent(
  node: CanvasNode,
  ids: Set<string>,
  graph: CanvasGraph,
) {
  let parent = node.layout.parentId;
  while (parent) {
    if (ids.has(parent)) return true;
    parent =
      graph.nodes.find((item) => item.id === parent)?.layout.parentId ?? null;
  }
  return false;
}
export function groupCanvasNodes(
  graph: CanvasGraph,
  selection: string[],
): CanvasOperation[] {
  const selected = new Set(selection);
  const nodes = graph.nodes.filter(
    (node) =>
      selected.has(node.id) && !hasSelectedParent(node, selected, graph),
  );
  if (!nodes.length) throw new Error("请选择要分组的节点");
  if (nodes.some((node) => node.layout.locked))
    throw new Error("选区包含锁定节点，请先解锁");
  const positions = nodes.map((node) => worldPosition(node, graph.nodes));
  const x = Math.min(...positions.map((point) => point.x)) - 32;
  const y = Math.min(...positions.map((point) => point.y)) - 64;
  const width =
    Math.max(
      ...positions.map(
        (point, index) => point.x + (nodes[index].layout.width ?? 300),
      ),
    ) -
    x +
    32;
  const height =
    Math.max(
      ...positions.map(
        (point, index) => point.y + (nodes[index].layout.height ?? 280),
      ),
    ) -
    y +
    32;
  const group: CanvasNode = {
    id: crypto.randomUUID(),
    kind: "group",
    schemaVersion: 1,
    title: "新分组",
    config: {},
    resource: null,
    configVersion: 0,
    layoutVersion: 0,
    layout: {
      x,
      y,
      width,
      height,
      parentId: null,
      locked: false,
      collapsed: false,
    },
  };
  return [
    { type: "canvas.node.create", node: group },
    ...nodes.map((node, index) => ({
      type: "canvas.node.layout" as const,
      nodeId: node.id,
      expectedLayoutVersion: node.layoutVersion,
      layout: {
        ...node.layout,
        parentId: group.id,
        x: positions[index].x - x,
        y: positions[index].y - y,
      },
    })),
  ];
}

export type CanvasAlignment =
  | "left"
  | "top"
  | "horizontal"
  | "vertical"
  | "arrange";
export function arrangeCanvasNodes(
  graph: CanvasGraph,
  selection: string[],
  mode: CanvasAlignment,
  geometry: CanvasGeometry = {},
): CanvasOperation[] {
  const selected = new Set(
    selection.length ? selection : graph.nodes.map((node) => node.id),
  );
  const nodes = graph.nodes.filter(
    (node) =>
      selected.has(node.id) &&
      !node.layout.locked &&
      !hasSelectedParent(node, selected, graph),
  );
  if (!nodes.length) throw new Error("没有可移动的节点");
  const positions = new Map(
    nodes.map((node) => [node.id, worldPosition(node, graph.nodes)]),
  );
  const left = Math.min(...[...positions.values()].map((point) => point.x));
  const top = Math.min(...[...positions.values()].map((point) => point.y));
  const order = [...nodes].sort((a, b) =>
    mode === "vertical"
      ? positions.get(a.id)!.y - positions.get(b.id)!.y
      : positions.get(a.id)!.x - positions.get(b.id)!.x,
  );
  const arranged =
    mode === "arrange" ? arrangeCanvasPaths(graph, nodes, geometry) : null;
  let cursor = 0;
  return order.map((node) => {
    const original = positions.get(node.id)!;
    let x = original.x;
    let y = original.y;
    if (mode === "left") x = left;
    if (mode === "top") y = top;
    if (mode === "horizontal") {
      x = left + cursor;
      y = top;
      cursor += (geometry[node.id]?.width ?? node.layout.width ?? 300) + 64;
    }
    if (mode === "vertical") {
      x = left;
      y = top + cursor;
      cursor += (geometry[node.id]?.height ?? node.layout.height ?? 280) + 64;
    }
    if (mode === "arrange") {
      ({ x, y } = arranged!.get(node.id)!);
    }
    const parent = node.layout.parentId
      ? graph.nodes.find((item) => item.id === node.layout.parentId)
      : undefined;
    const offset = parent ? worldPosition(parent, graph.nodes) : { x: 0, y: 0 };
    return {
      type: "canvas.node.layout",
      nodeId: node.id,
      expectedLayoutVersion: node.layoutVersion,
      layout: { ...node.layout, x: x - offset.x, y: y - offset.y },
    };
  });
}

export function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    !!target.closest(
      "input,textarea,select,[contenteditable='true'],[role='textbox'],[role='combobox'],video,audio",
    )
  );
}
