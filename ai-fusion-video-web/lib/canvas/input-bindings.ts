import { portType, portTypes, validateConnection } from "./graph";
import type {
  CanvasCapability,
  CanvasGraph,
  CanvasNode,
  CanvasOperation,
  CanvasPort,
  CanvasValueType,
} from "./types";

export const canvasInputPurposes: Record<string, string> = {
  prompt: "描述这一步要完成的创作。",
  context:
    "使用文字、角色与场景设定；需要参考画面时，请连接对应的图片或视频输入。",
  firstFrame: "作为视频的起始画面。",
  lastFrame: "作为视频的结束画面。",
  referenceImages: "将原图送入模型，作为主体、外观或画面参考。",
  referenceVideos: "将视频送入模型，作为动作或画面参考。",
  referenceAudios: "将音频送入模型，作为声音参考。",
  images: "将图片送入模型进行观察。",
};

export interface CanvasInputCandidate {
  node: CanvasNode;
  port: CanvasPort;
  type: CanvasValueType;
  issue: string | null;
}

function connectionGraph(
  graph: CanvasGraph,
  target: CanvasNode,
  input: CanvasPort,
) {
  const config = { ...target.config };
  delete config[input.id];
  return {
    ...graph,
    nodes: graph.nodes.map((node) =>
      node.id === target.id ? { ...target, config } : node,
    ),
    edges: input.collection
      ? graph.edges
      : graph.edges.filter(
          (edge) =>
            edge.kind !== "execution" ||
            edge.targetNodeId !== target.id ||
            edge.targetPort !== input.id,
        ),
  };
}

export function canvasInputCandidates(
  graph: CanvasGraph,
  catalog: CanvasCapability[],
  target: CanvasNode,
  input: CanvasPort,
): CanvasInputCandidate[] {
  const types = portTypes(input, target.config);
  const checked = connectionGraph(graph, target, input);
  const byKind = new Map(
    catalog.map((capability) => [capability.kind, capability]),
  );
  const outgoing = new Map<string, string[]>();
  let incomingCount = 0;
  for (const edge of checked.edges) {
    if (edge.kind !== "execution") continue;
    if (edge.targetNodeId === target.id && edge.targetPort === input.id)
      incomingCount++;
    const adjacent = outgoing.get(edge.sourceNodeId) ?? [];
    adjacent.push(edge.targetNodeId);
    outgoing.set(edge.sourceNodeId, adjacent);
  }
  const descendants = new Set<string>();
  const queue = [target.id];
  while (queue.length) {
    const id = queue.pop()!;
    if (descendants.has(id)) continue;
    descendants.add(id);
    queue.push(...(outgoing.get(id) ?? []));
  }
  return graph.nodes
    .filter((node) => node.id !== target.id)
    .flatMap((node) => {
      const capability = byKind.get(node.kind);
      if (!capability) return [];
      return capability.outputs.flatMap((port) => {
        const type = portType(port, node.config);
        if (
          !type ||
          !types.includes(type) ||
          (port.collection && !input.collection)
        )
          return [];
        const issue = descendants.has(node.id)
          ? "这条连线会形成执行循环"
          : incomingCount >= input.maxConnections
            ? "该输入已达到连接数量上限"
            : null;
        return [{ node, port, type, issue }];
      });
    });
}

export function connectCanvasInput(
  graph: CanvasGraph,
  catalog: CanvasCapability[],
  targetId: string,
  inputId: string,
  sourceId: string,
  outputId: string,
  edgeId: string,
): CanvasOperation[] {
  const target = graph.nodes.find((node) => node.id === targetId);
  const source = graph.nodes.find((node) => node.id === sourceId);
  const input = catalog
    .find((capability) => capability.kind === target?.kind)
    ?.inputs.find((port) => port.id === inputId);
  if (!source || !target || !input)
    throw new Error("输入来源已变化，请重新选择");
  const checked = connectionGraph(graph, target, input);
  const issue = validateConnection(
    checked,
    catalog,
    sourceId,
    outputId,
    targetId,
    inputId,
  );
  if (issue) throw new Error(issue);
  const previous = !input.collection
    ? graph.edges.find(
        (edge) =>
          edge.kind === "execution" &&
          edge.targetNodeId === targetId &&
          edge.targetPort === inputId,
      )
    : undefined;
  const occupied = new Set(
    checked.edges
      .filter(
        (edge) =>
          edge.kind === "execution" &&
          edge.targetNodeId === targetId &&
          edge.targetPort === inputId,
      )
      .map((edge) => edge.inputIndex),
  );
  let index = 0;
  while (occupied.has(index)) index++;
  const edge = {
    id: previous?.id ?? edgeId,
    kind: "execution" as const,
    sourceNodeId: sourceId,
    sourcePort: outputId,
    targetNodeId: targetId,
    targetPort: inputId,
    inputIndex: index,
    output: null,
    version: previous?.version ?? 0,
  };
  const operations: CanvasOperation[] = [
    {
      type: "canvas.node.assert",
      nodeId: source.id,
      expectedConfigVersion: source.configVersion,
    },
    {
      type: "canvas.node.assert",
      nodeId: target.id,
      expectedConfigVersion: target.configVersion,
    },
  ];
  if (Object.hasOwn(target.config, inputId))
    operations.push({
      type: "canvas.node.patch",
      nodeId: target.id,
      expectedConfigVersion: target.configVersion,
      configPatch: { [inputId]: null },
    });
  operations.push(
    previous
      ? { type: "canvas.edge.replace", edge, expectedVersion: previous.version }
      : { type: "canvas.edge.create", edge },
  );
  return operations;
}
