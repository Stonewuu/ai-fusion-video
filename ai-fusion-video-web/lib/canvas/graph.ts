import type {
  CanvasCapability,
  CanvasChanges,
  CanvasConfig,
  CanvasGraph,
  CanvasNode,
  CanvasOperation,
  CanvasPort,
  CanvasValueType,
} from "./types";

// Server versions are the identity boundary for graph content and layout.
// Keep unchanged references when an acknowledgement or stream reconnect brings
// a fresh snapshot, so saving one node does not rebuild every node's content.
export function reconcileCanvasGraph(
  current: CanvasGraph,
  incoming: CanvasGraph,
): CanvasGraph {
  const previousNodes = new Map(current.nodes.map((node) => [node.id, node]));
  const previousEdges = new Map(current.edges.map((edge) => [edge.id, edge]));
  const nodes = incoming.nodes.map((node) => {
    const previous = previousNodes.get(node.id);
    if (
      !previous ||
      previous.kind !== node.kind ||
      previous.schemaVersion !== node.schemaVersion
    )
      return node;
    if (
      previous.configVersion === node.configVersion &&
      previous.layoutVersion === node.layoutVersion
    )
      return previous;
    return {
      ...node,
      ...(previous.configVersion === node.configVersion
        ? { config: previous.config, resource: previous.resource }
        : {}),
      ...(previous.layoutVersion === node.layoutVersion
        ? { layout: previous.layout }
        : {}),
    };
  });
  const edges = incoming.edges.map((edge) => {
    const previous = previousEdges.get(edge.id);
    return previous?.version === edge.version ? previous : edge;
  });
  const sameNodes =
    nodes.length === current.nodes.length &&
    nodes.every((node, index) => node === current.nodes[index]);
  const sameEdges =
    edges.length === current.edges.length &&
    edges.every((edge, index) => edge === current.edges[index]);
  if (
    sameNodes &&
    sameEdges &&
    current.schemaVersion === incoming.schemaVersion &&
    current.projectId === incoming.projectId
  )
    return current;
  return {
    ...incoming,
    nodes: sameNodes ? current.nodes : nodes,
    edges: sameEdges ? current.edges : edges,
  };
}

export function applyCanvasChanges(
  graph: CanvasGraph,
  changes: CanvasChanges,
): CanvasGraph {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const edges = new Map(graph.edges.map((edge) => [edge.id, edge]));
  changes.deletedNodeIds.forEach((id) => nodes.delete(id));
  changes.deletedEdgeIds.forEach((id) => edges.delete(id));
  changes.upsertedNodes.forEach((node) => nodes.set(node.id, node));
  changes.upsertedEdges.forEach((edge) => edges.set(edge.id, edge));
  return reconcileCanvasGraph(graph, {
    ...graph,
    nodes: [...nodes.values()],
    edges: [...edges.values()],
  });
}

export function portType(
  port: CanvasPort,
  config: CanvasConfig,
): CanvasValueType | undefined {
  const types = portTypes(port, config);
  return types.length === 1 ? types[0] : undefined;
}

export function portTypes(
  port: CanvasPort,
  config: CanvasConfig,
): CanvasValueType[] {
  if (port.valueType) return [port.valueType];
  if (port.valueTypeField) {
    const value = config[port.valueTypeField];
    return typeof value === "string" && value ? [value as CanvasValueType] : [];
  }
  return port.acceptedTypes;
}

export interface CompatibleNode {
  capability: CanvasCapability;
  port: CanvasPort;
  config: CanvasConfig;
}
export function compatibleNodes(
  source: CanvasNode,
  output: CanvasPort,
  catalog: CanvasCapability[],
  projectId: number | null,
): CompatibleNode[] {
  const type = portType(output, source.config);
  if (!type) return [];
  return catalog.flatMap((capability) => {
    if (capability.requiresProject && projectId === null) return [];
    return capability.inputs.flatMap((port) => {
      const allowed = port.valueTypeField
        ? capability.configFields[port.valueTypeField]?.enumValues
        : undefined;
      if (
        (output.collection && !port.collection) ||
        (port.valueType && port.valueType !== type) ||
        (!port.valueType &&
          !port.valueTypeField &&
          !port.acceptedTypes.includes(type)) ||
        (allowed?.length && !allowed.includes(type))
      )
        return [];
      const config = initialConfig(capability);
      if (port.valueTypeField) config[port.valueTypeField] = type;
      return [{ capability, port, config }];
    });
  });
}

export function initialConfig(capability: CanvasCapability): CanvasConfig {
  if (capability.kind === "foreach") return { maxItems: 10 };
  return Object.fromEntries(
    Object.entries(capability.configFields)
      .filter(([, field]) => field.required && field.enumValues.length === 1)
      .map(([key, field]) => [key, field.enumValues[0]]),
  );
}

export function validateConnection(
  graph: CanvasGraph,
  catalog: CanvasCapability[],
  sourceId: string,
  sourcePort: string,
  targetId: string,
  targetPort: string,
): string | null {
  if (sourceId === targetId) return "节点不能连接自身";
  const source = graph.nodes.find((node) => node.id === sourceId);
  const target = graph.nodes.find((node) => node.id === targetId);
  const output = catalog
    .find((item) => item.kind === source?.kind)
    ?.outputs.find((port) => port.id === sourcePort);
  const input = catalog
    .find((item) => item.kind === target?.kind)
    ?.inputs.find((port) => port.id === targetPort);
  if (!source || !target || !output || !input)
    return "连接的节点或端口已不存在";
  const outputType = portType(output, source.config);
  const inputTypes = portTypes(input, target.config);
  if (!outputType || !inputTypes.length)
    return "请先设置节点的内容类型，再连接对应输入和输出";
  if (
    !inputTypes.includes(outputType) ||
    (output.collection && !input.collection)
  )
    return "端口类型不兼容；集合请通过候选选择节点处理";
  if (Object.hasOwn(target.config, input.id))
    return "该输入已有手填值，请先明确移除手填值";
  if (
    graph.edges.filter(
      (edge) =>
        edge.kind === "execution" &&
        edge.targetNodeId === targetId &&
        edge.targetPort === targetPort,
    ).length >= input.maxConnections
  )
    return "该输入已达到连接数量上限";
  const visited = new Set<string>();
  const remaining = [targetId];
  while (remaining.length) {
    const nodeId = remaining.pop()!;
    if (nodeId === sourceId) return "这条连线会形成执行循环";
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    graph.edges
      .filter(
        (edge) => edge.kind === "execution" && edge.sourceNodeId === nodeId,
      )
      .forEach((edge) => remaining.push(edge.targetNodeId));
  }
  return null;
}

export function worldPosition(
  node: CanvasNode,
  nodes: CanvasNode[],
): { x: number; y: number } {
  let x = node.layout.x;
  let y = node.layout.y;
  let parentId = node.layout.parentId;
  const visited = new Set([node.id]);
  const index = new Map(nodes.map((item) => [item.id, item]));
  while (parentId) {
    if (visited.has(parentId)) throw new Error("分组关系存在循环");
    visited.add(parentId);
    const parent = index.get(parentId);
    if (!parent) throw new Error("分组不存在");
    x += parent.layout.x;
    y += parent.layout.y;
    parentId = parent.layout.parentId;
  }
  return { x, y };
}

export function copyNodes(
  graph: CanvasGraph,
  selected: string[],
  offset = { x: 48, y: 48 },
): CanvasOperation[] {
  const included = new Set(selected);
  let size = 0;
  while (size !== included.size) {
    size = included.size;
    graph.nodes
      .filter(
        (node) => node.layout.parentId && included.has(node.layout.parentId),
      )
      .forEach((node) => included.add(node.id));
  }
  const mapping = new Map([...included].map((id) => [id, crypto.randomUUID()]));
  const nodes = graph.nodes.filter((node) => included.has(node.id));
  const operations: CanvasOperation[] = nodes.map((node) => {
    const nested = !!node.layout.parentId && included.has(node.layout.parentId);
    const position = nested ? node.layout : worldPosition(node, graph.nodes);
    return {
      type: "canvas.node.create",
      node: {
        ...node,
        id: mapping.get(node.id)!,
        config: Object.fromEntries(
          Object.entries(node.config).filter(
            ([key]) => node.kind !== "selection" || key !== "selectedOutputId",
          ),
        ),
        configVersion: 0,
        layoutVersion: 0,
        layout: {
          ...node.layout,
          parentId: nested ? mapping.get(node.layout.parentId!)! : null,
          x: position.x + (nested ? 0 : offset.x),
          y: position.y + (nested ? 0 : offset.y),
        },
      },
    };
  });
  graph.edges
    .filter(
      (edge) =>
        included.has(edge.sourceNodeId) && included.has(edge.targetNodeId),
    )
    .forEach((edge) =>
      operations.push({
        type: "canvas.edge.create",
        edge: {
          ...edge,
          id: crypto.randomUUID(),
          sourceNodeId: mapping.get(edge.sourceNodeId)!,
          targetNodeId: mapping.get(edge.targetNodeId)!,
          version: 0,
          output: null,
        },
      }),
    );
  return operations;
}
