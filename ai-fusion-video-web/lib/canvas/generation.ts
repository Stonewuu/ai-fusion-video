import type { CanvasNodeActivity, CanvasRunValue } from "@/lib/api/canvas-runs";
import type { CanvasCatalog, CanvasGraph, CanvasNode } from "./types";

export function isCanvasGenerationNode(node: CanvasNode) {
  return node.kind === "imageGeneration" || node.kind === "videoGeneration";
}

/** 生成表单中随节点类型变化的文案。 */
export function generationComposerCopy(kind: string, promptConnected: boolean) {
  if (kind === "textGeneration") {
    return {
      name: "文本快捷生成",
      action: "生成文本",
      promptKey: promptConnected ? "instruction" : "prompt",
      promptLabel: promptConnected ? "写作要求" : "写作内容",
      placeholder: promptConnected
        ? "写下要求，例如：扩写成 300 字的场景描写"
        : "描述你想写的内容…",
    };
  }
  const image = kind === "imageGeneration";
  return {
    name: image ? "图片快捷生成" : "视频快捷生成",
    action: image ? "生成图片" : "生成视频",
    promptKey: promptConnected ? "additionalPrompt" : "prompt",
    promptLabel: promptConnected ? "补充要求" : "提示词",
    placeholder: promptConnected ? "补充本次画面描述…" : "描述你想生成的画面…",
  };
}

export interface CanvasGenerationInputs {
  documentRevision: number;
  prompt: string;
  inputs: {
    edgeId: string;
    nodeId: string;
    title: string;
    port: string;
    values: CanvasRunValue[];
  }[];
  issues: { nodeId: string; title: string; message: string }[];
}

/** 快捷生成只复用可用产物；未完成的生成步骤需要用户先处理。 */
export function missingGenerationSources(
  graph: CanvasGraph,
  catalog: CanvasCatalog,
  nodeId: string,
  activities: CanvasNodeActivity[],
) {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const capabilities = new Map(catalog.nodes.map((item) => [item.kind, item]));
  const ready = new Set(
    activities
      .filter(
        ({ node }) =>
          node.status === "SUCCEEDED" &&
          node.derivedState === "CURRENT" &&
          !node.contentUnavailable,
      )
      .map(({ node }) => node.nodeId),
  );
  const visited = new Set<string>([nodeId]);
  const missing: CanvasNode[] = [];
  const visit = (target: string) => {
    for (const edge of graph.edges) {
      if (
        edge.kind !== "execution" ||
        edge.targetNodeId !== target ||
        edge.output ||
        visited.has(edge.sourceNodeId)
      ) {
        continue;
      }
      visited.add(edge.sourceNodeId);
      const source = nodes.get(edge.sourceNodeId)!;
      const executor = capabilities.get(source.kind)!.executor;
      if (["constant", "resource"].includes(executor)) {
        continue;
      }
      if (ready.has(source.id)) {
        continue;
      }
      if (source.kind === "textCombine" || source.kind === "collection") {
        visit(source.id);
      } else {
        missing.push(source);
      }
    }
  };
  visit(nodeId);
  return missing;
}

/** 所有节点均支持点击后在节点下方打开悬浮面板，不再弹出右侧属性栏。 */
export function hasCanvasQuickPanel(node: CanvasNode) {
  return Boolean(node.kind);
}

export type CanvasNodePanelKind = "generation" | "text" | "media" | "config";

const textPanelKinds = new Set(["text", "story", "note"]);
const mediaPanelKinds = new Set([
  "image",
  "video",
  "audio",
  "textFile",
  "document",
]);

/** 根据节点类型分派对应的节点下方悬浮表单形态。 */
export function canvasNodePanelKind(node: CanvasNode): CanvasNodePanelKind {
  if (hasCanvasGenerationComposer(node)) {
    return "generation";
  }
  if (textPanelKinds.has(node.kind)) {
    return "text";
  }
  if (mediaPanelKinds.has(node.kind)) {
    return "media";
  }
  return "config";
}

/** 使用生成表单的节点：生图、生视频以画面为主体，生成文本沿用普通节点外观。 */
export function hasCanvasGenerationComposer(node: CanvasNode) {
  return isCanvasGenerationNode(node) || node.kind === "textGeneration";
}

/**
 * 生成文本的结果可写回为其提供写作素材的文本或故事节点正文。只接受当前有效的结果，
 * 写回后输入变化、结果转为待更新，避免同一结果被重复写入或覆盖更新后的正文。
 */
export function textGenerationWriteback(
  graph: CanvasGraph,
  nodeId: string,
  activity: CanvasNodeActivity | undefined,
) {
  const node = activity?.node;
  if (
    !node ||
    node.status !== "SUCCEEDED" ||
    node.derivedState !== "CURRENT" ||
    node.contentUnavailable
  ) {
    return null;
  }
  const text = node.outputs.values.text?.[0]?.text;
  if (!text?.trim()) {
    return null;
  }
  const edge = graph.edges.find(
    (item) =>
      item.kind === "execution" &&
      item.targetNodeId === nodeId &&
      item.targetPort === "prompt",
  );
  const target = edge
    ? graph.nodes.find((item) => item.id === edge.sourceNodeId)
    : undefined;
  return target?.kind === "text" || target?.kind === "story"
    ? { target, text }
    : null;
}
