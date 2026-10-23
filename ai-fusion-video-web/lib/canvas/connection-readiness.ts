import { canvasAgentKinds } from "./agent-types";
import { canvasPortLabels } from "./labels";
import { inputCountsForNode, modelForVersion, modelInputs, modelKind } from "./model-constraints";
import type { CanvasCatalog, CanvasEdge, CanvasGraph } from "./types";

const imagePorts = new Set(["firstFrame", "lastFrame", "referenceImages"]);
const agentMediaTypes: Record<string, string> = {
  referenceImages: "image", referenceVideos: "video", referenceAudios: "audio", referenceDocuments: "file",
};

export function canvasConnectionIssue(edge: CanvasEdge, graph: CanvasGraph, catalog: CanvasCatalog): string | null {
  if (edge.kind !== "execution" || !edge.targetPort) {
    return null;
  }
  const target = graph.nodes.find((node) => node.id === edge.targetNodeId);
  if (!target?.config.modelId) {
    return null;
  }
  const kind = modelKind(target.kind);
  const agent = canvasAgentKinds.includes(target.kind);
  if (!kind && !agent) {
    return null;
  }
  const selected = catalog.models.find((item) => item.id === target.config.modelId);
  if (!selected) {
    return "所选模型已不可用，请选择其他模型";
  }
  const model = modelForVersion(selected, target.config.workflowVersionId);
  if (model.unavailable) {
    return model.unavailable;
  }
  const name = canvasPortLabels[edge.targetPort] ?? edge.targetPort;
  if (agent) {
    const type = agentMediaTypes[edge.targetPort];
    const supported = model.parameters.multimodalInputTypes;
    return type && (!Array.isArray(supported) || !supported.includes(type)) ? `当前模型不支持${name}，请更换模型` : null;
  }
  const inputs = modelInputs(model, target.config);
  if (!inputs) {
    return null;
  }
  const rule = inputs[edge.targetPort];
  if (!rule || rule.maximum === 0) {
    return `当前模型不支持${name}，请更换模型`;
  }
  const counts = inputCountsForNode(target, graph, catalog);
  for (const port of catalog.nodes.find((item) => item.kind === target.kind)!.inputs) {
    if (Object.hasOwn(target.config, port.id)) {
      counts[port.id] = 1;
    }
  }
  const count = counts[edge.targetPort] ?? 0;
  if (rule.maximum !== null && count > rule.maximum) {
    return `${name}最多允许 ${rule.maximum} 项，当前至少 ${count} 项`;
  }
  if (imagePorts.has(edge.targetPort)) {
    const images = [...imagePorts].reduce((sum, port) => sum + (counts[port] ?? 0), 0);
    if (typeof model.parameters.maxImageInputs === "number" && images > model.parameters.maxImageInputs) {
      return "图片输入总量超过当前模型限制";
    }
    if (model.parameters.exclusiveInputModes && ("firstFrame" in counts || "lastFrame" in counts)
      && Object.keys(counts).some((port) => port.startsWith("reference"))) {
      return "当前模型的首尾帧输入与参考素材不能同时使用";
    }
  }
  return null;
}
