import { validateCanvasConfig } from "./config";
import { validateModelNode } from "./model-constraints";
import { canvasMediaKinds } from "./placement";
import { canvasAgentKinds } from "./agent-types";
import type {
  CanvasCapability,
  CanvasCatalog,
  CanvasGraph,
  CanvasNode,
} from "./types";

export function canvasNodeConfigurationIssue(
  node: CanvasNode,
  capability: CanvasCapability,
  catalog: CanvasCatalog,
  graph: CanvasGraph,
): string | null {
  if (capability.resourceType && !node.resource)
    return `请选择关联的${capability.label}`;
  if (node.kind in canvasMediaKinds && !node.config.resourceId)
    return `请选择${capability.label}文件`;
  if (
    (node.kind === "text" || node.kind === "story") &&
    !String(node.config.content ?? "").trim()
  )
    return `请填写${capability.label}内容`;
  if (capability.configFields.modelId && !node.config.modelId)
    return "请选择模型";
  if (node.kind === "mcpTool" && !node.config.contractId)
    return "请选择要调用的外部工具";
  if (node.kind === "writeback" && !node.config.binding)
    return "请选择要写入的业务字段";
  const fields = validateCanvasConfig(capability, node.config);
  if (fields) return fields;
  if (canvasAgentKinds.includes(node.kind)) {
    const profile = catalog.agentProfiles.find((item) => item.type === node.config.agentType);
    const custom = node.config.skillMode === "CUSTOM";
    const names = custom && Array.isArray(node.config.skillNames) ? node.config.skillNames : profile?.defaultSkills ?? [];
    if (!custom && Array.isArray(node.config.skillNames) && node.config.skillNames.length) return "请将 Skill 使用方式改为单独指定";
    const missing = names.find((name) => !catalog.skills.some((skill) => skill.name === name));
    if (missing) return `Skill ${missing} 已不可用，请重新选择`;
  }
  const model = validateModelNode(node, graph, catalog);
  if (model) return model;
  for (const port of capability.inputs) {
    const fromContext =
      port.id === "prompt" &&
      graph.edges.some(
        (edge) =>
          edge.kind === "execution" &&
          edge.targetNodeId === node.id &&
          edge.targetPort === "context",
      );
    if (
      port.required &&
      !fromContext &&
      !String(node.config[port.id] ?? "").trim() &&
      !graph.edges.some(
        (edge) =>
          edge.kind === "execution" &&
          edge.targetNodeId === node.id &&
          edge.targetPort === port.id,
      )
    )
      return `请填写或连接${port.label}`;
  }
  return null;
}
