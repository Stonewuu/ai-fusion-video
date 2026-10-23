import type {
  CanvasCatalog,
  CanvasConfig,
  CanvasGraph,
  CanvasModelOption,
  CanvasModelInput,
  CanvasNode,
  CanvasNodeKind,
  CanvasValueType,
} from "./types";
import { canvasFieldLabels, canvasPortLabels } from "./labels";
import { validateCanvasConfig } from "./config";
import { portType, portTypes } from "./graph";
import {
  automaticVideoDuration,
  videoInputState,
  videoRatioOptions,
  videoRulesOf,
} from "../video-model-rules";

const modelKinds: Partial<Record<CanvasNodeKind, string>> = {
  imageGeneration: "image",
  videoGeneration: "video",
  textGeneration: "text",
  comfyui: "comfyui",
};

export function modelKind(kind: CanvasNodeKind) {
  return modelKinds[kind];
}

export function modelInputs(model: CanvasModelOption, config: CanvasConfig) {
  return model.kind === "comfyui"
    ? model.versions.find((item) => item.id === config.workflowVersionId)
        ?.inputs
    : model.inputs;
}

export function modelForVersion(model: CanvasModelOption, versionId: CanvasConfig[string] | undefined): CanvasModelOption {
  if (model.kind !== "comfyui") {
    return model;
  }
  const version = model.versions.find((item) => item.id === versionId);
  return version ? { ...model, inputs: version.inputs, parameters: version.parameters, unavailable: model.unavailable ?? version.unavailable } : model;
}

export function modelsForNode(kind: CanvasNodeKind, catalog: CanvasCatalog) {
  return catalog.models.filter(
    (model) =>
      model.kind === (modelKinds[kind] ?? "text") && !model.unavailable,
  );
}

export function modelFitsInputs(
  model: CanvasModelOption,
  inputs: Record<string, CanvasModelInput>,
  ports: string[],
  counts: Record<string, number>,
) {
  if (
    ports.some(
      (port) =>
        !inputs[port] ||
        inputs[port].maximum === 0 ||
        (inputs[port].maximum !== null &&
          (counts[port] ?? 0) > inputs[port].maximum),
    )
  ) {
    return false;
  }
  const images = ["firstFrame", "lastFrame", "referenceImages"].reduce(
    (sum, port) => sum + (counts[port] ?? 0),
    0,
  );
  if (
    typeof model.parameters.maxImageInputs === "number" &&
    images > model.parameters.maxImageInputs
  ) {
    return false;
  }
  if (
    model.parameters.exclusiveInputModes &&
    (ports.includes("firstFrame") || ports.includes("lastFrame")) &&
    ports.some((port) => port.startsWith("reference"))
  ) {
    return false;
  }
  return true;
}

export function inputCountsForNode(
  node: CanvasNode,
  graph: CanvasGraph,
  catalog: CanvasCatalog,
) {
  const counts: Record<string, number> = {};
  for (const edge of graph.edges) {
    if (
      edge.kind !== "execution" ||
      edge.targetNodeId !== node.id ||
      !edge.targetPort
    ) {
      continue;
    }
    counts[edge.targetPort] =
      (counts[edge.targetPort] ?? 0) +
      (edge.output
        ? 1
        : knownCount(graph, catalog, edge.sourceNodeId, edge.sourcePort!));
  }
  return counts;
}

export function knownCount(
  graph: CanvasGraph,
  catalog: CanvasCatalog,
  id: string,
  port: string,
  visited = new Set<string>(),
): number {
  if (visited.has(id)) {
    return 0;
  }
  visited.add(id);
  const node = graph.nodes.find((item) => item.id === id);
  if (!node) {
    return 0;
  }
  const output = catalog.nodes
    .find((item) => item.kind === node.kind)
    ?.outputs.find((item) => item.id === port);
  if (!output?.collection) {
    return 1;
  }
  if (["imageGeneration", "videoGeneration", "comfyui"].includes(node.kind)) {
    return Number(node.config.count ?? 0);
  }
  if (node.kind === "collection") {
    return graph.edges
      .filter((edge) => edge.kind === "execution" && edge.targetNodeId === id)
      .reduce(
        (count, edge) =>
          count +
          knownCount(
            graph,
            catalog,
            edge.sourceNodeId,
            edge.sourcePort!,
            new Set(visited),
          ),
        0,
      );
  }
  return 0;
}

export function validateModelNode(
  node: CanvasNode,
  graph: CanvasGraph,
  catalog: CanvasCatalog,
  bound: Record<string, number> = {},
): string | null {
  if (node.kind === "foreach") {
    const allowed = catalog.nodes.find((item) => item.kind === "foreach")!
      .configFields.operationKind.enumValues;
    const operation = catalog.nodes.find(
      (item) =>
        item.kind === node.config.operationKind && allowed.includes(item.kind),
    );
    const body = node.config.operationConfig;
    if (!operation || typeof body !== "object" || Array.isArray(body)) {
      return "请选择批量处理方式并完成配置";
    }
    const validation = validateCanvasConfig(operation, body);
    if (validation) {
      return validation;
    }
    const input = operation.inputs.find(
      (port) => port.id === node.config.inputPort,
    );
    const output = operation.outputs.find(
      (port) => port.id === node.config.outputPort,
    );
    if (
      !input ||
      !output ||
      !portTypes(input, body).includes(
        node.config.inputType as CanvasValueType,
      ) ||
      portType(output, body) !== node.config.resultType
    ) {
      return "批量操作的输入或输出类型不匹配";
    }
    for (const port of operation.inputs) {
      if (
        port.id !== input.id &&
        port.required &&
        !(port.id === "prompt" && input.id === "context") &&
        !String(body[port.id] ?? "").trim()
      ) {
        return `请填写批量操作的${canvasPortLabels[port.id] ?? port.label}`;
      }
    }
    return validateModelNode(
      { ...node, kind: operation.kind, config: body },
      { ...graph, edges: [] },
      catalog,
      { [input.id]: 1 },
    );
  }
  if (["musicGeneration", "sfxGeneration"].includes(node.kind)) {
    return catalog.audioServices.some(
      (service) => service.id === node.config.apiConfigId,
    )
      ? null
      : "请选择已启用的音频服务";
  }
  const kind = modelKinds[node.kind];
  if (!kind) {
    return null;
  }
  const selected = catalog.models.find((item) => item.id === node.config.modelId);
  if (!selected) {
    return "请选择已启用的模型；模型配置变化后可刷新列表";
  }
  const model = modelForVersion(selected, node.config.workflowVersionId);
  if (model.unavailable) {
    return model.unavailable;
  }
  if (model.kind !== kind) {
    return "模型类型与当前节点不匹配";
  }
  const inputs = modelInputs(model, node.config);
  if (!inputs) {
    return "请选择已发布的工作流版本";
  }
  if (kind === "comfyui") {
    if (model.resultType !== node.config.resultType) {
      return "工作流输出类型与模型不一致";
    }
    const version = model.versions.find(
      (item) => item.id === node.config.workflowVersionId,
    )!;
    const invalid = Object.keys(node.config).find(
      (key) =>
        ![
          "modelId",
          "workflowVersionId",
          "resultType",
          ...version.fields,
        ].includes(key),
    );
    if (invalid) {
      return `此工作流未绑定${canvasFieldLabels[invalid] ?? invalid}，请清除此参数`;
    }
  }
  const connected = new Set<string>();
  let images = 0;
  for (const port of catalog.nodes.find((item) => item.kind === node.kind)!
    .inputs) {
    const edges = graph.edges.filter(
      (edge) =>
        edge.kind === "execution" &&
        edge.targetNodeId === node.id &&
        edge.targetPort === port.id,
    );
    const literal = Object.hasOwn(node.config, port.id);
    if (!edges.length && !literal && bound[port.id] === undefined) {
      continue;
    }
    connected.add(port.id);
    const rule = inputs[port.id];
    const name = canvasPortLabels[port.id] ?? port.id;
    if (!rule) {
      return `当前模型不支持${name}，请更换模型或移除此输入`;
    }
    const count =
      bound[port.id] ??
      (literal
        ? 1
        : edges.reduce(
            (sum, edge) =>
              sum +
              (edge.output
                ? 1
                : knownCount(
                    graph,
                    catalog,
                    edge.sourceNodeId,
                    edge.sourcePort!,
                  )),
            0,
          ));
    if (rule.maximum !== null && count > rule.maximum) {
      return `${name}最多允许 ${rule.maximum} 项，当前至少 ${count} 项`;
    }
    if (["firstFrame", "lastFrame", "referenceImages"].includes(port.id)) {
      images += count;
    }
  }
  if (
    typeof model.parameters.maxImageInputs === "number" &&
    images > model.parameters.maxImageInputs
  ) {
    return "图片输入总量超过当前模型限制";
  }
  if (
    model.parameters.exclusiveInputModes &&
    (connected.has("firstFrame") || connected.has("lastFrame")) &&
    [...connected].some((port) => port.startsWith("reference"))
  ) {
    return "当前模型的首尾帧输入与多模态参考输入不能同时使用";
  }
  const rules = videoRulesOf(model.parameters);
  const mediaInputs = videoInputState(node.config, [...connected]);
  if (
    rules?.lastFrameRequiresFirst &&
    mediaInputs.lastFrame &&
    !mediaInputs.firstFrame
  ) {
    return "使用尾帧时，请同时连接首帧图片";
  }
  if (
    rules &&
    node.config.ratio !== undefined &&
    rules.frameRatioMode !== "inferred" &&
    !videoRatioOptions(
      (model.parameters.supportedAspectRatios as string[]) ?? [],
      rules,
      mediaInputs,
    ).includes(String(node.config.ratio))
  ) {
    return "请按当前输入重新选择画面比例";
  }
  for (const [field, parameter] of [
    ["ratio", "supportedAspectRatios"],
    ["resolution", "supportedResolutions"],
  ]) {
    const options = model.parameters[parameter];
    if (
      node.config[field] !== undefined &&
      Array.isArray(options) &&
      options.length &&
      !options.includes(node.config[field])
    ) {
      return `当前模型不支持此${canvasFieldLabels[field]}`;
    }
  }
  if (
    typeof node.config.duration === "number" &&
    !automaticVideoDuration(node.config.duration, rules)
  ) {
    if (node.config.duration <= 0) {
      return "当前模型需要指定正数时长";
    }
    if (
      typeof model.parameters.minDuration === "number" &&
      node.config.duration < model.parameters.minDuration
    ) {
      return "时长低于模型允许范围";
    }
    if (
      typeof model.parameters.maxDuration === "number" &&
      node.config.duration > model.parameters.maxDuration
    ) {
      return "时长超出模型允许范围";
    }
  }
  if (
    node.config.cameraFixed === true &&
    model.parameters.supportCameraFixed === false
  ) {
    return "当前模型不支持固定镜头，请关闭或更换模型";
  }
  return null;
}
