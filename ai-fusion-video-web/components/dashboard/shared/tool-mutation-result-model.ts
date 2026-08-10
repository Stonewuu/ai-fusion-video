import { getToolResourceLabel } from "./tool-resource-metadata";

type ToolResultObject = Record<string, unknown>;

interface EntityDisplayConfig {
  idKey: string;
  label: string;
  nestedKey?: string;
  summaryKeys?: string[];
}

export interface ToolMutationPresentation {
  counts: string[];
  failed: boolean;
  id?: unknown;
  label: string;
  message: string;
  summary?: string;
}

const TOOL_ENTITY_CONFIG: Readonly<Record<string, EntityDisplayConfig>> = {
  add_asset_item: {
    idKey: "assetItemId",
    label: "子资产",
    summaryKeys: ["assetName", "itemType"],
  },
  delete_project: { idKey: "deletedProjectId", label: "项目" },
  manage_script_scenes: { idKey: "scriptEpisodeId", label: "剧本分集" },
  save_project: {
    idKey: "id",
    label: "项目",
    nestedKey: "project",
    summaryKeys: ["name"],
  },
  save_script_episode: {
    idKey: "scriptEpisodeId",
    label: "剧本分集",
    summaryKeys: ["episodeNumber", "title"],
  },
  save_script_scene_items: { idKey: "scriptEpisodeId", label: "剧本分集" },
  update_asset: { idKey: "assetId", label: "资产" },
  update_asset_image: {
    idKey: "itemId",
    label: "子资产",
    summaryKeys: ["assetName", "itemType"],
  },
  update_asset_item: {
    idKey: "id",
    label: "子资产",
    nestedKey: "assetItem",
    summaryKeys: ["name", "itemType"],
  },
  update_script: { idKey: "scriptId", label: "剧本" },
  update_script_info: { idKey: "scriptId", label: "剧本" },
  update_script_scene: { idKey: "scriptSceneItemId", label: "剧本场次" },
  update_storyboard_item: {
    idKey: "id",
    label: "分镜镜头",
    nestedKey: "storyboardItem",
    summaryKeys: ["shotNumber", "content"],
  },
  update_storyboard_item_frame: {
    idKey: "storyboardItemId",
    label: "分镜镜头",
  },
  update_storyboard_item_video: {
    idKey: "storyboardItemId",
    label: "分镜镜头",
  },
  update_storyboard_scene: {
    idKey: "id",
    label: "分镜场次",
    nestedKey: "storyboardScene",
    summaryKeys: ["sceneNumber", "sceneHeading"],
  },
};

function asObject(value: unknown): ToolResultObject | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as ToolResultObject
    : undefined;
}

function truncate(value: string): string {
  return value.length > 80 ? `${value.slice(0, 80)}…` : value;
}

function resolveDeleteDisplay(
  data: ToolResultObject,
  toolName: string,
): { id?: unknown; label: string } | undefined {
  if (toolName === "delete_project") {
    return { id: data.deletedProjectId, label: "项目" };
  }
  if (!toolName.startsWith("delete_")) return undefined;
  const resourceType = typeof data.deletedResourceType === "string"
    ? data.deletedResourceType
    : undefined;
  return {
    id: data.deletedResourceId,
    label: getToolResourceLabel(toolName, resourceType) ?? "资源",
  };
}

function defaultMessage(
  data: ToolResultObject,
  toolName: string,
  label: string,
  failed: boolean,
): string {
  if (failed) return `${label}操作失败`;
  if (toolName.startsWith("delete_")) return `${label}已删除`;
  if (toolName === "save_project") {
    return data.operation === "created" ? "项目创建成功" : "项目更新成功";
  }
  if (toolName.startsWith("save_")) return `${label}保存成功`;
  if (toolName.startsWith("update_")) return `${label}更新成功`;
  return `${label}操作成功`;
}

export function buildToolMutationPresentation(
  data: unknown,
  toolName: string,
): ToolMutationPresentation {
  const obj = asObject(data) ?? {};
  const status = typeof obj.status === "string" ? obj.status : undefined;
  const failed = status === "error" || status === "failed";
  const deleteDisplay = resolveDeleteDisplay(obj, toolName);
  const config = TOOL_ENTITY_CONFIG[toolName];
  const nested = config?.nestedKey ? asObject(obj[config.nestedKey]) : undefined;
  const displaySource = nested ?? obj;
  const label = deleteDisplay?.label ?? config?.label ?? "数据";
  const id = deleteDisplay?.id ?? (config ? displaySource[config.idKey] : undefined);
  const summary = config?.summaryKeys
    ?.map((key) => displaySource[key])
    .filter((value): value is string | number => (
      typeof value === "string" || typeof value === "number"
    ))
    .map((value) => truncate(String(value)))
    .join(" · ");
  const message = typeof obj.message === "string"
    ? obj.message
    : defaultMessage(obj, toolName, label, failed);
  const counts = [
    ["createdCount", "新增"],
    ["deletedCount", "删除"],
    ["sceneCount", "场次"],
    ["shotCount", "镜头"],
  ]
    .filter(([key]) => typeof obj[key] === "number")
    .map(([key, countLabel]) => `${countLabel} ${String(obj[key])}`);

  return { counts, failed, id, label, message, summary };
}
