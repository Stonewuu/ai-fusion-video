import { RESOURCE_FIELDS } from "./resource-fields";
import { canvasFieldLabels } from "./labels";
import type { ResourceType } from "@/lib/api/resource-events";

export const businessOperationNames: Record<string, string> = {
  PATCH: "修改内容",
  CREATE: "新建内容",
  DELETE: "删除内容",
  INITIALIZE_WORKSPACE: "初始化工作区",
  FORK_PROJECT: "创建独立项目副本",
  IMPORT_TEMPLATE: "从模板创建项目",
  REPLACE_SCRIPT_SOURCE: "替换剧本原文",
  MANAGE_SCRIPT_SCENES: "整理剧本场次",
  BATCH_CREATE_ASSETS: "批量创建资产",
  MANAGE_STORYBOARD_CHILDREN: "整理分镜结构",
  BATCH_PATCH_STORYBOARD: "批量修改分镜",
  BIND_MEDIA: "采用媒体结果",
  COMPOSE_WRITEBACK: "保存合成结果",
  ADOPT_EPISODE_MEDIA: "采用成片",
  RESTORE: "恢复项目内容",
};
const historyLabels: Record<string, string> = {
  deleted: "删除状态",
  sortOrder: "顺序",
  episodeCount: "分集数量",
  sceneCount: "场次数量",
  totalDuration: "总时长",
  title: "名称",
  name: "名称",
  type: "类型",
  deletedId: "删除标记",
  composeStatus: "合成状态",
  composedVideoUrl: "成片",
  bgmAudioUrl: "配乐",
  bgmSfxAudioUrl: "音效",
  composedAt: "合成时间",
  composeJobId: "合成任务",
  bgmTaskId: "音频任务",
  ownerId: "所有者编号",
  ownerType: "所有者类型",
  projectId: "所属项目",
  scriptId: "所属剧本",
  storyboardId: "所属分镜",
  assetId: "主资产",
  episodeId: "所属分集",
  storyboardEpisodeId: "分镜分集",
  storyboardSceneId: "分镜场次",
};
export const historyFieldLabel = (type: ResourceType, field: string) =>
  RESOURCE_FIELDS[type].find((item) => item.key === field)?.label ??
  historyLabels[field] ??
  canvasFieldLabels[field] ??
  field;
export function historyText(value: unknown): string {
  if (value == null || value === "") return "未填写";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value !== "object") return String(value);
  if (Array.isArray(value)) return value.length ? value.map(historyText).join("\n\n") : "无";
  const content = value as Record<string, unknown>;
  if (typeof content.content === "string" && typeof content.type === "number" && content.type >= 1 && content.type <= 5) {
    const label = typeof content.character_name === "string" && content.character_name
      ? content.character_name : ["", "对白", "动作", "旁白", "镜头指令", "环境描写"][content.type];
    return `${label}${content.parenthetical ? `（${content.parenthetical}）` : ""}：${content.content}`;
  }
  return Object.entries(content).map(([key, item]) => `${historyLabels[key] ?? canvasFieldLabels[key] ?? key}：${historyText(item)}`).join("\n");
}
