import type { ResourceType } from "@/lib/api/resource-events";
import type { ResourceField } from "./resource-fields";

/** 自由文本字段的快捷选项；始终保留自定义文本，不将建议当成接口枚举。 */
export const resourceFieldSuggestions: Record<string, string[]> = {
  timeOfDay: ["日", "夜", "清晨", "黄昏"],
  genre: ["剧情", "悬疑", "喜剧", "科幻", "奇幻", "动画", "纪录"],
  targetAudience: ["全年龄", "儿童", "青少年", "成年人"],
  shotType: ["远景", "全景", "中景", "近景", "特写", "大特写"],
  cameraMovement: ["固定", "推镜", "拉镜", "摇镜", "移镜", "跟拍", "环绕"],
  cameraAngle: ["平视", "俯视", "仰视", "鸟瞰", "过肩", "主观视角"],
  cameraEquipment: ["三脚架", "手持", "稳定器", "轨道", "摇臂", "无人机"],
  focalLength: ["24mm", "35mm", "50mm", "85mm"],
  transition: ["硬切", "叠化", "淡入", "淡出", "匹配剪辑"],
};

export function resourceFieldSection(type: ResourceType, field: ResourceField): string {
  if (field.section) return field.section;
  if (["image", "video", "audio"].includes(field.kind) || ["aiPrompt", "artStyleImagePrompt"].includes(field.key)) return "媒体与创作";
  if (type === "project" && field.key.startsWith("artStyle")) return "画风设定";
  if (type === "script" && ["genre", "targetAudience", "durationEstimate"].includes(field.key)) return "作品设定";
  if (type === "script" && field.key === "rawContent") return "原始素材";
  if (field.kind === "characters" || field.kind === "tags") return "人物与标签";
  if (type === "shot" && ["dialogue", "sound", "soundEffect", "music"].includes(field.key)) return "对白与声音";
  if (["remark", "status"].includes(field.key)) return "管理与备注";
  return "基本内容";
}
