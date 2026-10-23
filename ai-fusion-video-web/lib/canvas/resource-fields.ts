import type { ResourceType } from "@/lib/api/resource-events";

export interface ResourceField {
  key: string;
  label: string;
  kind:
    | "text"
    | "long"
    | "number"
    | "choice"
    | "image"
    | "video"
    | "audio"
    | "asset"
    | "assetItem"
    | "scriptEpisode"
    | "tags"
    | "dialogues"
    | "characters"
    | "json";
  section?: string;
  options?: { value: string; label: string }[];
  multiple?: boolean;
  assetType?: string;
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  jsonType?: "object" | "array";
}
const field = (
  key: string,
  label: string,
  kind: ResourceField["kind"] = "text",
  extra: Partial<ResourceField> = {},
): ResourceField => ({ key, label, kind, ...extra });
const status = field("status", "状态", "choice", {
  required: true,
  options: [
    { value: "0", label: "草稿" },
    { value: "1", label: "正常" },
  ],
});
const sceneFields = [
  field("sceneNumber", "场号"),
  field("sceneHeading", "场景标题"),
  field("location", "地点"),
  field("timeOfDay", "时间"),
  field("intExt", "内外景", "choice", {
    options: [
      { value: "内", label: "内景" },
      { value: "外", label: "外景" },
      { value: "内外", label: "内外景" },
    ],
  }),
];
const scriptMetadata = [
  field("storySynopsis", "故事梗概", "long"),
  field("summary", "摘要", "long"),
  field("genre", "题材"),
  field("targetAudience", "目标受众"),
  field("durationEstimate", "预计时长（秒）", "number", { min: 0 }),
];
export const RESOURCE_FIELDS: Record<ResourceType, ResourceField[]> = {
  project: [
    field("name", "项目名称", "text", { required: true }),
    field("description", "项目简介", "long"),
    field("status", "项目状态", "choice", {
      options: [
        { value: "0", label: "筹备" },
        { value: "1", label: "制作中" },
        { value: "2", label: "已完成" },
        { value: "3", label: "已归档" },
      ],
    }),
    field("artStyle", "画风预设"),
    field("artStyleDescription", "画风描述", "long"),
    field("artStyleImagePrompt", "参考图提示词", "long"),
    field("artStyleImageUrl", "画风参考图", "image"),
    field("coverUrl", "项目封面", "image"),
    field("properties", "项目扩展设置", "json", {
      section: "扩展",
      jsonType: "object",
    }),
  ],
  script: [
    ...scriptMetadata,
    field("content", "剧本文本", "long"),
    field("rawContent", "原文", "long"),
    field("charactersJson", "人物表", "characters"),
    status,
  ],
  scriptEpisode: [
    field("episodeNumber", "集号", "number", { min: 1, required: true }),
    field("title", "分集名称"),
    field("synopsis", "梗概", "long"),
    field("rawContent", "分集原文", "long"),
    field("durationEstimate", "预计时长（秒）", "number", { min: 0 }),
    status,
  ],
  scriptScene: [
    ...sceneFields,
    field("sceneDescription", "场景描述", "long"),
    field("dialogues", "对白与动作", "dialogues"),
    field("characters", "角色名称", "tags"),
    field("characterAssetIds", "关联角色", "asset", {
      assetType: "character",
      multiple: true,
      section: "关联资产",
    }),
    field("sceneAssetId", "关联场景", "asset", {
      assetType: "scene",
      section: "关联资产",
    }),
    field("propAssetIds", "关联道具", "asset", {
      assetType: "prop",
      multiple: true,
      section: "关联资产",
    }),
    status,
  ],
  asset: [
    field("name", "资产名称", "text", { required: true }),
    field("type", "资产类型", "choice", {
      required: true,
      options: [
        { value: "character", label: "角色" },
        { value: "scene", label: "场景" },
        { value: "prop", label: "道具" },
      ],
    }),
    field("description", "描述", "long"),
    field("aiPrompt", "生成提示词", "long"),
    field("coverUrl", "封面", "image"),
    field("tags", "标签", "tags"),
    status,
  ],
  assetItem: [
    field("name", "变体名称"),
    field("itemType", "变体类别", "choice", {
      options: [
        { value: "variant", label: "变体" },
        { value: "outfit", label: "服装" },
        { value: "expression", label: "表情" },
        { value: "pose", label: "姿势" },
        { value: "angle", label: "角度" },
      ],
    }),
    field("aiPrompt", "生成提示词", "long"),
    field("imageUrl", "变体图片", "image"),
    field("thumbnailUrl", "缩略图", "image"),
  ],
  storyboard: [
    field("description", "分镜说明", "long"),
    field("customColumns", "自定义列", "json", {
      section: "扩展",
      jsonType: "array",
    }),
    status,
  ],
  storyboardEpisode: [
    field("episodeNumber", "集号", "number", { min: 1, required: true }),
    field("title", "分集名称"),
    field("synopsis", "梗概", "long"),
    field("scriptEpisodeId", "对应剧本分集", "scriptEpisode"),
    status,
  ],
  storyboardScene: [...sceneFields, status],
  shot: [
    field("shotNumber", "镜号"),
    field("content", "画面内容", "long"),
    field("sceneExpectation", "场景要求", "long"),
    field("shotType", "景别"),
    field("duration", "时长（秒）", "number", {
      min: 0,
      max: 86400,
      step: 0.01,
    }),
    field("dialogue", "对白", "long"),
    field("sound", "声音", "long"),
    field("soundEffect", "音效"),
    field("music", "音乐"),
    field("cameraMovement", "运镜", "text", { section: "摄像与转场" }),
    field("cameraAngle", "角度", "text", { section: "摄像与转场" }),
    field("cameraEquipment", "设备", "text", { section: "摄像与转场" }),
    field("focalLength", "焦距", "text", { section: "摄像与转场" }),
    field("transition", "转场", "text", { section: "摄像与转场" }),
    field("firstFramePrompt", "首帧提示词", "long", {
      section: "媒体与生成提示",
    }),
    field("lastFramePrompt", "尾帧提示词", "long", {
      section: "媒体与生成提示",
    }),
    field("videoPrompt", "视频提示词", "long", { section: "媒体与生成提示" }),
    field("firstFrameImageUrl", "首帧", "image", { section: "媒体与生成提示" }),
    field("lastFrameImageUrl", "尾帧", "image", { section: "媒体与生成提示" }),
    field("imageUrl", "画面图片", "image", { section: "媒体与生成提示" }),
    field("referenceImageUrl", "参考图片", "image", {
      section: "媒体与生成提示",
    }),
    field("videoUrl", "镜头视频", "video", { section: "媒体与生成提示" }),
    field("generatedImageUrl", "生成图片", "image", {
      section: "媒体与生成提示",
    }),
    field("generatedVideoUrl", "生成视频", "video", {
      section: "媒体与生成提示",
    }),
    field("characterIds", "角色变体", "assetItem", {
      assetType: "character",
      multiple: true,
      section: "关联资产",
    }),
    field("sceneAssetItemId", "场景变体", "assetItem", {
      assetType: "scene",
      section: "关联资产",
    }),
    field("propIds", "道具变体", "assetItem", {
      assetType: "prop",
      multiple: true,
      section: "关联资产",
    }),
    field("remark", "备注", "long"),
    field("customData", "扩展字段", "json", {
      section: "扩展",
      jsonType: "object",
    }),
    status,
  ],
};
export const CHILD_TYPES: Partial<
  Record<ResourceType, { type: ResourceType; parentKey: string }>
> = {
  script: { type: "scriptEpisode", parentKey: "scriptId" },
  scriptEpisode: { type: "scriptScene", parentKey: "episodeId" },
  storyboard: { type: "storyboardEpisode", parentKey: "storyboardId" },
  storyboardEpisode: { type: "storyboardScene", parentKey: "episodeId" },
  storyboardScene: { type: "shot", parentKey: "storyboardSceneId" },
  asset: { type: "assetItem", parentKey: "assetId" },
};
