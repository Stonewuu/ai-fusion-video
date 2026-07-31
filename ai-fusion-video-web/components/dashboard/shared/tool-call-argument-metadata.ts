import { getToolResourceLabel } from "./tool-resource-metadata";

const TOOL_ARGUMENT_LABELS: Readonly<Record<string, string>> = {
  action: "操作",
  aiPrompt: "AI 提示词",
  aiGenerated: "AI 生成",
  appearance: "外观",
  assetId: "资产 ID",
  assetIds: "资产 ID 列表",
  assetItemId: "子资产 ID",
  assetName: "资产名称",
  assets: "资产列表",
  assetType: "资产类型",
  artStyle: "项目画风",
  artStyleDescription: "画风描述",
  artStyleImagePrompt: "画风图片提示词",
  artStyleImageUrl: "画风参考图",
  atmosphere: "氛围",
  cameraAngle: "机位角度",
  cameraEquipment: "摄影设备",
  cameraFixed: "固定镜头",
  cameraMovement: "运镜方式",
  character_asset_id: "角色资产 ID",
  character_asset_ids: "角色资产 ID 列表",
  character_name: "角色名称",
  characterIds: "角色 ID 列表",
  characters: "出场角色",
  charactersJson: "角色设定",
  clothing: "服装",
  color: "颜色",
  content: "内容",
  coverUrl: "封面链接",
  createdCount: "创建数量",
  customData: "自定义数据",
  description: "描述",
  detailLevel: "详情级别",
  dialogue: "对白",
  dialogues: "对白列表",
  duration: "时长",
  environment: "环境",
  episode_version: "分集版本",
  episodeId: "分集 ID",
  episodeNumber: "集数",
  firstFrameImageUrl: "首帧图片",
  focalLength: "焦距",
  framePrompt: "画面提示词",
  frameType: "帧类型",
  genre: "题材类型",
  generatedImageUrl: "生成图片链接",
  generatedVideoUrl: "生成视频链接",
  hairstyle: "发型",
  height: "高度",
  imageUrl: "图片链接",
  imageUrls: "参考图片",
  importance: "重要程度",
  initialItem: "初始子资产",
  int_ext: "内外景",
  intExt: "内外景",
  itemId: "子资产 ID",
  items: "子项列表",
  itemType: "子资产类型",
  lastFrameImageUrl: "尾帧图片",
  location: "地点",
  material: "材质",
  message: "任务说明",
  modelType: "模型类型",
  music: "音乐",
  name: "名称",
  negativePrompt: "反向提示词",
  overwriteMode: "写入方式",
  parenthetical: "表演提示",
  personality: "性格",
  projectId: "项目 ID",
  prompt: "生成提示词",
  prop_asset_ids: "道具资产 ID 列表",
  properties: "属性",
  propIds: "道具 ID 列表",
  ratio: "画面比例",
  rawContent: "原始内容",
  referenceImageUrl: "参考图片链接",
  referenceAudioUrls: "参考音频",
  referenceImageUrls: "参考图片",
  referenceVideoUrls: "参考视频",
  remark: "备注",
  resourceId: "资源 ID",
  resourceType: "资源类型",
  scene_asset_id: "场景资产 ID",
  scene_description: "场景描述",
  scene_heading: "场次标题",
  sceneAssetItemId: "场景子资产 ID",
  sceneCount: "场次数量",
  sceneDescription: "场景描述",
  sceneExpectation: "画面预期",
  sceneHeading: "场次标题",
  sceneNumber: "场次号",
  scenes: "场次列表",
  scriptEpisodeId: "剧本分集 ID",
  scriptEpisodeIds: "剧本分集 ID 列表",
  scriptId: "剧本 ID",
  scriptSceneItemId: "剧本场次 ID",
  scriptSceneItemIds: "剧本场次 ID 列表",
  shotCount: "镜头数量",
  shotNumber: "镜头编号",
  shots: "镜头列表",
  shotType: "景别",
  sortOrder: "排序",
  sound: "声音",
  soundEffect: "音效",
  sourceType: "来源类型",
  status: "状态",
  storyboardEpisodeId: "分镜分集 ID",
  storyboardId: "分镜 ID",
  storyboardItemId: "分镜条目 ID",
  storyboardSceneId: "分镜场次 ID",
  storySynopsis: "故事梗概",
  style: "画面风格",
  synopsis: "本集梗概",
  time_of_day: "时间段",
  timeOfDay: "时间段",
  thumbnailUrl: "缩略图链接",
  title: "标题",
  transition: "转场方式",
  type: "类型",
  videoPrompt: "视频提示词",
  videoUrl: "视频链接",
  width: "宽度",
};

interface ToolArgumentLabelContext {
  arguments?: Readonly<Record<string, unknown>>;
  toolName?: string;
}

const ARGUMENT_VALUE_LABELS: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  action: {
    add: "新增",
    delete: "删除",
  },
  assetType: {
    character: "角色",
    prop: "道具",
    scene: "场景",
  },
  detailLevel: {
    full: "完整内容",
    outline: "大纲",
    scenes_only: "仅场次",
    summary: "摘要",
  },
  frameType: {
    first: "首帧",
    last: "尾帧",
  },
  int_ext: {
    EXT: "外景",
    INT: "内景",
    ext: "外景",
    int: "内景",
  },
  intExt: {
    EXT: "外景",
    INT: "内景",
    ext: "外景",
    int: "内景",
  },
  itemType: {
    back: "背面",
    detail: "细节",
    expression: "表情",
    front: "正面",
    initial: "初始",
    original: "原始",
    pose: "姿势",
    side: "侧面",
    variant: "变体",
  },
  modelType: {
    all: "全部",
    image: "图片模型",
    video: "视频模型",
  },
  overwriteMode: {
    append: "追加",
    false: "追加写入",
    overwrite: "覆盖",
    replace: "替换",
    true: "覆盖已有数据",
  },
  sourceType: {
    ai: "AI 生成",
    manual: "手动创建",
  },
  type: {
    character: "角色",
    prop: "道具",
    scene: "场景",
  },
};

const TOOL_ARGUMENT_VALUE_LABELS: Readonly<
  Record<string, Readonly<Record<string, Readonly<Record<string, string>>>>>
> = {
  save_project: {
    status: {
      0: "筹备中",
      1: "进行中",
      2: "已完成",
      3: "已归档",
    },
  },
  save_script_episode: {
    sourceType: {
      0: "手动创建",
      1: "AI 创作",
      2: "文本解析",
    },
  },
  update_asset_item: {
    sourceType: {
      1: "用户上传",
      2: "AI 生成",
    },
  },
  update_storyboard_item: {
    status: {
      0: "草稿",
      1: "正常",
    },
  },
  update_storyboard_scene: {
    status: {
      0: "草稿",
      1: "正常",
    },
  },
};

export function getToolArgumentLabel(
  key: string,
  context: ToolArgumentLabelContext = {},
): string {
  if (key === "resourceType" && context.toolName?.startsWith("delete_")) {
    return "删除对象";
  }
  if (key === "resourceId") {
    const resourceType = context.arguments?.resourceType;
    const resourceLabel = getToolResourceLabel(
      context.toolName,
      typeof resourceType === "string" ? resourceType : undefined,
    );
    if (resourceLabel) return `${resourceLabel} ID`;
  }
  const declaredLabel = TOOL_ARGUMENT_LABELS[key];
  if (declaredLabel !== undefined) return declaredLabel;

  const readable = key
    .replace(/[_-]+/g, " ")
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim();
  if (!readable) {
    throw new Error("Tool argument key must not be blank");
  }
  return `${readable.charAt(0).toUpperCase()}${readable.slice(1)}`;
}

export function getToolArgumentValueLabel(
  key: string,
  value: number | string,
  toolName?: string,
): string | undefined {
  const normalizedValue = String(value);
  if (key === "resourceType") {
    const resourceLabel = getToolResourceLabel(toolName, normalizedValue);
    if (resourceLabel) return resourceLabel;
  }
  return TOOL_ARGUMENT_VALUE_LABELS[toolName ?? ""]?.[key]?.[normalizedValue]
    ?? ARGUMENT_VALUE_LABELS[key]?.[normalizedValue];
}
