export const toolDisplayNames: Record<string, string> = {
  get_project: "查询项目详情",
  list_my_projects: "列出我的项目",
  save_project: "保存项目",
  delete_project: "删除项目",
  list_project_assets: "列出资产",
  create_asset: "创建资产",
  get_asset: "查询资产详情",
  update_asset: "更新资产",
  batch_create_assets: "批量创建资产",
  query_asset_items: "查询子资产",
  batch_create_asset_items: "批量创建子资产",
  add_asset_item: "添加子资产",
  update_asset_item: "更新子资产",
  delete_asset_resource: "删除资产资源",
  update_asset_image: "更新子资产图片",
  query_asset_metadata: "查询资产属性定义",
  get_project_script: "查询项目剧本信息",
  get_script: "查询剧本",
  get_script_structure: "查询剧本结构",
  update_script: "更新剧本正文",
  update_script_info: "更新剧本信息",
  save_script_episode: "保存集记录",
  save_script_scene_items: "保存场次数据",
  get_script_episode: "查询集详情",
  get_script_scene: "查询剧本场次详情",
  manage_script_scenes: "管理剧本场次",
  update_script_scene: "更新剧本场次",
  delete_script_child_resource: "删除剧本子资源",
  get_project_storyboard: "查询项目分镜",
  get_storyboard: "查询分镜详情",
  insert_storyboard_item: "插入分镜条目",
  update_storyboard_scene: "更新分镜场次",
  update_storyboard_item: "更新分镜镜头",
  delete_storyboard_child_resource: "删除分镜子资源",
  save_storyboard_episode: "保存分镜集",
  save_storyboard_scene_shots: "保存场次分镜",
  get_generation_model_capabilities: "查询生成模型能力",
  generate_image: "AI 生成图片",
  generate_video: "AI 生成视频",
  load_skill_through_path: "加载技能资源",
  update_storyboard_item_video: "更新分镜视频",
  update_storyboard_item_frame: "更新分镜首尾帧",
  get_storyboard_scene_items: "查询场次镜头列表",
  episode_scene_writer: "分集场次解析（子Agent）",
  episode_script_creator: "分集剧本创作（子Agent）",
  episode_storyboard_writer: "分集分镜编写（子Agent）",
  storyboard_asset_preprocessor: "子资产预处理（子Agent）",
  generate_asset_image: "生成资产图片（子Agent）",
  generate_storyboard_frame: "生成分镜首尾帧（子Agent）",
  generate_storyboard_video: "生成分镜视频（子Agent）",
};

export const subAgentToolNames = [
  "episode_scene_writer",
  "episode_script_creator",
  "episode_storyboard_writer",
  "storyboard_asset_preprocessor",
  "generate_asset_image",
  "generate_storyboard_frame",
  "generate_storyboard_video",
];

export const agentTypeNames: Record<string, string> = {
  script_full_parse: "剧本全量解析",
  story_to_script: "故事转剧本",
  script_to_storyboard: "剧本转分镜",
  script_episode_parse: "分集上传解析",
  episode_scene_writer: "分集场次编写",
  episode_script_creator: "分集剧本创作",
  episode_storyboard_writer: "分集分镜编写",
  storyboard_asset_preprocessor: "子资产预处理",
  asset_image_gen: "资产图片生成",
  asset_image_executor: "资产图片执行",
  storyboard_frame_gen: "分镜首尾帧生成",
  storyboard_frame_executor: "分镜首尾帧执行",
  storyboard_video_gen: "分镜视频生成",
  storyboard_video_executor: "分镜视频执行",
  storyboard_episode_compose: "分集合成视频",
  script_assistant: "剧本助手",
  ai_media: "默认助手",
  concept_visualizer: "概念可视化",
};

export const assetTypeNames: Record<string, string> = {
  character: "角色",
  scene: "场景",
  prop: "道具",
  vehicle: "载具",
  building: "建筑",
  costume: "服装",
  effect: "特效",
};

export const storyboardShotTypeNames: Record<string, string> = {
  远景: "远景",
  全景: "全景",
  中景: "中景",
  近景: "近景",
  特写: "特写",
};

export function getToolDisplayName(name: string) {
  return toolDisplayNames[name] || name;
}

export function isSubAgentTool(name: string) {
  return subAgentToolNames.includes(name);
}

export function getAgentTypeName(type: string): string {
  return agentTypeNames[type] || type;
}
