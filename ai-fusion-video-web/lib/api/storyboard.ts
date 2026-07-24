import { http } from "./client";

// ========== 类型定义 ==========

/** 分镜脚本 */
export interface Storyboard {
  id: number;
  projectId: number;
  scriptId: number | null;
  title: string;
  description: string | null;
  customColumns: string | null;
  scope: number;
  ownerType: number;
  ownerId: number;
  totalDuration: number | null;
  status: number;
  createTime: string;
  updateTime: string;
}

/** 分镜集合成状态: 0未开始 1合成中 2已完成 3失败 */
export type EpisodeComposeStatus = 0 | 1 | 2 | 3;

/** 分镜集 */
export interface StoryboardEpisode {
  id: number;
  storyboardId: number;
  scriptEpisodeId: number | null;
  episodeNumber: number | null;
  title: string | null;
  synopsis: string | null;
  sortOrder: number;
  status: number;
  composedVideoUrl: string | null;
  composeStatus: EpisodeComposeStatus;
  composeErrorMsg: string | null;
  composedAt: string | null;
  createTime: string;
  updateTime: string;
}

/** 分镜场次 */
export interface StoryboardScene {
  id: number;
  episodeId: number;
  storyboardId: number;
  sceneNumber: string | null;
  sceneHeading: string | null;
  location: string | null;
  timeOfDay: string | null;
  intExt: string | null;
  sortOrder: number;
  status: number;
  createTime: string;
  updateTime: string;
}

/** 分镜条目 */
export interface StoryboardItem {
  id: number;
  storyboardId: number;
  storyboardEpisodeId: number | null;
  storyboardSceneId: number | null;
  sortOrder: number;
  shotNumber: string | null;
  autoShotNumber: string | null;
  imageUrl: string | null;
  referenceImageUrl: string | null;
  videoUrl: string | null;
  generatedImageUrl: string | null;
  firstFrameImageUrl: string | null;
  lastFrameImageUrl: string | null;
  firstFramePrompt: string | null;
  lastFramePrompt: string | null;
  generatedVideoUrl: string | null;
  videoPrompt: string | null;
  shotType: string | null;
  duration: number | null;
  content: string | null;
  sceneExpectation: string | null;
  sound: string | null;
  dialogue: string | null;
  soundEffect: string | null;
  music: string | null;
  cameraMovement: string | null;
  cameraAngle: string | null;
  cameraEquipment: string | null;
  focalLength: string | null;
  transition: string | null;
  characterIds: string | null;
  sceneAssetItemId: number | null;
  propIds: string | null;
  remark: string | null;
  customData: string | null;
  aiGenerated: boolean;
  status: number;
  createTime: string;
  updateTime: string;
}

// ========== 请求类型 ==========

/** 创建分镜请求 */
export interface StoryboardCreateReq {
  projectId: number;
  scriptId?: number;
  title: string;
  description?: string;
}

/** 更新分镜请求 */
export interface StoryboardUpdateReq {
  id: number;
  title?: string;
  description?: string;
  status?: number;
}

/** 创建分镜集请求 */
export interface StoryboardEpisodeCreateReq {
  storyboardId: number;
  scriptEpisodeId?: number;
  episodeNumber?: number;
  title?: string;
  synopsis?: string;
  sortOrder?: number;
}

/** 更新分镜集请求 */
export interface StoryboardEpisodeUpdateReq {
  id: number;
  scriptEpisodeId?: number;
  episodeNumber?: number;
  title?: string;
  synopsis?: string;
  sortOrder?: number;
}

/** 创建分镜场次请求 */
export interface StoryboardSceneCreateReq {
  episodeId: number;
  storyboardId: number;
  sceneNumber?: string;
  sceneHeading?: string;
  location?: string;
  timeOfDay?: string;
  intExt?: string;
  sortOrder?: number;
}

/** 更新分镜场次请求 */
export interface StoryboardSceneUpdateReq {
  id: number;
  sceneNumber?: string;
  sceneHeading?: string;
  location?: string;
  timeOfDay?: string;
  intExt?: string;
  sortOrder?: number;
}

/** 创建分镜条目请求 */
export interface StoryboardItemCreateReq {
  storyboardId: number;
  storyboardEpisodeId?: number;
  storyboardSceneId?: number;
  shotNumber?: string;
  shotType?: string;
  content?: string;
  sceneExpectation?: string;
  dialogue?: string;
  soundEffect?: string;
  music?: string;
  cameraMovement?: string;
  cameraAngle?: string;
  transition?: string;
  duration?: number;
  sortOrder?: number;
  firstFrameImageUrl?: string | null;
  lastFrameImageUrl?: string | null;
  firstFramePrompt?: string | null;
  lastFramePrompt?: string | null;
  characterIds?: string | null;
  sceneAssetItemId?: number | null;
  propIds?: string | null;
}

/** 更新分镜条目请求 */
export interface StoryboardItemUpdateReq {
  id: number;
  shotNumber?: string;
  shotType?: string;
  content?: string;
  sceneExpectation?: string;
  dialogue?: string;
  soundEffect?: string;
  music?: string;
  cameraMovement?: string;
  cameraAngle?: string;
  transition?: string;
  duration?: number;
  sortOrder?: number;
  imageUrl?: string;
  firstFrameImageUrl?: string | null;
  lastFrameImageUrl?: string | null;
  firstFramePrompt?: string | null;
  lastFramePrompt?: string | null;
  videoPrompt?: string | null;
  status?: number;
}

/** 分镜条目资产关联局部更新请求 */
export interface StoryboardItemAssetsUpdateReq {
  /** 字段缺省表示不修改，空数组表示清空 */
  characterIds?: number[];
  /** 字段缺省表示不修改，显式 null 表示清空 */
  sceneAssetItemId?: number | null;
  /** 字段缺省表示不修改，空数组表示清空 */
  propIds?: number[];
}

/** 分镜首尾帧类型 */
export type StoryboardFrameType = "first" | "last";

/** 更新分镜首尾帧请求 */
export interface StoryboardFrameUpdateReq {
  frameType: StoryboardFrameType;
  imageUrl?: string | null;
  prompt?: string | null;
}

// ========== API ==========

export const storyboardApi = {
  // ========== 分镜脚本 ==========

  /** 获取分镜详情 */
  get: (id: number) => http.get<never, Storyboard>(`/api/storyboard/${id}`),

  /** 按项目查询分镜列表 */
  list: (projectId: number) =>
    http.get<never, Storyboard[]>(`/api/storyboard/list?projectId=${projectId}`),

  /** 创建分镜 */
  create: (data: StoryboardCreateReq) =>
    http.post<never, Storyboard>("/api/storyboard", data),

  /** 更新分镜 */
  update: (data: StoryboardUpdateReq) =>
    http.put<never, Storyboard>("/api/storyboard", data),

  /** 删除分镜 */
  delete: (id: number) => http.delete<never, boolean>(`/api/storyboard/${id}`),

  // ========== 分镜集 ==========

  /** 获取分镜集列表 */
  listEpisodes: (storyboardId: number) =>
    http.get<never, StoryboardEpisode[]>(`/api/storyboard/${storyboardId}/episodes`),

  /** 获取分镜集详情 */
  getEpisode: (id: number) =>
    http.get<never, StoryboardEpisode>(`/api/storyboard/episode/${id}`),

  /** 创建分镜集 */
  createEpisode: (data: StoryboardEpisodeCreateReq) =>
    http.post<never, StoryboardEpisode>("/api/storyboard/episode", data),

  /** 更新分镜集 */
  updateEpisode: (data: StoryboardEpisodeUpdateReq) =>
    http.put<never, StoryboardEpisode>("/api/storyboard/episode", data),

  /** 删除分镜集 */
  deleteEpisode: (id: number) =>
    http.delete<never, boolean>(`/api/storyboard/episode/${id}`),

  /** 绑定分镜集和剧本分集 */
  bindScriptEpisode: (id: number, scriptEpisodeId: number) =>
    http.put<never, StoryboardEpisode>(
      `/api/storyboard/episode/${id}/bindScriptEpisode`,
      { scriptEpisodeId }
    ),

  /** 清空分镜集下的场次和镜头 */
  clearEpisodeContent: (id: number) =>
    http.post<never, boolean>(`/api/storyboard/episode/${id}/clearContent`),

  /** 提交本集合成视频任务（异步） */
  composeEpisodeVideo: (episodeId: number) =>
    http.post<never, string>(`/api/storyboard/episode/${episodeId}/compose-video`),

  // ========== 分镜场次 ==========

  /** 按集获取分镜场次列表 */
  listScenesByEpisode: (episodeId: number) =>
    http.get<never, StoryboardScene[]>(`/api/storyboard/episode/${episodeId}/scenes`),

  /** 按分镜获取分镜场次列表 */
  listScenesByStoryboard: (storyboardId: number) =>
    http.get<never, StoryboardScene[]>(`/api/storyboard/${storyboardId}/scenes`),

  /** 获取分镜场次详情 */
  getScene: (id: number) =>
    http.get<never, StoryboardScene>(`/api/storyboard/scene/${id}`),

  /** 创建分镜场次 */
  createScene: (data: StoryboardSceneCreateReq) =>
    http.post<never, StoryboardScene>("/api/storyboard/scene", data),

  /** 更新分镜场次 */
  updateScene: (data: StoryboardSceneUpdateReq) =>
    http.put<never, StoryboardScene>("/api/storyboard/scene", data),

  /** 删除分镜场次 */
  deleteScene: (id: number) =>
    http.delete<never, boolean>(`/api/storyboard/scene/${id}`),

  // ========== 分镜条目 ==========

  /** 获取分镜条目列表（按分镜） */
  listItems: (storyboardId: number) =>
    http.get<never, StoryboardItem[]>(`/api/storyboard/${storyboardId}/items`),

  /** 获取分镜条目列表（按场次） */
  listItemsByScene: (sceneId: number) =>
    http.get<never, StoryboardItem[]>(`/api/storyboard/scene/${sceneId}/items`),

  /** 获取分镜条目详情 */
  getItem: (id: number) =>
    http.get<never, StoryboardItem>(`/api/storyboard/item/${id}`),

  /** 创建分镜条目 */
  createItem: (data: StoryboardItemCreateReq) =>
    http.post<never, StoryboardItem>("/api/storyboard/item", data),

  /** 更新分镜条目 */
  updateItem: (data: StoryboardItemUpdateReq) =>
    http.put<never, StoryboardItem>("/api/storyboard/item", data),

  /** 局部更新分镜条目资产关联 */
  updateItemAssets: (id: number, data: StoryboardItemAssetsUpdateReq) =>
    http.patch<never, StoryboardItem>(`/api/storyboard/item/${id}/assets`, data),

  /** 更新分镜条目首尾帧 */
  updateFrame: (id: number, data: StoryboardFrameUpdateReq) =>
    http.put<never, StoryboardItem>(`/api/storyboard/item/${id}/updateFrame`, data),

  /** 删除分镜条目 */
  deleteItem: (id: number) =>
    http.delete<never, boolean>(`/api/storyboard/item/${id}`),

  /** 批量创建分镜条目 */
  batchCreateItems: (storyboardId: number, items: StoryboardItemCreateReq[]) =>
    http.post<never, boolean>(`/api/storyboard/${storyboardId}/items/batch`, items),

  /** 批量更新分镜条目排序 */
  batchUpdateItemSort: (ids: number[]) =>
    http.post<never, boolean>("/api/storyboard/items/batch-sort", { ids }),
};
