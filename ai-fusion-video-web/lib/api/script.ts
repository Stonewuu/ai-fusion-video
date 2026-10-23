import { http } from "./client";

// ========== 类型定义 ==========

export interface Script {
  id: number;
  version: number;
  projectId: number;
  title: string;
  content: string | null;
  rawContent: string | null;
  totalEpisodes: number;
  storySynopsis: string | null;
  charactersJson: { name: string; assetId?: number; description?: string; importance?: string }[] | null;
  sourceType: number;
  parsingStatus: number;
  parsingProgress: string | null;
  summary: string | null;
  genre: string | null;
  targetAudience: string | null;
  durationEstimate: number | null;
  aiGenerated: boolean;
  status: number;
  createTime: string;
  updateTime: string;
}

export interface ScriptEpisode {
  id: number;
  scriptId: number;
  episodeNumber: number;
  title: string;
  synopsis: string | null;
  rawContent: string | null;
  durationEstimate: number | null;
  totalScenes: number;
  sourceType: number;
  sortOrder: number;
  parsingStatus: number;
  status: number;
  version: number;
  createTime: string;
  updateTime: string;
}

/** 对白/动作元素 */
export interface DialogueElement {
  type: number; // 1=对白, 2=动作, 3=旁白, 4=镜头指令, 5=环境描写
  character_name?: string;
  character_asset_id?: number;
  content: string;
  parenthetical?: string;
  sortOrder?: number;
}

export interface SceneItem {
  id: number;
  episodeId: number;
  scriptId: number;
  sceneNumber: string;
  sceneHeading: string;
  location: string | null;
  timeOfDay: string | null;
  intExt: string | null;
  characters: string[] | null;
  characterAssetIds: number[] | null;
  sceneAssetId: number | null;
  propAssetIds: number[] | null;
  sceneDescription: string | null;
  dialogues: DialogueElement[] | null;
  sortOrder: number;
  status: number;
  version: number;
  createTime: string;
  updateTime: string;
}

export interface ScriptUpdateReq {
  id: number;
  version: number;
  content?: string | null;
  rawContent?: string | null;
  storySynopsis?: string | null;
  genre?: string | null;
  targetAudience?: string | null;
  durationEstimate?: number | null;
}

export interface EpisodeCreateReq {
  scriptId: number;
  parentVersion: number;
  episodeNumber: number;
  title?: string | null;
  synopsis?: string | null;
  rawContent?: string | null;
  durationEstimate?: number | null;
  sortOrder?: number;
}

export interface EpisodeUpdateReq {
  id: number;
  title?: string | null;
  synopsis?: string | null;
  rawContent?: string | null;
  durationEstimate?: number | null;
  sortOrder?: number;
  version: number;
}

export interface SceneCreateReq {
  episodeId: number;
  parentVersion: number;
  insertAfterId?: number;
  sceneNumber?: string | null;
  sceneHeading?: string | null;
  location?: string | null;
  timeOfDay?: string | null;
  intExt?: string | null;
  sceneDescription?: string | null;
  sortOrder?: number;
}

export interface SceneUpdateReq {
  id: number;
  sceneNumber?: string | null;
  sceneHeading?: string | null;
  location?: string | null;
  timeOfDay?: string | null;
  intExt?: string | null;
  characters?: string[] | null;
  characterAssetIds?: number[] | null;
  sceneAssetId?: number | null;
  propAssetIds?: number[] | null;
  sceneDescription?: string | null;
  dialogues?: DialogueElement[] | null;
  sortOrder?: number;
  version: number;
}

// ========== API ==========

export const scriptApi = {
  /** 按项目获取唯一剧本 */
  getByProject: (projectId: number) =>
    http.get<never, Script | null>(`/api/script/project/${projectId}`),

  /** 获取剧本详情 */
  get: (id: number) => http.get<never, Script>(`/api/script/${id}`),

  /** 更新剧本 */
  update: ({ id, version, ...patch }: ScriptUpdateReq, requestId: string = crypto.randomUUID()) =>
    http.put<never, Script>("/api/script", { id, version, requestId, patch }),

  /** 替换原文并重置结构化分集与场次 */
  replaceSource: (id: number, version: number, rawContent: string, requestId: string = crypto.randomUUID()) =>
    http.put<never, Script>(`/api/script/${id}/source`, { version, requestId, rawContent }),

  // ========== 分集 ==========

  /** 获取分集列表 */
  listEpisodes: (scriptId: number) =>
    http.get<never, ScriptEpisode[]>(`/api/script/${scriptId}/episodes`),

  /** 获取分集详情 */
  getEpisode: (id: number) =>
    http.get<never, ScriptEpisode>(`/api/script/episode/${id}`),

  /** 创建分集 */
  createEpisode: ({ scriptId: parentId, parentVersion, ...fields }: EpisodeCreateReq, requestId: string = crypto.randomUUID()) =>
    http.post<never, ScriptEpisode>("/api/script/episode", { parentId, parentVersion, requestId, fields }),

  /** 更新分集 */
  updateEpisode: ({ id, version, ...patch }: EpisodeUpdateReq, requestId: string = crypto.randomUUID()) =>
    http.put<never, ScriptEpisode>("/api/script/episode", { id, version, requestId, patch }),

  /** 删除分集 */
  deleteEpisode: (id: number, version: number, requestId: string = crypto.randomUUID()) =>
    http.delete<never, boolean>(`/api/script/episode/${id}`, { params: { version, requestId } }),

  // ========== 场次 ==========

  /** 获取场次列表（按分集） */
  listScenes: (episodeId: number) =>
    http.get<never, SceneItem[]>(`/api/script/episode/${episodeId}/scenes`),

  /** 获取场次详情 */
  getScene: (id: number) => http.get<never, SceneItem>(`/api/script/scene/${id}`),

  /** 创建场次 */
  createScene: ({ episodeId: parentId, parentVersion, ...fields }: SceneCreateReq, requestId: string = crypto.randomUUID()) =>
    http.post<never, SceneItem>("/api/script/scene", { parentId, parentVersion, requestId, fields }),

  /** 更新场次 */
  updateScene: ({ id, version, ...patch }: SceneUpdateReq, requestId: string = crypto.randomUUID()) =>
    http.put<never, SceneItem>("/api/script/scene", { id, version, requestId, patch }),

  /** 删除场次 */
  deleteScene: (id: number, version: number, requestId: string = crypto.randomUUID()) =>
    http.delete<never, boolean>(`/api/script/scene/${id}`, { params: { version, requestId } }),

  reorderScenes: (resourceId: number, expectedVersion: number, sceneIds: number[], requestId: string = crypto.randomUUID()) =>
    http.post<never, { episode: ScriptEpisode; scenes: SceneItem[] }>("/api/script/scenes/commands", {
      requestId, resourceId, expectedVersion, mode: "REORDER", scenes: [], sceneIds,
    }),
};
