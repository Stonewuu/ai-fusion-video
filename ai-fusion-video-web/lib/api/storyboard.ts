import { http } from "./client";

// ========== 类型定义 ==========

/** 分镜脚本 */
export interface Storyboard {
  id: number;
  version: number;
  projectId: number;
  scriptId: number | null;
  title: string;
  description: string | null;
  customColumns: Record<string, unknown>[] | null;
  scope: number;
  ownerType: number;
  ownerId: number;
  totalDuration: number | null;
  status: number;
  createTime: string;
  updateTime: string;
}

/** 分镜概览统计 */
export interface StoryboardStatistics {
  episodeCount: number;
  sceneCount: number;
  itemCount: number;
}

/** 分镜集合成状态: 0未开始 1合成中 2已完成 3失败 */
export type EpisodeComposeStatus = 0 | 1 | 2 | 3;

/** 分镜集 */
export interface StoryboardEpisode {
  storyboardVersion: number;
  id: number;
  version: number;
  storyboardId: number;
  scriptEpisodeId: number | null;
  episodeNumber: number | null;
  title: string | null;
  synopsis: string | null;
  sortOrder: number;
  status: number;
  composedVideoUrl: string | null;
  composeStatus: EpisodeComposeStatus;
  composeJobId?: string | null;
  composeErrorMsg: string | null;
  composedAt: string | null;
  createTime: string;
  updateTime: string;
}

/** 分镜场次 */
export interface StoryboardScene {
  storyboardVersion: number;
  episodeVersion: number;
  id: number;
  version: number;
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
  storyboardVersion: number;
  episodeVersion?: number;
  sceneVersion?: number;
  id: number;
  version: number;
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
  characterIds: number[] | null;
  sceneAssetItemId: number | null;
  propIds: number[] | null;
  remark: string | null;
  customData: Record<string, unknown> | null;
  aiGenerated: boolean;
  status: number;
  createTime: string;
  updateTime: string;
}

/** 可修改字段与读模型共用原生 JSON 类型；集合排序单独提交。 */
export type StoryboardUpdateReq = { id: number; version: number } & Partial<Pick<Storyboard, "description" | "customColumns" | "status">>;
export interface StoryboardEpisodeCreateReq {
  storyboardId: number;
  parentVersion: number;
  scriptEpisodeId?: number | null;
  episodeNumber: number;
  title?: string | null;
  synopsis?: string | null;
  sortOrder?: number;
}
export type StoryboardEpisodeUpdateReq = { id: number; version: number } &
  Partial<Pick<StoryboardEpisode, "scriptEpisodeId" | "episodeNumber" | "title" | "synopsis" | "status">>;
export interface StoryboardSceneCreateReq {
  episodeId: number;
  parentVersion: number;
  sceneNumber?: string;
  sceneHeading?: string;
  location?: string;
  timeOfDay?: string;
  intExt?: string;
  sortOrder?: number;
}
export type StoryboardSceneUpdateReq = { id: number; version: number } &
  Partial<Pick<StoryboardScene, "sceneNumber" | "sceneHeading" | "location" | "timeOfDay" | "intExt" | "status">>;
export type StoryboardItemFields = Pick<StoryboardItem,
  "shotNumber" | "shotType" | "duration" | "content" | "sceneExpectation" | "sound" | "dialogue" |
  "soundEffect" | "music" | "cameraMovement" | "cameraAngle" | "cameraEquipment" | "focalLength" |
  "transition" | "characterIds" | "sceneAssetItemId" | "propIds" | "remark" | "customData" |
  "aiGenerated" | "status" | "imageUrl" | "referenceImageUrl" | "videoUrl" | "generatedImageUrl" |
  "firstFrameImageUrl" | "lastFrameImageUrl" | "firstFramePrompt" | "lastFramePrompt" | "generatedVideoUrl" | "videoPrompt">;
export type StoryboardItemCreateReq = Partial<StoryboardItemFields> & {
  storyboardSceneId: number;
  parentVersion: number;
  sortOrder?: number;
  insertAfterId?: number;
};
export type StoryboardItemUpdateReq = { id: number; version: number } & Partial<StoryboardItemFields>;
export interface StoryboardItemAssetsUpdateReq {
  version: number;
  characterIds?: number[];
  sceneAssetItemId?: number | null;
  propIds?: number[];
}
export type StoryboardFrameType = "first" | "last";
export interface StoryboardFrameUpdateReq {
  version: number;
  frameType: StoryboardFrameType;
  imageUrl?: string | null;
  prompt?: string | null;
}
export interface StoryboardChildrenResult<P, C> {
  parent: P;
  children: C[];
  mode: "APPEND" | "REPLACE" | "CLEAR" | "DELETE" | "REORDER";
}
export interface StoryboardBatchPatchEntry {
  resourceType: "storyboard" | "storyboardEpisode" | "storyboardScene" | "shot";
  resourceId: number;
  expectedVersion: number;
  patch: Record<string, unknown>;
}

function patch<T>(path: string, data: { id: number; version: number }, requestId: string) {
  const { id, version, ...fields } = data;
  return http.put<never, T>(path, { id, version, requestId, patch: fields });
}
function remove(path: string, version: number, requestId: string) {
  return http.delete<never, boolean>(path, { params: { version, requestId } });
}
function create<T>(path: string, parentId: number, parentVersion: number, fields: unknown, requestId: string) {
  return http.post<never, T>(path, { parentId, parentVersion, requestId, fields });
}

export const storyboardApi = {
  get: (id: number) => http.get<never, Storyboard>("/api/storyboard/" + id),
  getByProject: (projectId: number) => http.get<never, Storyboard | null>("/api/storyboard/project/" + projectId),
  update: (data: StoryboardUpdateReq, requestId: string = crypto.randomUUID()) => patch<Storyboard>("/api/storyboard", data, requestId),
  clearContent: (id: number, version: number, requestId: string = crypto.randomUUID()) =>
    http.post<never, Storyboard>("/api/storyboard/" + id + "/clearContent", { version, requestId }),
  getStatistics: (storyboardId: number) => http.get<never, StoryboardStatistics>("/api/storyboard/" + storyboardId + "/statistics"),

  listEpisodes: (storyboardId: number) => http.get<never, StoryboardEpisode[]>("/api/storyboard/" + storyboardId + "/episodes"),
  getEpisode: (id: number) => http.get<never, StoryboardEpisode>("/api/storyboard/episode/" + id),
  createEpisode: (data: StoryboardEpisodeCreateReq, requestId: string = crypto.randomUUID()) => {
    const { storyboardId, parentVersion, ...fields } = data;
    return create<StoryboardEpisode>("/api/storyboard/episode", storyboardId, parentVersion, fields, requestId);
  },
  updateEpisode: (data: StoryboardEpisodeUpdateReq, requestId: string = crypto.randomUUID()) => patch<StoryboardEpisode>("/api/storyboard/episode", data, requestId),
  deleteEpisode: (id: number, version: number, requestId: string = crypto.randomUUID()) => remove("/api/storyboard/episode/" + id, version, requestId),
  bindScriptEpisode: (id: number, version: number, scriptEpisodeId: number | null, requestId: string = crypto.randomUUID()) =>
    http.put<never, StoryboardEpisode>("/api/storyboard/episode/" + id + "/bindScriptEpisode", { version, requestId, patch: { scriptEpisodeId } }),
  clearEpisodeContent: (id: number, version: number, requestId: string = crypto.randomUUID()) =>
    http.post<never, StoryboardEpisode>("/api/storyboard/episode/" + id + "/clearContent", { version, requestId }),
  composeEpisodeVideo: (episodeId: number, version: number, requestId: string, includeBgm: boolean) => http.post<never, string>("/api/storyboard/episode/" + episodeId + "/compose-video", { version, requestId, includeBgm }),

  listScenesByEpisode: (episodeId: number) => http.get<never, StoryboardScene[]>("/api/storyboard/episode/" + episodeId + "/scenes"),
  listScenesByStoryboard: (storyboardId: number) => http.get<never, StoryboardScene[]>("/api/storyboard/" + storyboardId + "/scenes"),
  getScene: (id: number) => http.get<never, StoryboardScene>("/api/storyboard/scene/" + id),
  createScene: (data: StoryboardSceneCreateReq, requestId: string = crypto.randomUUID()) => {
    const { episodeId, parentVersion, ...fields } = data;
    return create<StoryboardScene>("/api/storyboard/scene", episodeId, parentVersion, fields, requestId);
  },
  updateScene: (data: StoryboardSceneUpdateReq, requestId: string = crypto.randomUUID()) => patch<StoryboardScene>("/api/storyboard/scene", data, requestId),
  deleteScene: (id: number, version: number, requestId: string = crypto.randomUUID()) => remove("/api/storyboard/scene/" + id, version, requestId),

  listItems: (storyboardId: number) => http.get<never, StoryboardItem[]>("/api/storyboard/" + storyboardId + "/items"),
  listItemsByScene: (sceneId: number) => http.get<never, StoryboardItem[]>("/api/storyboard/scene/" + sceneId + "/items"),
  getItem: (id: number) => http.get<never, StoryboardItem>("/api/storyboard/item/" + id),
  createItem: (data: StoryboardItemCreateReq, requestId: string = crypto.randomUUID()) => {
    const { storyboardSceneId, parentVersion, ...fields } = data;
    return create<StoryboardItem>("/api/storyboard/item", storyboardSceneId, parentVersion, fields, requestId);
  },
  updateItem: (data: StoryboardItemUpdateReq, requestId: string = crypto.randomUUID()) => patch<StoryboardItem>("/api/storyboard/item", data, requestId),
  updateItemAssets: (id: number, data: StoryboardItemAssetsUpdateReq, requestId: string = crypto.randomUUID()) => {
    const { version, ...fields } = data;
    return http.patch<never, StoryboardItem>("/api/storyboard/item/" + id + "/assets", { version, requestId, patch: fields });
  },
  updateFrame: (id: number, data: StoryboardFrameUpdateReq, requestId: string = crypto.randomUUID()) => {
    const { version, ...fields } = data;
    return http.put<never, StoryboardItem>("/api/storyboard/item/" + id + "/updateFrame", { version, requestId, patch: fields });
  },
  deleteItem: (id: number, version: number, requestId: string = crypto.randomUUID()) => remove("/api/storyboard/item/" + id, version, requestId),
  reorderScenes: (episodeId: number, version: number, ids: number[], requestId: string = crypto.randomUUID()) =>
    http.post<never, StoryboardChildrenResult<StoryboardEpisode, StoryboardScene>>("/api/storyboard/children", {
      requestId, resourceType: "storyboardEpisode", resourceId: episodeId, expectedVersion: version,
      mode: "REORDER", children: [], childIds: ids,
    }),
  reorderItems: (sceneId: number, version: number, ids: number[], requestId: string = crypto.randomUUID()) =>
    http.post<never, StoryboardChildrenResult<StoryboardScene, StoryboardItem>>("/api/storyboard/children", {
      requestId, resourceType: "storyboardScene", resourceId: sceneId, expectedVersion: version,
      mode: "REORDER", children: [], childIds: ids,
    }),
  batchCreateItems: (sceneId: number, version: number, items: Partial<StoryboardItemFields>[], requestId: string = crypto.randomUUID()) =>
    http.post<never, StoryboardChildrenResult<StoryboardScene, StoryboardItem>>("/api/storyboard/children", {
      requestId, resourceType: "storyboardScene", resourceId: sceneId, expectedVersion: version,
      mode: "APPEND", children: items, childIds: [],
    }),
  batchPatch: (storyboardId: number, entries: StoryboardBatchPatchEntry[], requestId: string = crypto.randomUUID()) =>
    http.post<never, { storyboardId: number; storyboardVersion: number; resources: { resourceType: string; resourceId: number; data: Storyboard | StoryboardEpisode | StoryboardScene | StoryboardItem }[] }>(
      "/api/storyboard/batch", { requestId, storyboardId, entries }),
};
