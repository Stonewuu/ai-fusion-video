import { http } from "./client";
import type { Script } from "./script";
import type { Storyboard, StoryboardStatistics } from "./storyboard";

// ========== 类型定义 ==========

export interface Project {
  id: number;
  name: string;
  description: string;
  coverUrl: string | null;
  scope: number;
  ownerType: number;
  ownerId: number;
  status: number;
  properties: Record<string, unknown> | null;
  artStyle: string | null;
  artStyleDescription: string | null;
  artStyleImagePrompt: string | null;
  artStyleImageUrl: string | null;
  createTime: string;
  updateTime: string;
}

export interface ProjectCreateReq {
  name: string;
  description?: string;
  properties?: string;
}

export interface ProjectWorkspaceOverview {
  script: Script | null;
  scriptEpisodeCount: number;
  scriptSceneCount: number;
  storyboard: Storyboard | null;
  storyboardStatistics: StoryboardStatistics;
}

// ========== API ==========

export const projectApi = {
  /** 获取当前用户的项目列表 */
  list: () => http.get<never, Project[]>("/api/project/list"),

  /** 获取项目详情 */
  get: (id: number) => http.get<never, Project>(`/api/project/${id}`),

  /** 获取项目固定剧本与分镜工作区概览 */
  getWorkspaceOverview: (id: number) =>
    http.get<never, ProjectWorkspaceOverview>(`/api/project/${id}/workspace-overview`),

  /** 幂等补齐旧项目缺失的剧本与分镜容器，不修改已有数据 */
  initializeWorkspace: (id: number) =>
    http.post<never, ProjectWorkspaceOverview>(`/api/project/${id}/workspace/initialize`),

  /** 创建项目 */
  create: (data: ProjectCreateReq) => http.post<never, Project>("/api/project", data),

  /** 更新项目 */
  update: (data: Partial<Project> & { id: number }) =>
    http.put<never, Project>("/api/project", data),

  /** 更新项目扩展属性（properties 传对象，自动序列化为 JSON 字符串） */
  updateProperties: (id: number, properties: Record<string, unknown>) =>
    http.put<never, Project>("/api/project", {
      id,
      properties: JSON.stringify(properties),
    }),

  /** 删除项目 */
  delete: (id: number) => http.delete<never, boolean>(`/api/project/${id}`),
};
