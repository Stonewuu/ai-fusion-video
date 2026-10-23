import { http } from "./client";
import type { Script } from "./script";
import type { Storyboard, StoryboardStatistics } from "./storyboard";

// ========== 类型定义 ==========

export interface Project {
  id: number;
  version: number;
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
  properties?: Record<string, unknown>;
  artStyle?: string;
}

export interface ProjectWorkspaceOverview {
  projectVersion: number;
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
  initializeWorkspace: (id: number, version: number, requestId = crypto.randomUUID()) =>
    http.post<never, ProjectWorkspaceOverview>(`/api/project/${id}/workspace/initialize`, { version, requestId }),

  /** 创建项目 */
  create: (fields: ProjectCreateReq, requestId = crypto.randomUUID()) =>
    http.post<never, Project>("/api/project", { requestId, fields }),

  /** 更新项目 */
  update: (
    data: Partial<Pick<Project, "name" | "description" | "coverUrl" | "status" | "properties" | "artStyle" | "artStyleDescription" | "artStyleImagePrompt" | "artStyleImageUrl">>
      & { id: number; version: number },
    requestId = crypto.randomUUID(),
  ) => {
    const { id, version, ...patch } = data;
    return http.put<never, Project>("/api/project", { id, version, requestId, patch });
  },

  /** 按读取版本原子更新项目扩展属性 */
  updateProperties: (id: number, version: number, properties: Record<string, unknown>, requestId = crypto.randomUUID()) =>
    http.put<never, Project>("/api/project", {
      id,
      version,
      requestId,
      patch: { properties },
    }),

  /** 删除项目 */
  delete: (id: number, version: number, requestId = crypto.randomUUID()) =>
    http.delete<never, boolean>(`/api/project/${id}`, { params: { version, requestId } }),
};
