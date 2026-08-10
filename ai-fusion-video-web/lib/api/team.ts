import { http } from "./client";

// ========== 类型定义 ==========

/** 团队 */
export interface Team {
  id: number;
  name: string;
  logo: string | null;
  description: string | null;
  ownerUserId: number;
  status: number;
  memberCount?: number;
  createTime: string;
  updateTime: string;
}

/** 团队成员 */
export interface TeamMember {
  id: number;
  teamId: number;
  userId: number;
  role: number;
  status: number;
  joinTime: string | null;
}

/** 创建团队请求 */
export interface TeamCreateReq {
  name: string;
  description?: string;
}

/** 更新团队请求 */
export interface TeamUpdateReq {
  id: number;
  name?: string;
  description?: string;
  logo?: string;
  status?: number;
}

/** 添加成员请求 */
export interface TeamMemberAddReq {
  teamId: number;
  userId: number;
  role: number;
}

// ========== API ==========

export const teamApi = {
  // ========== 团队 ==========

  /** 获取团队详情 */
  get: (id: number) => http.get<never, Team>(`/api/team/get?id=${id}`),

  /** 创建团队 */
  create: (data: TeamCreateReq) =>
    http.post<never, number>("/api/team/create", data),

  /** 更新团队 */
  update: (data: TeamUpdateReq) =>
    http.put<never, boolean>("/api/team/update", data),

  /** 删除团队 */
  delete: (id: number) =>
    http.delete<never, boolean>(`/api/team/delete?id=${id}`),

  // ========== 成员管理 ==========

  /** 获取团队成员列表 */
  listMembers: (teamId: number) =>
    http.get<never, TeamMember[]>(`/api/team/member/list?teamId=${teamId}`),

  /** 添加团队成员 */
  addMember: (data: TeamMemberAddReq) =>
    http.post<never, number>("/api/team/member/add", data),

  /** 移除团队成员 */
  removeMember: (teamId: number, userId: number) =>
    http.post<never, boolean>(`/api/team/member/remove?teamId=${teamId}&userId=${userId}`),

  /** 变更成员角色 */
  changeMemberRole: (teamId: number, userId: number, role: number) =>
    http.post<never, boolean>(`/api/team/member/change-role?teamId=${teamId}&userId=${userId}&role=${role}`),
};
