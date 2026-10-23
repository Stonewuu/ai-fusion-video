import { http } from "./client";
import type { CanvasPerson } from "./canvas-collaboration";

export interface ProjectMember {
  person: CanvasPerson;
  role: number | null;
  protectedOwner: boolean;
  enabled: boolean;
}
export interface ProjectMembers {
  revision: number;
  canManage: boolean;
  teamOwned: boolean;
  list: ProjectMember[];
  total: number;
}
export interface ProjectMemberChange {
  commandId: string;
  projectId: number;
  revision: number;
  userId: number;
  previousRole: number | null;
  currentRole: number | null;
  removedCanvasGrants: number;
  createdAt: string;
}
export interface ProjectMemberAudit {
  change: ProjectMemberChange;
  actor: CanvasPerson;
  member: CanvasPerson;
}
export const projectMembersApi = {
  list: (id: number, page = 1, signal?: AbortSignal) =>
    http.get<never, ProjectMembers>(`/api/project/${id}/members`, { params: { page }, signal }),
  find: (id: number, username: string) =>
    http.get<never, ProjectMember>(`/api/project/${id}/members/find`, { params: { username } }),
  change: (id: number, command: { requestId: string; expectedRevision: number; userId: number; role: number | null }) =>
    http.post<never, ProjectMemberChange>(`/api/project/${id}/members`, command),
  history: (id: number, page = 1, signal?: AbortSignal) =>
    http.get<never, { list: ProjectMemberAudit[]; total: number }>(`/api/project/${id}/members/history`, { params: { page }, signal }),
};
export const projectRoleLabel = (role: number | null) =>
  role === 1 ? "拥有者" : role === 2 ? "可管理" : role === 3 ? "可编辑" : role === 4 ? "可查看" : "无单独授权";
