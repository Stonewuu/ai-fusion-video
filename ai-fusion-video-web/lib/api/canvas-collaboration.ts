import { http } from "./client";
import type { CanvasReceipt, CanvasSnapshot } from "@/lib/canvas/types";

export interface CanvasPerson {
  userId: number;
  username: string;
  name: string;
}
export interface CanvasMember {
  person: CanvasPerson;
  effective: CanvasSnapshot["permissions"];
  granted: CanvasSnapshot["permissions"] | null;
  source: "OWNER" | "CANVAS" | "PROJECT";
  protectedOwner: boolean;
}
export interface CanvasGrant {
  requestId: string;
  expectedRevision: number;
  userId: number;
  canEdit: boolean;
  canExecute: boolean;
  canManage: boolean;
  remove: boolean;
}
export interface CanvasPresenceSnapshot {
  participants: {
    sessionId: string;
    person: CanvasPerson;
    nodeIds: string[];
    updatedAt: number;
  }[];
  hasMore: boolean;
  permissions: CanvasSnapshot["permissions"];
}
export const canvasCollaborationApi = {
  heartbeat: (
    id: string,
    sessionId: string,
    nodeIds: string[],
    signal?: AbortSignal,
  ): Promise<CanvasPresenceSnapshot> =>
    http.post(
      `/api/canvases/${id}/presence`,
      { sessionId, nodeIds },
      { signal },
    ),
  leave: (id: string, sessionId: string): Promise<boolean> =>
    http.delete(`/api/canvases/${id}/presence/${sessionId}`),
  members: (
    id: string,
    page = 1,
    signal?: AbortSignal,
  ): Promise<{ list: CanvasMember[]; total: number }> =>
    http.get(`/api/canvases/${id}/members`, {
      params: { page, size: 40 },
      signal,
    }),
  find: (id: string, username: string): Promise<CanvasMember> =>
    http.get(`/api/canvases/${id}/members/find`, { params: { username } }),
  grant: (id: string, body: CanvasGrant): Promise<CanvasReceipt> =>
    http.post(`/api/canvases/${id}/members`, body),
};
