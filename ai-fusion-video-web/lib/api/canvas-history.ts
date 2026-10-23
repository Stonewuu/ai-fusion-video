import { http } from "./client";
import type { ResourceType } from "./resource-events";
import type { ResourceReceipt } from "./canvas";

export interface BusinessHistorySummary {
  commandId: string;
  actorUserId: number;
  actorName: string;
  origin: string;
  operation: string;
  resources: {
    resourceType: ResourceType;
    resourceId: number;
    version: number;
    deleted: boolean;
  }[];
  committedAt: string;
  snapshotSchemaVersion: number;
}
export interface BusinessRevision {
  commandId: string;
  resourceType: ResourceType;
  resourceId: number;
  projectId: number | null;
  version: number;
  before: Record<string, unknown> | null;
  after: Record<string, unknown>;
  fields: string[];
}
export interface BusinessHistoryDetail {
  summary: BusinessHistorySummary;
  revisions: BusinessRevision[];
  originalRequest: Record<string, unknown>;
  originalPrevious: Record<string, unknown>;
  agentRunId: string | null;
  toolCallId: string | null;
}
export interface BusinessRestorePlan {
  commandId: string;
  resourceType: ResourceType;
  digest: string;
  canRestore: boolean;
  notices: string[];
  changes: {
    resourceType: ResourceType;
    resourceId: number;
    currentVersion: number;
    projectId: number | null;
    current: Record<string, unknown>;
    proposed: Record<string, unknown>;
    fields: string[];
    removesCreatedObject: boolean;
  }[];
  conflicts: {
    resourceType: ResourceType;
    resourceId: number;
    field: string;
    message: string;
  }[];
}
const path = (id: string) => `/api/canvases/${id}/resource-history`;
export const canvasHistoryApi = {
  list: (
    id: string,
    before?: string,
    signal?: AbortSignal,
  ): Promise<{ items: BusinessHistorySummary[]; nextCursor: string | null }> =>
    http.get(path(id), { params: { before, limit: 40 }, signal }),
  detail: (
    id: string,
    commandId: string,
    signal?: AbortSignal,
  ): Promise<BusinessHistoryDetail> =>
    http.get(`${path(id)}/${commandId}`, { signal }),
  preview: (id: string, commandId: string): Promise<BusinessRestorePlan> =>
    http.get(`${path(id)}/${commandId}/restore-preview`),
  restore: (
    id: string,
    plan: BusinessRestorePlan,
    requestId: string,
  ): Promise<ResourceReceipt> =>
    http.post(`${path(id)}/restore`, {
      requestId,
      commandId: plan.commandId,
      resourceType: plan.resourceType,
      expectedDigest: plan.digest,
    }),
};
