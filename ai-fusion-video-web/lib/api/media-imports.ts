import { http } from "./client";
import type { GenerationScope } from "./generation-execution";
import type { ManagedMedia } from "./media";
import type { CanvasValueType, CanvasReceipt } from "@/lib/canvas/types";

export interface ImportSource {
  title: string;
  source: { type: CanvasValueType; id: number; version: number; field: string };
}
export interface ImportPoint {
  x: number;
  y: number;
}
export interface ImportRequest {
  requestId: string;
  scope: GenerationScope;
  sources: ImportSource[];
  position: ImportPoint | null;
}
export type ImportStatus =
  | "QUEUED"
  | "WAITING"
  | "RUNNING"
  | "SUCCEEDED"
  | "PARTIAL_FAILED"
  | "FAILED"
  | "CANCELLED";
export interface ImportItem {
  index: number;
  title: string;
  source: ImportSource["source"];
  nodeId: string;
  status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";
  attempt: number;
  errorCode: string | null;
  message: string | null;
  commandId: string | null;
  resource: ManagedMedia | null;
}
export interface MediaImport {
  id: string;
  canvasId: string;
  requestId: string;
  status: ImportStatus;
  revision: number;
  cancelRequested: boolean;
  automaticPlacement: boolean;
  canControl: boolean;
  phase: string;
  items: ImportItem[];
  createdAt: string;
  updatedAt: string;
}
export interface ImportSummary {
  id: string;
  status: ImportStatus;
  revision: number;
  cancelRequested: boolean;
  automaticPlacement: boolean;
  phase: string;
  total: number;
  succeeded: number;
  failed: number;
  completed: number;
  readyToPlace: number;
  createdAt: string;
}
const path = (canvasId: string) =>
  `/api/canvases/${encodeURIComponent(canvasId)}/media-imports`;
const jobPath = (id: string) => `/api/media-imports/${encodeURIComponent(id)}`;
export const mediaImportApi = {
  submit: (canvasId: string, request: ImportRequest) =>
    http.post<never, MediaImport>(path(canvasId), request),
  list: (canvasId: string, page: number, signal?: AbortSignal) =>
    http.get<never, { items: ImportSummary[]; page: number; hasMore: boolean }>(
      path(canvasId),
      { params: { page }, signal },
    ),
  active: (canvasId: string, signal?: AbortSignal) =>
    http.get<never, ImportSummary[]>(`${path(canvasId)}/active`, { signal }),
  get: (id: string, signal?: AbortSignal) =>
    http.get<never, MediaImport>(jobPath(id), { signal }),
  withdraw: (canvasId: string, requestId: string, scope: GenerationScope) =>
    http.post<never, { accepted: boolean; job: MediaImport | null }>(
      `${path(canvasId)}/requests/${encodeURIComponent(requestId)}/withdraw`,
      { scope },
    ),
  cancel: (id: string, scope: GenerationScope) =>
    http.post<never, MediaImport>(`${jobPath(id)}/cancel`, { scope }),
  retry: (job: MediaImport, indices: number[], scope: GenerationScope) =>
    http.post<never, MediaImport>(`${jobPath(job.id)}/retry`, {
      scope,
      expectedRevision: job.revision,
      indices,
    }),
  place: (
    id: string,
    scope: GenerationScope,
    automatic: boolean,
    position: ImportPoint | null = null,
  ) =>
    http.post<never, { job: MediaImport; command: CanvasReceipt | null }>(
      `${jobPath(id)}/place`,
      {
        scope,
        automatic,
        position,
      },
    ),
};
