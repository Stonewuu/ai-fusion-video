import { http } from "./client";

export interface MediaPreview {
  sourceId: string;
  sourceVersion: number;
  profile: "canvas_640_v1";
  jobId: string | null;
  status:
    | "ABSENT"
    | "QUEUED"
    | "WAITING"
    | "RUNNING"
    | "SUCCEEDED"
    | "FAILED"
    | "CANCELLED";
  error: string | null;
  previewId: string | null;
  width: number | null;
  height: number | null;
  sampleAtMs: number;
  canRetry: boolean;
}
export const mediaPreviewApi = {
  get: (id: string, signal: AbortSignal) =>
    http.get<never, MediaPreview>(
      `/api/media-resources/${encodeURIComponent(id)}/preview`,
      { signal },
    ),
  ensure: (id: string, retryJobId: string | null, signal: AbortSignal) =>
    http.post<never, MediaPreview>(
      `/api/media-resources/${encodeURIComponent(id)}/preview`,
      { retryJobId },
      { signal },
    ),
};
