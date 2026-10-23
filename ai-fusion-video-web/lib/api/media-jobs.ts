import { http } from "./client";
import type { GenerationScope } from "./generation-execution";

export interface MediaJob {
  id: string;
  kind: string;
  projectId: number;
  targetId: number;
  status: "QUEUED" | "RUNNING" | "WAITING" | "SUCCEEDED" | "FAILED" | "CANCELLED" | "CONFLICT" | "NEEDS_REVIEW";
  phase: string;
  error: string | null;
  results: { video?: string; baseVideo?: string; music?: string; sfx?: string };
  revision: number;
  createdAt: string;
  cancelRequested: boolean;
}
export type EpisodeAudioReviewAction = "RECHECK" | "RECORD_RECEIPT" | "RETRY_INGEST" | "CONFIRM_FINISHED";
export interface EpisodeAudioReview {
  jobId: string;
  revision: number;
  factsHash: string;
  status: MediaJob["status"];
  cancelRequested: boolean;
  capacityHeld: boolean;
  steps: { step: "music" | "sfx"; providerTaskId: string | null; submittedAt: string; unresolved: boolean }[];
  actions: EpisodeAudioReviewAction[];
  lastReview: { action: EpisodeAudioReviewAction; note: string; recordedAt: string } | null;
}
export const mediaJobRunning = (job: MediaJob) => ["QUEUED", "RUNNING", "WAITING"].includes(job.status);
export const mediaJobsApi = {
  get: (id: string, signal?: AbortSignal): Promise<MediaJob> => http.get(`/api/media-jobs/${id}`, { signal }),
  list: (projectId: number, episodeId: number): Promise<MediaJob[]> => http.get("/api/media-jobs", { params: { projectId, episodeId } }),
  cancel: (id: string): Promise<boolean> => http.post(`/api/media-jobs/${id}/cancel`),
  audio: (id: string, signal?: AbortSignal): Promise<EpisodeAudioReview> => http.get(`/api/media-jobs/${id}/audio`, { signal }),
  reviewAudio: (id: string, request: {
    expectedRevision: number; expectedFactsHash: string; action: EpisodeAudioReviewAction; note: string;
    step: string | null; providerTaskId: string | null; confirmedEnded: boolean; scope: GenerationScope;
  }): Promise<EpisodeAudioReview> => http.post(`/api/media-jobs/${id}/audio/review`, request),
};
