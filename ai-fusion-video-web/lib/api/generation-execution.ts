import { http } from "./client";

export interface GenerationScope {
  userId: number;
  teamId: number | null;
}
export type GenerationState =
  | "QUEUED"
  | "RUNNING"
  | "CANCEL_REQUESTED"
  | "UNKNOWN_SUBMISSION"
  | "NEEDS_REVIEW"
  | "SUCCEEDED"
  | "PARTIAL_FAILED"
  | "FAILED"
  | "CANCELLED";
export type GenerationReviewAction =
  | "RECHECK"
  | "RETRY_INGEST"
  | "RESTORE_ACCEPTED_RECEIPT"
  | "CONFIRM_NOT_ACCEPTED_AND_RETRY"
  | "CONFIRM_PROVIDER_FINISHED";
export interface GenerationExecutionSummary {
  id: string;
  state: GenerationState;
  cancelRequested: boolean;
  canControl: boolean;
}
export interface GenerationExecutionItem {
  id: string;
  itemIds: number[];
  index: number;
  state: string;
  phase: string;
  providerTaskId: string | null;
  cancelRequested: boolean;
  cancelSent: boolean;
  remoteCapacityHeld: boolean;
  errorCode: string | null;
  errorMessage: string | null;
  revision: number;
  deadline: string;
  resultReceived: boolean;
  availableActions: GenerationReviewAction[];
  attempts: {
    id: string;
    number: number;
    providerTaskId: string | null;
    submitted: boolean;
    acknowledged: boolean;
    resolution: string | null;
  }[];
}
export interface GenerationExecution extends GenerationExecutionSummary {
  kind: "IMAGE" | "VIDEO";
  recordId: number;
  taskId: string;
  revision: number;
  modelId: number;
  modelName: string;
  createdAt: string;
  items: GenerationExecutionItem[];
}
export const generationExecutionApi = {
  scope: () =>
    http.get<never, GenerationScope>("/api/generation/executions/scope"),
  get: (id: string, signal?: AbortSignal) =>
    http.get<never, GenerationExecution>(
      `/api/generation/executions/${encodeURIComponent(id)}`,
      { signal },
    ),
  withdraw: (
    requestId: string,
    kind: "IMAGE" | "VIDEO",
    scope: GenerationScope,
  ) =>
    http.post<
      never,
      {
        submissionId: string | null;
        taskId: string | null;
        state: GenerationState | "CANCELLED_BEFORE_ACCEPTANCE";
      }
    >(
      `/api/generation/executions/requests/${encodeURIComponent(requestId)}/withdraw`,
      { kind, scope },
    ),
  review: (
    id: string,
    body: {
      expectedRevision: number;
      action: GenerationReviewAction;
      note: string;
    },
  ) =>
    http.post<never, boolean>(
      `/api/generation/executions/jobs/${encodeURIComponent(id)}/review`,
      body,
    ),
};
