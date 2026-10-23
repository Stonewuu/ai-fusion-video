import { http } from "./client";
import type { GenerationScope } from "./generation-execution";
import type {
  CanvasResourceRef,
  CanvasValueType,
  CanvasWritebackField,
} from "@/lib/canvas/types";
import type { ManagedMedia } from "./media";

export type CanvasRunMode =
  | "SINGLE"
  | "SELECTION"
  | "DEPENDENCIES"
  | "ALL"
  | "STALE";
export type CanvasRunState =
  | "QUEUED"
  | "WAITING_INPUT"
  | "READY"
  | "RUNNING"
  | "WAITING_SELECTION"
  | "WAITING_CONFIRMATION"
  | "UNKNOWN_SUBMISSION"
  | "NEEDS_REVIEW"
  | "CANCEL_REQUESTED"
  | "SUCCEEDED"
  | "PARTIAL_FAILED"
  | "FAILED"
  | "BLOCKED"
  | "CANCELLED";
export interface CanvasRunBudget {
  maxNodes: number;
  maxImages: number;
  maxVideoSeconds: number;
  maxParallel: number;
  maxBatchItems: number;
}
export interface CanvasRunCost {
  nodes: number;
  images: number;
  videoSeconds: number;
  batchItems: number;
  agentRuns: number;
  mcpCalls: number;
  amount: string;
}
export interface CanvasRunPlanRequest {
  expectedRevision: number;
  mode: CanvasRunMode;
  nodeIds: string[];
  reuse: Record<string, string>;
  budget: CanvasRunBudget;
  scope: GenerationScope;
}
export interface CanvasRunPlan {
  id: string;
  canvasId: string;
  documentRevision: number;
  hash: string;
  mode: CanvasRunMode;
  nodes: {
    nodeId: string;
    title: string;
    kind: string;
    action: "EXECUTE" | "REUSE";
    reusedNodeRunId: string | null;
    dependencies: string[];
    imageCount: number;
    videoSeconds: number;
    externalOperation: { name: string; serverName: string; toolName: string; effect: "READ_ONLY" | "WRITE" | "DESTRUCTIVE"; idempotency: string; timeoutSeconds: number; maximumCalls: number } | null;
  }[];
  budget: CanvasRunBudget;
  cost: CanvasRunCost;
  issues: { nodeId: string | null; code: string; message: string }[];
  expiresAt: string;
}
export interface CanvasRunStart {
  requestId: string;
  planId: string;
  planHash: string;
  scope: GenerationScope;
}
export interface CanvasRunValue {
  id: string;
  type: CanvasValueType;
  text: string | null;
  mediaId: string | null;
  mediaVersion: number | null;
  resource: CanvasResourceRef | null;
  data: ManagedMedia | Record<string, unknown> | null;
  sourceNodeRunId: string | null;
}
export interface CanvasRunPorts {
  values: Record<string, CanvasRunValue[]>;
}
export interface CanvasNodeRun {
  id: string;
  nodeId: string;
  parentNodeRunId: string | null;
  itemIndex: number;
  attempt: number;
  writeback: CanvasWritebackPreview | null;
  contentUnavailable: boolean;
  title: string;
  kind: string;
  action: "EXECUTE" | "REUSE";
  status: CanvasRunState;
  error: string | null;
  revision: number;
  inputs: CanvasRunPorts | null;
  outputs: CanvasRunPorts;
  externalKind: string | null;
  externalId: string | null;
  derivedState: "CURRENT" | "STALE" | "NODE_DELETED" | "SOURCE_UNAVAILABLE";
}
export interface CanvasWritebackPreview {
  hash: string;
  field: CanvasWritebackField;
  target: CanvasResourceRef;
  targetTitle: string;
  before: string | null;
  value: CanvasRunValue;
  notice?: string;
  dangerous?: boolean;
  removedResources?: {
    type: CanvasValueType;
    id: number;
    title: string;
    version: number;
  }[];
  decision?: {
    action: "APPROVE" | "SKIP";
    note: string;
    actorUserId: number;
    at: string;
    commandId?: string;
    resourceVersion?: number;
  };
}
export interface CanvasRun {
  id: string;
  canvasId: string;
  documentRevision: number;
  mode: CanvasRunMode;
  status: CanvasRunState;
  cancelRequested: boolean;
  revision: number;
  error: string | null;
  cost: CanvasRunCost;
  createdAt: string;
  updatedAt: string;
  nodes: CanvasNodeRun[];
  canControl: boolean;
  statistics: CanvasRunStatistics;
}
export interface CanvasRunStatistics {
  observedAt: string;
  elapsedMs: number;
  nodes: {
    total: number;
    succeeded: number;
    failed: number;
    blocked: number;
    cancelled: number;
    pending: number;
    reused: number;
    batchAttempts: number;
    batchRetries: number;
  };
  media: {
    images: number;
    videos: number;
    audios: number;
    videoDurationMs: number;
    audioDurationMs: number;
    unknownDurations: number;
    bytes: number;
    unavailableNodes: number;
  };
  tokens: {
    submittedCalls: number;
    measuredCalls: number;
    input: number | null;
    output: number | null;
    total: number | null;
  };
}
export interface CanvasRunSummary {
  id: string;
  status: CanvasRunState;
  mode: CanvasRunMode;
  revision: number;
  createdAt: string;
  nodes: number;
}
export interface CanvasNodeActivity {
  runId: string;
  runStatus: CanvasRunState;
  canControl: boolean;
  node: CanvasNodeRun;
  children: CanvasNodeRun[];
}
export type CanvasMediaReviewAction =
  | "RECHECK"
  | "RETRY_INGEST"
  | "RECORD_RECEIPT"
  | "CONFIRM_FINISHED";
export interface CanvasMediaExecution {
  id: string;
  status: string;
  phase: string;
  error: string | null;
  revision: number;
  providerTaskId: string | null;
  remoteCapacityHeld: boolean;
  cancelRequested: boolean;
  actions: CanvasMediaReviewAction[];
}
export type CanvasTextReviewAction = "USE_RECEIVED" | "CONFIRM_FINISHED";
export interface CanvasTextExecution {
  id: string;
  status: string;
  phase: string;
  error: string | null;
  revision: number;
  receivedText: string | null;
  completeResponse: boolean;
  remoteCapacityHeld: boolean;
  cancelRequested: boolean;
  actions: CanvasTextReviewAction[];
}
const path = (id: string) => `/api/canvases/${encodeURIComponent(id)}`;
export const canvasRunApi = {
  nodeActivities: (id: string, signal?: AbortSignal): Promise<CanvasNodeActivity[]> =>
    http.get(`${path(id)}/node-activities`, { signal }),
  writeback: (
    runId: string,
    nodeId: string,
    request: {
      expectedRevision: number;
      proposalHash: string;
      action: "APPROVE" | "SKIP";
      note: string;
      acceptReset?: boolean;
      scope: GenerationScope;
    },
  ): Promise<CanvasRun> =>
    http.post(
      `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/writeback`,
      request,
    ),
  reviewBatch: (
    runId: string,
    nodeId: string,
    request: {
      expectedRevision: number;
      action: "RETRY_FAILED" | "USE_SUCCEEDED" | "FINISH";
      note: string;
      scope: GenerationScope;
    },
  ): Promise<CanvasRun> =>
    http.post(
      `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/batch/review`,
      request,
    ),
  text: (runId: string, nodeId: string): Promise<CanvasTextExecution> =>
    http.get(
      `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/text`,
    ),
  reviewText: (
    runId: string,
    nodeId: string,
    request: {
      expectedRevision: number;
      action: CanvasTextReviewAction;
      note: string;
      scope: GenerationScope;
    },
  ): Promise<CanvasTextExecution> =>
    http.post(
      `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/text/review`,
      request,
    ),
  media: (runId: string, nodeId: string): Promise<CanvasMediaExecution> =>
    http.get(
      `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/media`,
    ),
  reviewMedia: (
    runId: string,
    nodeId: string,
    request: {
      expectedRevision: number;
      action: CanvasMediaReviewAction;
      note: string;
      providerTaskId?: string;
      scope: GenerationScope;
    },
  ): Promise<CanvasMediaExecution> =>
    http.post(
      `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/media/review`,
      request,
    ),
  plan: (id: string, request: CanvasRunPlanRequest): Promise<CanvasRunPlan> =>
    http.post(`${path(id)}/run-plans`, request),
  start: (id: string, request: CanvasRunStart): Promise<CanvasRun> =>
    http.post(`${path(id)}/runs`, request),
  list: (
    id: string,
    page = 0,
    signal?: AbortSignal,
  ): Promise<CanvasRunSummary[]> =>
    http.get(`${path(id)}/runs`, { params: { page }, signal }),
  get: (id: string, signal?: AbortSignal): Promise<CanvasRun> =>
    http.get(`/api/canvas-runs/${encodeURIComponent(id)}`, { signal }),
  cancel: (id: string, scope: GenerationScope): Promise<CanvasRun> =>
    http.post(`/api/canvas-runs/${encodeURIComponent(id)}/cancel`, { scope }),
  select: (
    id: string,
    node: CanvasNodeRun,
    outputId: string,
    scope: GenerationScope,
  ): Promise<CanvasRun> =>
    http.post(
      `/api/canvas-runs/${encodeURIComponent(id)}/nodes/${encodeURIComponent(node.id)}/select`,
      { expectedRevision: node.revision, outputId, scope },
    ),
  withdraw: (
    id: string,
    requestId: string,
    scope: GenerationScope,
  ): Promise<CanvasRun | null> =>
    http.post(
      `${path(id)}/run-requests/${encodeURIComponent(requestId)}/withdraw`,
      { scope },
    ),
};
