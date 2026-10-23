import { http } from "./client";
import type { PendingToolCallInfo } from "./ai-pipeline";
import type { GenerationExecution } from "./generation-execution";

export interface CanvasAgentTask {
  id: string;
  parentId: string | null;
  name: string;
  status: string;
  error: string | null;
  summary: string | null;
  summaryTruncated: boolean;
  startedAt: string;
  confirmation: null | {
    replyId: string;
    expiresAt: string;
    tools: Array<PendingToolCallInfo & { arguments: string }>;
  };
}
export interface CanvasAgentExecution {
  nodeRunId: string;
  conversationId: string;
  canControl: boolean;
  maxAgentRuns: number;
  unresolvedModelCalls: number;
  unresolvedModelRequestIds: string[];
  tasks: CanvasAgentTask[];
  generations: Pick<
    GenerationExecution,
    "id" | "kind" | "state" | "modelName"
  >[];
}
const path = (runId: string, nodeId: string) =>
  `/api/canvas-runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/agent`;
export const canvasAgentApi = {
  get: (
    runId: string,
    nodeId: string,
    signal?: AbortSignal,
  ): Promise<CanvasAgentExecution> => http.get(path(runId, nodeId), { signal }),
  confirm: (
    runId: string,
    nodeId: string,
    request: {
      runId: string;
      replyId: string;
      decisions: Array<{ toolCallId: string; approved: boolean }>;
    },
  ): Promise<boolean> => http.post(`${path(runId, nodeId)}/confirm`, request),
};
