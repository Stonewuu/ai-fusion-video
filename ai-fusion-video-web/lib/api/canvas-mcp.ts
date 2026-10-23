import { http } from "./client";
import type { GenerationScope } from "./generation-execution";
import type { CanvasRunState } from "./canvas-runs";

export interface McpJsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  properties?: Record<string, McpJsonSchema>;
  required?: string[];
  enum?: unknown[];
  [key: string]: unknown;
}
export type McpEffect = "READ_ONLY" | "WRITE" | "DESTRUCTIVE";
export type McpIdempotency = "READ_ONLY" | "KEYED" | "NO_REPLAY";
export type McpOutputMode = "TEXT" | "STRUCTURED_JSON" | "JSON_TEXT";
export interface McpWorkflowTool {
  serverName: string; toolName: string; description: string | null;
  inputSchema: McpJsonSchema; outputSchema: McpJsonSchema | null;
  readOnly: boolean; idempotent: boolean; destructive: boolean;
  configurationHash: string; schemaHash: string;
}
export interface McpDefinition {
  tool: McpWorkflowTool; effect: McpEffect; idempotency: McpIdempotency;
  idempotencyParameter: string; outputMode: McpOutputMode; outputSchema: McpJsonSchema; timeoutSeconds: number;
}
export interface McpContract {
  id: string; userId: number; name: string; version: number; enabled: boolean; definition: McpDefinition; hash: string;
}
export interface McpContractSave {
  id?: string; expectedVersion?: number; name: string; toolName: string; schemaHash: string; configurationHash: string;
  effect: McpEffect; idempotency: McpIdempotency; idempotencyParameter: string;
  outputMode: McpOutputMode; outputSchema: McpJsonSchema; timeoutSeconds: number; enabled: boolean;
}
export type McpReviewAction = "USE_RECEIVED" | "CONFIRM_FINISHED";
export interface McpExecution {
  id: string; name: string; serverName: string; toolName: string; effect: McpEffect; idempotency: McpIdempotency;
  status: CanvasRunState; error: string | null; revision: number; arguments: Record<string, unknown>;
  received: { valid: boolean; output?: unknown; response?: unknown; message?: string; responseHash: string } | null;
  remoteCapacityHeld: boolean; cancelRequested: boolean; actions: McpReviewAction[];
  history: { id: string; action: McpReviewAction; note: string; userId: number }[];
}
export const mcpEffectNames: Record<McpEffect, string> = { READ_ONLY: "只读", WRITE: "修改外部内容", DESTRUCTIVE: "可能删除或覆盖外部内容" };
export const mcpIdempotencyNames: Record<McpIdempotency, string> = { READ_ONLY: "只读操作", KEYED: "使用幂等键", NO_REPLAY: "不重放原请求" };
export const canvasMcpApi = {
  tools: (): Promise<McpWorkflowTool[]> => http.get("/api/canvas-mcp/tools"),
  list: (): Promise<McpContract[]> => http.get("/api/canvas-mcp/contracts"),
  save: (request: McpContractSave): Promise<McpContract> => http.post("/api/canvas-mcp/contracts", request),
  remove: (id: string, expectedVersion: number): Promise<boolean> => http.delete(`/api/canvas-mcp/contracts/${id}`, { params: { expectedVersion } }),
  execution: (runId: string, nodeId: string): Promise<McpExecution> => http.get(`/api/canvas-runs/${runId}/nodes/${nodeId}/mcp`),
  review: (runId: string, nodeId: string, request: {
    requestId: string; expectedRevision: number; action: McpReviewAction; note: string; providerFinished: boolean; scope: GenerationScope;
  }): Promise<McpExecution> => http.post(`/api/canvas-runs/${runId}/nodes/${nodeId}/mcp/review`, request),
};
