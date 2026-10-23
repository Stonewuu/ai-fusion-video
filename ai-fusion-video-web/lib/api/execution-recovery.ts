import { http } from "./client";

export type RecoveryKind =
  | "GENERATION"
  | "TEXT"
  | "CANVAS_MEDIA"
  | "COMFY_TEST"
  | "EPISODE_AUDIO"
  | "AGENT_MODEL"
  | "MCP";
export interface RecoveryExecution {
  kind: RecoveryKind;
  id: string;
  status: string;
  revision: number;
  actorUserId: number;
  actorTeamId: number | null;
  projectId: number | null;
  canvasId: string | null;
  capacityHeld: boolean;
  cancelRequested: boolean;
  remote: {
    step: string;
    attemptId: string;
    providerTaskId: string | null;
    acknowledged: boolean;
    responseRecorded: boolean;
  }[];
  factsHash: string;
  canResolve: boolean;
  reason: string;
}
export interface RecoveryDecision {
  adminUserId: number;
  requestId: string;
  expectedRevision: number;
  expectedFactsHash: string;
  note: string;
  providerFinished: boolean;
}
export interface RecoveryReceipt {
  decisionId: string;
  kind: RecoveryKind;
  jobId: string;
  status: string;
  revision: number;
}
export interface RecoveryHistory {
  decisionId: string;
  adminUserId: number;
  kind: RecoveryKind;
  jobId: string;
  actorUserId: number;
  note: string;
  createdAt: string;
}
export const recoveryKindNames: Record<RecoveryKind, string> = {
  GENERATION: "图片 / 视频生成",
  TEXT: "文本生成",
  CANVAS_MEDIA: "画布音频 / 媒体",
  COMFY_TEST: "ComfyUI 试运行",
  EPISODE_AUDIO: "分集配乐 / 音效",
  AGENT_MODEL: "智能体模型调用",
  MCP: "MCP 工具调用",
};
export const executionRecoveryApi = {
  list: (
    page: number,
    signal?: AbortSignal,
  ): Promise<{ items: RecoveryExecution[]; hasMore: boolean; page: number }> =>
    http.get("/api/system/execution-recovery", { params: { page }, signal }),
  history: (page: number, signal?: AbortSignal): Promise<RecoveryHistory[]> =>
    http.get("/api/system/execution-recovery/history", {
      params: { page },
      signal,
    }),
  get: (
    target: Pick<RecoveryExecution, "kind" | "id">,
    signal?: AbortSignal,
  ): Promise<RecoveryExecution> =>
    http.get(`/api/system/execution-recovery/${target.kind}/${target.id}`, {
      signal,
    }),
  confirm: (
    target: Pick<RecoveryExecution, "kind" | "id">,
    decision: RecoveryDecision,
  ): Promise<RecoveryReceipt> =>
    http.post(
      `/api/system/execution-recovery/${target.kind}/${target.id}/confirm-finished`,
      decision,
    ),
};
