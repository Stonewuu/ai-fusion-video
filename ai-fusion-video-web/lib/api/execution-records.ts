import { http } from "./client";

export type ExecutionRecordKind = "GENERATION" | "TEXT" | "CANVAS_MEDIA" | "COMFY_TEST" | "EPISODE_AUDIO" | "AGENT_MODEL" | "MCP";

export interface ExecutionRecordRef {
  kind: ExecutionRecordKind;
  id: string;
}

export interface ExecutionRecordSummary extends ExecutionRecordRef {
  title: string;
  status: string;
  localFinished: boolean;
  remoteUnknown: boolean;
  message: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ExecutionRecordDetail {
  task: ExecutionRecordSummary;
  projectId: number | null;
  canvasId: string | null;
  capacityHeld: boolean;
  remote: Array<{
    step: string;
    attemptId: string;
    providerTaskId: string | null;
    acknowledged: boolean;
    responseRecorded: boolean;
  }>;
  history: Array<{ adminUserId: number; note: string; createdAt: string }>;
}

export const executionRecordApi = {
  list: (page: number, signal?: AbortSignal): Promise<{ items: ExecutionRecordSummary[]; hasMore: boolean; page: number }> =>
    http.get("/api/execution-records", { params: { page }, signal }),
  get: (target: ExecutionRecordRef, signal?: AbortSignal): Promise<ExecutionRecordDetail> =>
    http.get(`/api/execution-records/${target.kind}/${encodeURIComponent(target.id)}`, { signal }),
};

export function executionRecordStatus(record: ExecutionRecordSummary) {
  if (record.remoteUnknown) {
    return "远端结果未知";
  }
  if (!record.localFinished) {
    return "正在自动处理";
  }
  if (record.status === "CANCELLED") {
    return "已取消";
  }
  if (record.status === "SUCCEEDED") {
    return "已完成";
  }
  if (record.status === "CONFLICT") {
    return "内容版本已变化";
  }
  if (record.status === "UNKNOWN_SUBMISSION") {
    return "远端响应已记录";
  }
  return record.status === "PARTIAL_FAILED" ? "部分失败" : "执行失败";
}
