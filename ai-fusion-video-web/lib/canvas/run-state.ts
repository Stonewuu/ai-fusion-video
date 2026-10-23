import type {
  CanvasRunBudget,
  CanvasRunMode,
  CanvasRunStart,
  CanvasRunState,
  CanvasRunCost,
} from "@/lib/api/canvas-runs";
import type { GenerationScope } from "@/lib/api/generation-execution";

export const runLabels: Record<CanvasRunState, string> = {
  QUEUED: "等待启动",
  WAITING_INPUT: "等待前置步骤",
  READY: "准备执行",
  RUNNING: "执行中",
  WAITING_SELECTION: "请选择结果",
  WAITING_CONFIRMATION: "等待确认",
  UNKNOWN_SUBMISSION: "调用状态待确认",
  NEEDS_REVIEW: "需要处理",
  CANCEL_REQUESTED: "正在停止",
  SUCCEEDED: "已完成",
  PARTIAL_FAILED: "部分未完成",
  FAILED: "执行失败",
  BLOCKED: "前置步骤未完成",
  CANCELLED: "已停止",
};
export const runModes: { value: CanvasRunMode; label: string }[] = [
  { value: "DEPENDENCIES", label: "所选节点及所需的前置步骤" },
  { value: "SELECTION", label: "仅运行所选节点" },
  { value: "SINGLE", label: "仅运行当前节点" },
  { value: "ALL", label: "运行整张画布" },
  { value: "STALE", label: "更新需要重做的节点及前置步骤" },
];
export const defaultRunBudget: CanvasRunBudget = {
  maxNodes: 128,
  maxImages: 32,
  maxVideoSeconds: 300,
  maxParallel: 2,
  maxBatchItems: 128,
};
export function runFinished(state: CanvasRunState) {
  return [
    "SUCCEEDED",
    "PARTIAL_FAILED",
    "FAILED",
    "CANCELLED",
    "BLOCKED",
  ].includes(state);
}
export interface PendingCanvasRun {
  version: 1;
  canvasId: string;
  title: string;
  cost: CanvasRunCost;
  request: CanvasRunStart;
  createdAt: string;
}
export function runRequestKey(canvasId: string, scope: GenerationScope) {
  return `canvas-run-request:v1:${scope.userId}:${scope.teamId ?? "personal"}:${canvasId}`;
}
export function parseRunRequest(
  raw: string,
  canvasId: string,
  scope: GenerationScope,
): PendingCanvasRun {
  const item = JSON.parse(raw) as Partial<PendingCanvasRun>;
  if (
    item.version !== 1 ||
    item.canvasId !== canvasId ||
    typeof item.title !== "string" ||
    typeof item.createdAt !== "string" ||
    !Number.isFinite(Date.parse(item.createdAt)) ||
    !item.request ||
    !/^[A-Za-z0-9:._-]{1,96}$/.test(item.request.requestId) ||
    !/^[a-f0-9-]{36}$/.test(item.request.planId) ||
    !/^[a-f0-9]{64}$/.test(item.request.planHash) ||
    item.request.scope?.userId !== scope.userId ||
    item.request.scope.teamId !== scope.teamId ||
    !item.cost ||
    ![
      item.cost.nodes,
      item.cost.images,
      item.cost.videoSeconds,
      item.cost.batchItems,
    ].every((value) => Number.isSafeInteger(value) && value >= 0)
  )
    throw new Error(
      "本地运行请求已损坏或作用域不符，请核对运行记录后清除此记录",
    );
  return item as PendingCanvasRun;
}
