import type { GenerationReviewAction } from "./api/generation-execution";

export const generationStateLabels: Record<string, string> = {
  QUEUED: "排队中",
  READY: "等待执行",
  RUNNING: "处理中",
  WAITING: "等待供应商",
  CANCEL_REQUESTED: "取消中",
  UNKNOWN_SUBMISSION: "受理情况待核实",
  NEEDS_REVIEW: "需要核实",
  SUCCEEDED: "已完成",
  PARTIAL_FAILED: "部分完成",
  FAILED: "生成失败",
  CANCELLED: "已取消",
};
export const generationPhaseLabels: Record<string, string> = {
  NEW: "等待准备",
  PREPARING: "准备素材",
  SUBMITTING: "确认提交",
  POLLING: "查询生成进度",
  INGESTING: "接收生成结果",
  FINISHED: "已结束",
};
export const generationReviewLabels: Record<GenerationReviewAction, string> = {
  RECHECK: "重新查询",
  RETRY_INGEST: "继续接收结果",
  RESTORE_ACCEPTED_RECEIPT: "恢复已受理任务",
  CONFIRM_NOT_ACCEPTED_AND_RETRY: "确认未受理并重新提交",
  CONFIRM_PROVIDER_FINISHED: "确认供应商已结束",
};
export const generationTerminalStates = new Set([
  "SUCCEEDED",
  "PARTIAL_FAILED",
  "FAILED",
  "CANCELLED",
]);
