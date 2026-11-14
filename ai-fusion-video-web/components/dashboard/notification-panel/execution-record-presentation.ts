import type { ExecutionRecordDetail } from "@/lib/api/execution-records";
import { generationPhaseLabels } from "@/lib/generation-execution-view";

export function executionRecordDuration(data: ExecutionRecordDetail) {
  const end = data.task.localFinished ? Date.parse(data.task.updatedAt) : Date.now();
  const seconds = Math.max(0, Math.floor((end - Date.parse(data.task.createdAt)) / 1000));
  if (seconds < 60) {
    return `${seconds} 秒`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes} 分 ${seconds % 60} 秒`;
  }
  return `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
}

export function executionRecordStage(data: ExecutionRecordDetail) {
  const generation = data.generation;
  if (generation?.resultReceived && data.task.status === "FAILED") {
    return "结果保存失败";
  }
  if (generation && generation.phase !== "FINISHED") {
    return generationPhaseLabels[generation.phase] ?? "—";
  }
  return data.task.localFinished ? "已结束" : "处理中";
}
